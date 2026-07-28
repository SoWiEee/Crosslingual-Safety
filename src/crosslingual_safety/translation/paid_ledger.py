from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

from crosslingual_safety.ids import stable_id
from crosslingual_safety.translation.providers import (
    GoogleCloudAuthenticationError,
    GoogleCloudIndeterminatePaidAttemptError,
    GoogleCloudInvalidRequestError,
    GoogleCloudNMTTranslator,
    GoogleCloudPermissionError,
    GoogleCloudQuotaError,
    GoogleCloudTranslationResponseError,
    ProviderTranslation,
)

OutcomeStatus = Literal["success", "failed", "indeterminate"]

INDETERMINATE_ERROR_TYPE = "GoogleCloudIndeterminatePaidAttemptError"
INDETERMINATE_ERROR_MESSAGE = (
    "Google Cloud Translation paid attempt outcome is indeterminate; manual review is required"
)
GENERIC_PROVIDER_ERROR_TYPE = "GoogleCloudProviderError"
GENERIC_PROVIDER_ERROR_MESSAGE = "Google Cloud Translation provider request failed"
PROVEN_PREPROCESSING_REJECTIONS = (
    GoogleCloudAuthenticationError,
    GoogleCloudPermissionError,
    GoogleCloudQuotaError,
    GoogleCloudInvalidRequestError,
)
RESERVATION_FIELDS = frozenset(
    {
        "reservation_id",
        "task_key",
        "attempt_number",
        "case_id",
        "source_language",
        "target_language",
        "provider",
        "provider_contract_sha256",
        "source_text_sha256",
        "source_character_count",
        "created_at",
    }
)
OUTCOME_FIELDS = frozenset(
    {
        "outcome_id",
        "reservation_id",
        "task_key",
        "status",
        "charged_character_count",
        "audit_reference",
        "created_at",
        "translated_text",
        "provider_request_id",
        "error_type",
        "error_message",
    }
)


class PaidCallFailure(TypedDict):
    error_type: str
    error_message: str
    charged_character_count: int
    audit_reference: str


class PaidCallLedgerError(ValueError):
    """A fixed-message failure in the local paid-call audit ledger."""


def is_proven_preprocessing_rejection(error: BaseException) -> bool:
    """Return true only for an explicit provider rejection that proves no processing."""

    return type(error) in PROVEN_PREPROCESSING_REJECTIONS


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows: list[dict[str, object]] = []
        for line in lines:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PaidCallLedgerError("paid-call ledger contains an invalid row")
            rows.append(cast(dict[str, object], value))
        return rows
    except PaidCallLedgerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise PaidCallLedgerError("paid-call ledger could not be read safely") from None


def _append_immutable_jsonl(
    path: Path,
    rows: list[Mapping[str, object]],
    *,
    key: str,
) -> None:
    try:
        existing = _read_jsonl(path)
        by_key: dict[str, dict[str, object]] = {}
        for existing_row in existing:
            row_key = existing_row.get(key)
            if not isinstance(row_key, str) or not row_key:
                raise PaidCallLedgerError("paid-call ledger contains an invalid identity")
            prior = by_key.get(row_key)
            if prior is not None and _canonical_json(prior) != _canonical_json(existing_row):
                raise PaidCallLedgerError("paid-call ledger contains an immutable conflict")
            by_key[row_key] = existing_row
        for row in rows:
            row_dict = dict(row)
            row_key = row_dict.get(key)
            if not isinstance(row_key, str) or not row_key:
                raise PaidCallLedgerError("paid-call ledger row identity is invalid")
            prior = by_key.get(row_key)
            if prior is not None and _canonical_json(prior) != _canonical_json(row_dict):
                raise PaidCallLedgerError("paid-call ledger contains an immutable conflict")
            by_key[row_key] = row_dict
        content = "".join(
            _canonical_json(row) + "\n"
            for row in sorted(by_key.values(), key=lambda value: cast(str, value[key]))
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if os.name != "nt":
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory = os.open(path.parent, flags)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except PaidCallLedgerError:
        raise
    except (OSError, TypeError, ValueError):
        raise PaidCallLedgerError("paid-call ledger could not be persisted safely") from None


@dataclass(frozen=True)
class PaidTranslationTask:
    task_key: str
    case_id: str
    source_language: str
    target_language: str
    source_text_sha256: str
    provider: str
    provider_contract_sha256: str
    source_character_count: int

    @classmethod
    def build(
        cls,
        *,
        case_id: str,
        source_text: str,
        source_language: str,
        target_language: str,
        provider: str,
        provider_contract: Mapping[str, object],
    ) -> PaidTranslationTask:
        source_text_sha256 = _sha256_text(source_text)
        provider_contract_sha256 = _sha256_text(_canonical_json(dict(provider_contract)))
        task_key = stable_id(
            "paid-translation-task",
            case_id,
            source_text_sha256,
            source_language,
            target_language,
            provider,
            provider_contract_sha256,
        )
        return cls(
            task_key=task_key,
            case_id=case_id,
            source_language=source_language,
            target_language=target_language,
            source_text_sha256=source_text_sha256,
            provider=provider,
            provider_contract_sha256=provider_contract_sha256,
            source_character_count=len(source_text),
        )


class PaidCallLedger:
    """Immutable paid-call records used to recover from ordinary process crashes."""

    def __init__(self, audit_dir: Path) -> None:
        self.reservations_path = audit_dir / "translation_reservations.jsonl"
        self.outcomes_path = audit_dir / "translation_reservation_outcomes.jsonl"

    def reservations(self) -> list[dict[str, object]]:
        rows = _read_jsonl(self.reservations_path)
        seen: set[str] = set()
        for row in rows:
            reservation_id, _ = self.validate_reservation_identity(row)
            if reservation_id in seen:
                raise PaidCallLedgerError("paid-call ledger contains a duplicate reservation")
            seen.add(reservation_id)
        return rows

    def outcomes(self) -> list[dict[str, object]]:
        reservations = {str(row["reservation_id"]): row for row in self.reservations()}
        rows = _read_jsonl(self.outcomes_path)
        seen_outcomes: set[str] = set()
        seen_reservations: set[str] = set()
        for row in rows:
            reservation_id = row.get("reservation_id")
            reservation = (
                reservations.get(reservation_id) if isinstance(reservation_id, str) else None
            )
            if reservation is None:
                raise PaidCallLedgerError("paid-call outcome reservation is invalid")
            outcome_id = self.validate_outcome_for_reservation(row, reservation)
            validated_reservation_id = str(reservation["reservation_id"])
            if outcome_id in seen_outcomes or validated_reservation_id in seen_reservations:
                raise PaidCallLedgerError("paid-call ledger contains a duplicate outcome")
            seen_outcomes.add(outcome_id)
            seen_reservations.add(validated_reservation_id)
        return rows

    def append_reservation(self, reservation: Mapping[str, object]) -> None:
        self.reservations()
        self.validate_reservation_identity(reservation)
        _append_immutable_jsonl(
            self.reservations_path,
            [reservation],
            key="reservation_id",
        )

    def append_outcome(self, outcome: Mapping[str, object]) -> None:
        existing_outcomes = self.outcomes()
        reservation_id = outcome.get("reservation_id")
        reservation = next(
            (row for row in self.reservations() if row.get("reservation_id") == reservation_id),
            None,
        )
        if reservation is None:
            raise PaidCallLedgerError("paid-call outcome reservation is invalid")
        outcome_id = self.validate_outcome_for_reservation(outcome, reservation)
        if any(row.get("outcome_id") == outcome_id for row in existing_outcomes):
            prior = next(row for row in existing_outcomes if row.get("outcome_id") == outcome_id)
            if _canonical_json(prior) != _canonical_json(dict(outcome)):
                raise PaidCallLedgerError("paid-call ledger contains an immutable conflict")
        _append_immutable_jsonl(
            self.outcomes_path,
            [outcome],
            key="outcome_id",
        )

    def charged_characters(self) -> int:
        total = 0
        for reservation in self.reservations():
            self.validate_reservation_identity(reservation)
            character_count = reservation.get("source_character_count")
            assert isinstance(character_count, int) and not isinstance(character_count, bool)
            total += character_count
        return total

    @staticmethod
    def validate_reservation_identity(
        reservation: Mapping[str, object],
    ) -> tuple[str, int]:
        if set(reservation) != RESERVATION_FIELDS:
            raise PaidCallLedgerError("paid-call reservation schema is invalid")
        reservation_id = reservation.get("reservation_id")
        task_key = reservation.get("task_key")
        attempt_number = reservation.get("attempt_number")
        case_id = reservation.get("case_id")
        source_language = reservation.get("source_language")
        target_language = reservation.get("target_language")
        provider = reservation.get("provider")
        provider_contract_sha256 = reservation.get("provider_contract_sha256")
        source_text_sha256 = reservation.get("source_text_sha256")
        character_count = reservation.get("source_character_count")
        created_at = reservation.get("created_at")
        if (
            not isinstance(reservation_id, str)
            or not isinstance(task_key, str)
            or isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number <= 0
            or not isinstance(case_id, str)
            or not case_id
            or not isinstance(source_language, str)
            or not source_language
            or not isinstance(target_language, str)
            or not target_language
            or not isinstance(provider, str)
            or not provider
            or not isinstance(provider_contract_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", provider_contract_sha256) is None
            or not isinstance(source_text_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_text_sha256) is None
            or isinstance(character_count, bool)
            or not isinstance(character_count, int)
            or character_count < 0
            or not isinstance(created_at, str)
            or not created_at
            or len(created_at) > 64
            or any(character in created_at for character in "\r\n")
        ):
            raise PaidCallLedgerError("paid-call reservation identity is invalid")
        expected_task_key = stable_id(
            "paid-translation-task",
            case_id,
            source_text_sha256,
            source_language,
            target_language,
            provider,
            provider_contract_sha256,
        )
        expected_reservation_id = stable_id(
            "paid-translation-reservation",
            expected_task_key,
            str(attempt_number),
        )
        if task_key != expected_task_key or reservation_id != expected_reservation_id:
            raise PaidCallLedgerError("paid-call reservation identity is invalid")
        return reservation_id, attempt_number

    @classmethod
    def validate_outcome_for_reservation(
        cls,
        outcome: Mapping[str, object],
        reservation: Mapping[str, object],
    ) -> str:
        reservation_id, _ = cls.validate_reservation_identity(reservation)
        if set(outcome) != OUTCOME_FIELDS:
            raise PaidCallLedgerError("paid-call outcome schema is invalid")

        outcome_id = outcome.get("outcome_id")
        outcome_reservation_id = outcome.get("reservation_id")
        if (
            not isinstance(outcome_id, str)
            or outcome_id != stable_id("paid-translation-outcome", reservation_id)
            or outcome_reservation_id != reservation_id
        ):
            raise PaidCallLedgerError("paid-call outcome identity is invalid")

        task_key = reservation["task_key"]
        character_count = reservation["source_character_count"]
        status = outcome.get("status")
        charged_character_count = outcome.get("charged_character_count")
        created_at = outcome.get("created_at")
        if (
            outcome.get("task_key") != task_key
            or status not in {"success", "failed", "indeterminate"}
            or isinstance(charged_character_count, bool)
            or not isinstance(charged_character_count, int)
            or charged_character_count != character_count
            or outcome.get("audit_reference") != f"translation_reservations.jsonl#{reservation_id}"
            or not isinstance(created_at, str)
            or not created_at
            or len(created_at) > 64
            or any(character in created_at for character in "\r\n")
        ):
            raise PaidCallLedgerError("paid-call outcome context is invalid")

        translated_text = outcome.get("translated_text")
        provider_request_id = outcome.get("provider_request_id")
        error_type = outcome.get("error_type")
        error_message = outcome.get("error_message")
        if status == "success":
            if (
                not isinstance(translated_text, str)
                or not translated_text.strip()
                or (
                    provider_request_id is not None
                    and (
                        not isinstance(provider_request_id, str)
                        or len(provider_request_id) > 256
                        or re.fullmatch(r"[A-Za-z0-9._=-]+", provider_request_id) is None
                    )
                )
                or error_type is not None
                or error_message is not None
            ):
                raise PaidCallLedgerError("paid-call success outcome is invalid")
        elif status == "failed":
            if (
                translated_text is not None
                or provider_request_id is not None
                or error_type != GENERIC_PROVIDER_ERROR_TYPE
                or error_message != GENERIC_PROVIDER_ERROR_MESSAGE
            ):
                raise PaidCallLedgerError("paid-call failure outcome is invalid")
        elif (
            translated_text is not None
            or provider_request_id is not None
            or error_type != INDETERMINATE_ERROR_TYPE
            or error_message != INDETERMINATE_ERROR_MESSAGE
        ):
            raise PaidCallLedgerError("paid-call indeterminate outcome is invalid")
        return outcome_id

    @classmethod
    def validate_reservation_for_task(
        cls,
        reservation: Mapping[str, object],
        task: PaidTranslationTask,
    ) -> tuple[str, int]:
        reservation_id, attempt_number = cls.validate_reservation_identity(reservation)
        expected = {
            "task_key": task.task_key,
            "case_id": task.case_id,
            "source_language": task.source_language,
            "target_language": task.target_language,
            "provider": task.provider,
            "provider_contract_sha256": task.provider_contract_sha256,
            "source_text_sha256": task.source_text_sha256,
            "source_character_count": task.source_character_count,
        }
        if any(reservation.get(name) != value for name, value in expected.items()):
            raise PaidCallLedgerError("paid-call reservation task identity is invalid")
        return reservation_id, attempt_number

    def next_attempt_number(self, task: PaidTranslationTask) -> int:
        attempts: list[int] = []
        for reservation in self.reservations():
            self.validate_reservation_identity(reservation)
            if reservation.get("task_key") != task.task_key:
                continue
            _, attempt_number = self.validate_reservation_for_task(reservation, task)
            attempts.append(attempt_number)
        return max(attempts, default=0) + 1

    def unresolved_for_task(self, task: PaidTranslationTask) -> list[dict[str, object]]:
        completed = {
            outcome.get("reservation_id")
            for outcome in self.outcomes()
            if isinstance(outcome.get("reservation_id"), str)
        }
        unresolved: list[dict[str, object]] = []
        for reservation in self.reservations():
            self.validate_reservation_identity(reservation)
            if reservation.get("task_key") != task.task_key:
                continue
            reservation_id, _ = self.validate_reservation_for_task(reservation, task)
            if reservation_id not in completed:
                unresolved.append(reservation)
        return unresolved

    def latest_outcome_for_task(self, task: PaidTranslationTask) -> dict[str, object] | None:
        reservations: dict[str, dict[str, object]] = {}
        for reservation in self.reservations():
            self.validate_reservation_identity(reservation)
            if reservation.get("task_key") != task.task_key:
                continue
            reservation_id, _ = self.validate_reservation_for_task(reservation, task)
            reservations[reservation_id] = reservation
        candidates = [
            outcome for outcome in self.outcomes() if outcome.get("reservation_id") in reservations
        ]
        if not candidates:
            return None

        def attempt_number(outcome: Mapping[str, object]) -> int:
            outcome_reservation_id = outcome.get("reservation_id")
            reservation = (
                reservations.get(outcome_reservation_id, {})
                if isinstance(outcome_reservation_id, str)
                else {}
            )
            value = reservation.get("attempt_number")
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        return max(candidates, key=attempt_number)

    def make_reservation(
        self,
        task: PaidTranslationTask,
        *,
        character_count: int,
        clock: Callable[[], str] = _utc_now,
    ) -> dict[str, object]:
        if character_count != task.source_character_count:
            raise PaidCallLedgerError("paid-call reservation context is invalid")
        attempt_number = self.next_attempt_number(task)
        reservation_id = stable_id(
            "paid-translation-reservation",
            task.task_key,
            str(attempt_number),
        )
        reservation: dict[str, object] = {
            "reservation_id": reservation_id,
            "task_key": task.task_key,
            "attempt_number": attempt_number,
            "case_id": task.case_id,
            "source_language": task.source_language,
            "target_language": task.target_language,
            "provider": task.provider,
            "provider_contract_sha256": task.provider_contract_sha256,
            "source_text_sha256": task.source_text_sha256,
            "source_character_count": character_count,
            "created_at": clock(),
        }
        self.append_reservation(reservation)
        return reservation

    def record_success(
        self,
        reservation: Mapping[str, object],
        output: ProviderTranslation,
        *,
        clock: Callable[[], str] = _utc_now,
    ) -> dict[str, object]:
        outcome = self._base_outcome(reservation, "success", clock)
        outcome.update(
            {
                "translated_text": output.text,
                "provider_request_id": output.provider_request_id,
                "error_type": None,
                "error_message": None,
            }
        )
        self.append_outcome(outcome)
        return outcome

    def record_failure(
        self,
        reservation: Mapping[str, object],
        *,
        clock: Callable[[], str] = _utc_now,
    ) -> dict[str, object]:
        outcome = self._base_outcome(reservation, "failed", clock)
        outcome.update(
            {
                "translated_text": None,
                "provider_request_id": None,
                "error_type": GENERIC_PROVIDER_ERROR_TYPE,
                "error_message": GENERIC_PROVIDER_ERROR_MESSAGE,
            }
        )
        self.append_outcome(outcome)
        return outcome

    def project_indeterminate(
        self,
        reservation: Mapping[str, object],
        *,
        clock: Callable[[], str] = _utc_now,
    ) -> dict[str, object]:
        outcome = self._base_outcome(reservation, "indeterminate", clock)
        outcome.update(
            {
                "translated_text": None,
                "provider_request_id": None,
                "error_type": INDETERMINATE_ERROR_TYPE,
                "error_message": INDETERMINATE_ERROR_MESSAGE,
            }
        )
        self.append_outcome(outcome)
        return outcome

    @staticmethod
    def _base_outcome(
        reservation: Mapping[str, object],
        status: OutcomeStatus,
        clock: Callable[[], str],
    ) -> dict[str, object]:
        PaidCallLedger.validate_reservation_identity(reservation)
        reservation_id = reservation.get("reservation_id")
        task_key = reservation.get("task_key")
        character_count = reservation.get("source_character_count")
        if (
            not isinstance(reservation_id, str)
            or not reservation_id
            or not isinstance(task_key, str)
            or not task_key
            or isinstance(character_count, bool)
            or not isinstance(character_count, int)
            or character_count < 0
        ):
            raise PaidCallLedgerError("paid-call reservation outcome context is invalid")
        return {
            "outcome_id": stable_id("paid-translation-outcome", reservation_id),
            "reservation_id": reservation_id,
            "task_key": task_key,
            "status": status,
            "charged_character_count": character_count,
            "audit_reference": f"translation_reservations.jsonl#{reservation_id}",
            "created_at": clock(),
        }


class LedgeredGoogleCloudTranslator:
    """Google translator wrapper that closes the paid-call persistence window."""

    def __init__(
        self,
        translator: GoogleCloudNMTTranslator,
        ledger: PaidCallLedger,
        *,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self.translator = translator
        self.ledger = ledger
        self.clock = clock
        self.translator_id = translator.translator_id
        self.version = translator.version
        self.method = translator.method
        self.decoding_config = dict(translator.decoding_config)
        self.provider_contract = dict(translator.provider_contract)
        self.translator.characters_used = ledger.charged_characters()
        self.translator.paid_call_reservation = self._reserve_paid_call
        self._task: PaidTranslationTask | None = None
        self._active_reservation: dict[str, object] | None = None
        self._current_outcome: dict[str, object] | None = None

    @property
    def characters_used(self) -> int:
        return self.translator.characters_used

    def supports(self, source_language: str, target_language: str) -> bool:
        return self.translator.supports(source_language, target_language)

    def begin_task(
        self,
        *,
        case_id: str,
        source_text: str,
        source_language: str,
        target_language: str,
    ) -> PaidTranslationTask:
        task = PaidTranslationTask.build(
            case_id=case_id,
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            provider=self.translator_id,
            provider_contract=self.provider_contract,
        )
        unresolved = self.ledger.unresolved_for_task(task)
        for reservation in unresolved:
            self.ledger.project_indeterminate(reservation, clock=self.clock)
        self._task = task
        self._active_reservation = None
        self._current_outcome = self.ledger.latest_outcome_for_task(task)
        return task

    def current_failure(self) -> PaidCallFailure | None:
        outcome = self._current_outcome
        if outcome is None:
            return None
        status = outcome.get("status")
        if status not in {"failed", "indeterminate"}:
            return None
        reservation_id = outcome.get("reservation_id")
        character_count = outcome.get("charged_character_count")
        if (
            not isinstance(reservation_id, str)
            or re.fullmatch(r"[0-9a-f]{20}", reservation_id) is None
            or isinstance(character_count, bool)
            or not isinstance(character_count, int)
            or character_count < 0
        ):
            raise PaidCallLedgerError("paid-call outcome context is invalid")
        error_type, error_message = (
            (INDETERMINATE_ERROR_TYPE, INDETERMINATE_ERROR_MESSAGE)
            if status == "indeterminate"
            else (GENERIC_PROVIDER_ERROR_TYPE, GENERIC_PROVIDER_ERROR_MESSAGE)
        )
        return {
            "error_type": error_type,
            "error_message": error_message,
            "charged_character_count": character_count,
            "audit_reference": f"translation_reservations.jsonl#{reservation_id}",
        }

    def _reserve_paid_call(self, character_count: int) -> None:
        task = self._task
        if task is None:
            raise PaidCallLedgerError("paid-call task context is unavailable")
        self._active_reservation = self.ledger.make_reservation(
            task,
            character_count=character_count,
            clock=self.clock,
        )

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> ProviderTranslation:
        task = self._task
        if task is None:
            raise PaidCallLedgerError("paid-call task context is unavailable")
        observed = PaidTranslationTask.build(
            case_id=task.case_id,
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            provider=self.translator_id,
            provider_contract=self.provider_contract,
        )
        if observed.task_key != task.task_key:
            raise PaidCallLedgerError("paid-call task context is invalid")
        prior_outcome = self._current_outcome
        if prior_outcome is not None and prior_outcome.get("status") == "indeterminate":
            raise GoogleCloudIndeterminatePaidAttemptError(INDETERMINATE_ERROR_MESSAGE)
        if prior_outcome is not None and prior_outcome.get("status") == "success":
            translated_text = prior_outcome.get("translated_text")
            provider_request_id = prior_outcome.get("provider_request_id")
            if not isinstance(translated_text, str) or not translated_text.strip():
                raise GoogleCloudTranslationResponseError(
                    "Google Cloud Translation returned an unusable response"
                )
            safe_request_id = (
                provider_request_id
                if isinstance(provider_request_id, str)
                and len(provider_request_id) <= 256
                and re.fullmatch(r"[A-Za-z0-9._=-]+", provider_request_id)
                else None
            )
            return ProviderTranslation(translated_text, safe_request_id)

        self._active_reservation = None
        self._current_outcome = None
        try:
            output = self.translator.translate(text, source_language, target_language)
        except Exception as error:
            if self._active_reservation is not None:
                self._current_outcome = (
                    self.ledger.record_failure(
                        self._active_reservation,
                        clock=self.clock,
                    )
                    if is_proven_preprocessing_rejection(error)
                    else self.ledger.project_indeterminate(
                        self._active_reservation,
                        clock=self.clock,
                    )
                )
            raise
        if self._active_reservation is None:
            raise PaidCallLedgerError("paid-call reservation was not persisted")
        self._current_outcome = self.ledger.record_success(
            self._active_reservation,
            output,
            clock=self.clock,
        )
        return output
