import hashlib
import json

from crosslingual_safety.ids import canonicalize_text, stable_id
from crosslingual_safety.schemas import PromptCase, TranslationRecord
from crosslingual_safety.translation.providers import ProviderTranslation, Translator
from crosslingual_safety.translation.storage import TranslationStore, utc_now


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class TranslationService:
    def __init__(self, store: TranslationStore) -> None:
        self.store = store

    def translate_case(
        self,
        case: PromptCase,
        target_language: str,
        translator: Translator,
        candidate_set: str = "default",
        defer_snapshot: bool = False,
        force_retranslate: bool = False,
    ) -> TranslationRecord:
        if not translator.supports(case.source_language, target_language):
            raise ValueError(
                f"translator {translator.translator_id} does not support "
                f"{case.source_language}->{target_language}"
            )
        config_json = json.dumps(translator.decoding_config, sort_keys=True, separators=(",", ":"))
        revision_id = utc_now() if force_retranslate else None
        base_provider_cache_key = stable_id(
            case.canonical_payload,
            case.source_language,
            target_language,
            translator.translator_id,
            translator.version,
            config_json,
        )
        provider_cache_key = (
            stable_id(base_provider_cache_key, revision_id)
            if revision_id is not None
            else base_provider_cache_key
        )
        translation_id = stable_id(
            case.case_id,
            provider_cache_key,
            candidate_set,
            revision_id or "",
        )
        cached = self.store.get(translation_id)
        if cached is not None:
            return cached
        if any(record.frozen for record in self.store.find(case.case_id, target_language)):
            raise ValueError(f"cannot translate frozen case {case.case_id} to {target_language}")

        cached_response = self.store.get_provider_response(provider_cache_key)
        if cached_response is None:
            output = translator.translate(
                case.canonical_payload,
                case.source_language,
                target_language,
            )
            self.store.add_provider_response(
                provider_cache_key,
                output.text,
                output.provider_request_id,
            )
        else:
            output = ProviderTranslation(*cached_response)
        normalized = canonicalize_text(output.text)
        record = TranslationRecord(
            translation_id=translation_id,
            case_id=case.case_id,
            source_language=case.source_language,
            target_language=target_language,
            source_text=case.canonical_payload,
            raw_translated_text=output.text,
            normalized_translated_text=normalized,
            method=translator.method,
            translator_id=translator.translator_id,
            translator_version=translator.version,
            decoding_config=translator.decoding_config,
            source_text_sha256=_sha256(case.canonical_payload),
            translated_text_sha256=_sha256(normalized),
            provider_request_id=output.provider_request_id,
            provider_cache_key=provider_cache_key,
            candidate_set=candidate_set,
            revision_id=revision_id,
            frozen=False,
            created_at=utc_now(),
            review_status="pending",
        )
        return self.store.add_translation(record, defer_snapshot=defer_snapshot)

    def add_human_revision(
        self,
        parent: TranslationRecord,
        revised_text: str,
        reviewer_id: str,
        rubric_version: str,
    ) -> TranslationRecord:
        normalized = canonicalize_text(revised_text)
        revision_id = stable_id(
            parent.translation_id,
            normalized,
            reviewer_id,
            rubric_version,
        )
        translator_id = (
            parent.translator_id
            if parent.translator_id.endswith("+human_revision")
            else f"{parent.translator_id}+human_revision"
        )
        record = TranslationRecord(
            translation_id=stable_id(parent.translation_id, revision_id, translator_id),
            case_id=parent.case_id,
            source_language=parent.source_language,
            target_language=parent.target_language,
            source_text=parent.source_text,
            raw_translated_text=revised_text,
            normalized_translated_text=normalized,
            method="human_revision",
            translator_id=translator_id,
            translator_version=parent.translator_version,
            decoding_config={
                "parent_translation_id": parent.translation_id,
                "reviewer_id": reviewer_id,
                "rubric_version": rubric_version,
            },
            source_text_sha256=parent.source_text_sha256,
            translated_text_sha256=_sha256(normalized),
            provider_request_id=None,
            provider_cache_key=parent.provider_cache_key,
            candidate_set=parent.candidate_set,
            revision_id=revision_id,
            frozen=False,
            created_at=utc_now(),
            review_status="pending",
        )
        return self.store.add_translation(record)
