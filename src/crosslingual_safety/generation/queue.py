import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crosslingual_safety.schemas import GenerationRequest


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class JobQueue:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                claimed_at TEXT,
                completed_at TEXT,
                error_type TEXT,
                error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS requests (
                run_id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES jobs(run_id)
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def enqueue(self, jobs: list[tuple[str, GenerationRequest]]) -> int:
        before = self.connection.total_changes
        with self.connection:
            for model_name, request in jobs:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO jobs
                    (run_id, experiment_id, variant_id, model_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        request.run_id,
                        request.experiment_id,
                        request.variant_id,
                        model_name,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO requests (run_id, request_json)
                    VALUES (?, ?)
                    """,
                    (request.run_id, request.model_dump_json()),
                )
        return (self.connection.total_changes - before) // 2

    def reset_stale(self, lease_seconds: int = 900) -> int:
        cutoff = (
            (datetime.now(UTC) - timedelta(seconds=lease_seconds))
            .isoformat()
            .replace("+00:00", "Z")
        )
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE jobs
                SET status = 'pending', claimed_at = NULL
                WHERE status = 'running' AND claimed_at < ?
                """,
                (cutoff,),
            )
        return cursor.rowcount

    def pending(self, experiment_id: str) -> list[tuple[str, str, int, GenerationRequest]]:
        rows = self.connection.execute(
            """
            SELECT jobs.run_id, jobs.model_id, jobs.attempts, requests.request_json
            FROM jobs JOIN requests USING (run_id)
            WHERE jobs.experiment_id = ? AND jobs.status = 'pending'
            ORDER BY jobs.run_id
            """,
            (experiment_id,),
        ).fetchall()
        return [
            (
                run_id,
                model_id,
                int(attempts),
                GenerationRequest.model_validate_json(request_json),
            )
            for run_id, model_id, attempts, request_json in rows
        ]

    def claim(self, run_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE jobs
                SET status = 'running', claimed_at = ?
                WHERE run_id = ? AND status = 'pending'
                """,
                (_now(), run_id),
            )
        return cursor.rowcount == 1

    def complete(
        self,
        run_id: str,
        status: str,
        total_attempts: int,
        expected_attempts: int,
        error_type: str | None,
        error_message: str | None,
    ) -> bool:
        job_status = self._job_status(status)
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE jobs
                SET status = ?, attempts = ?, completed_at = ?,
                    error_type = ?, error_message = ?
                WHERE run_id = ? AND status = 'running' AND attempts = ?
                """,
                (
                    job_status,
                    total_attempts,
                    _now(),
                    error_type,
                    error_message,
                    run_id,
                    expected_attempts,
                ),
            )
        return cursor.rowcount == 1

    def reconcile(
        self,
        run_id: str,
        status: str,
        total_attempts: int,
        error_type: str | None,
        error_message: str | None,
    ) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE jobs
                SET status = ?, attempts = ?, completed_at = ?,
                    error_type = ?, error_message = ?
                WHERE run_id = ? AND status IN ('pending', 'running') AND attempts < ?
                """,
                (
                    self._job_status(status),
                    total_attempts,
                    _now(),
                    error_type,
                    error_message,
                    run_id,
                    total_attempts,
                ),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _job_status(status: str) -> str:
        if status in {"success", "provider_blocked"}:
            return status
        if status in {"rate_limited", "timeout", "server_error", "empty_response"}:
            return "retryable_error"
        return "permanent_error"

    def status_counts(self, experiment_id: str) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) FROM jobs WHERE experiment_id = ? GROUP BY status",
            (experiment_id,),
        ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def retry_failed(self, experiment_id: str) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE jobs
                SET status = 'pending', claimed_at = NULL, completed_at = NULL
                WHERE experiment_id = ? AND status = 'retryable_error'
                """,
                (experiment_id,),
            )
        return cursor.rowcount
