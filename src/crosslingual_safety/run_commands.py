"""Typer registration for the stable four-option ``run`` facade."""

from pathlib import Path
from typing import Annotated, Any, cast

import typer

from crosslingual_safety.unified_run import (
    PUBLIC_JAILBREAKS,
    PUBLIC_LANGUAGES,
    PUBLIC_SOURCES,
    RunRequest,
    execute_run,
    load_run_settings,
    parse_selection,
    plan_run,
)


def register_run_commands(app: typer.Typer) -> None:
    @app.command("run")
    def run_command(
        source: Annotated[str, typer.Option("--source", help="manual or bench")] = "manual",
        language: Annotated[
            str,
            typer.Option("--language", help="Language code, comma-separated, or all"),
        ] = "all",
        jailbreak: Annotated[
            str,
            typer.Option("--jailbreak", help="none, gra, psa, comma-separated, or all"),
        ] = "none",
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Print the deterministic plan without side effects"),
        ] = False,
    ) -> None:
        if source not in PUBLIC_SOURCES:
            raise typer.BadParameter(
                f"--source must be one of: {', '.join(PUBLIC_SOURCES)}", param_hint="--source"
            )
        try:
            languages = parse_selection(language, PUBLIC_LANGUAGES, "--language")
            jailbreaks = parse_selection(jailbreak, PUBLIC_JAILBREAKS, "--jailbreak")
            request = RunRequest(
                source=cast(Any, source),
                languages=languages,
                jailbreaks=jailbreaks,
                dry_run=dry_run,
            )
            settings = load_run_settings(Path("configs/run.yaml"))
            plan = plan_run(request, settings)
        except (ValueError, OSError) as error:
            raise typer.BadParameter(str(error)) from None

        if dry_run:
            typer.echo(
                f"cases={len(plan.cases)} translations={plan.translation_jobs} "
                f"psa_summaries={plan.psa_summary_count} "
                f"victim_requests={plan.victim_request_count} "
                f"run_id={plan.run_id} path={plan.parent_path}"
            )
            return
        execute_run(plan, settings)


__all__ = ["register_run_commands"]
