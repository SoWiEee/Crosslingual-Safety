import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from crosslingual_safety.cli import app
from crosslingual_safety.manual import load_manual_prompts
from crosslingual_safety.manual_commands import _preparation_lock

runner = CliRunner()
DEFAULT_MODELS = (
    "llama31_8b",
    "gemma_4_12b",
    "gemma_4_26b",
    "nemotron_cascade_2_30b",
    "llama33_70b",
)


def _write_fake_models(tmp_path: Path) -> Path:
    names = (*DEFAULT_MODELS, "nemotron_3_ultra_550b", "llama_guard_3_8b")
    config = {
        "models": {
            name: {
                "provider": "fake",
                "model_id": f"fake/{name}",
                "endpoint_type": "chat",
                "concurrency": 4,
                "requests_per_minute": 6000,
                "test_only": True,
                "fake_response": f"response from {name}",
            }
            for name in names
        }
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_txt_input_has_stable_raw_snapshot(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt.txt"
    input_path.write_bytes(b"  A manual prompt.\r\n")

    batch = load_manual_prompts(input_path, source_language="en")

    assert batch.input_sha256 == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert [prompt.model_dump(mode="json") for prompt in batch.prompts] == [
        {
            "prompt_id": "cccb29211cc277a26738",
            "prompt": "A manual prompt.",
            "source_language": "en",
            "role": None,
            "category": None,
            "system_prompt": None,
        }
    ]
    assert batch.snapshot_jsonl == (
        '{"category":null,"prompt":"A manual prompt.","prompt_id":"cccb29211cc277a26738",'
        '"role":null,"source_language":"en","system_prompt":null}\n'
    )


def test_txt_requires_explicit_source_language(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt.txt"
    input_path.write_text("Prompt", encoding="utf-8")

    with pytest.raises(ValueError, match="source_language is required"):
        load_manual_prompts(input_path)


def test_jsonl_is_strict_and_preserves_per_prompt_role(tmp_path: Path) -> None:
    input_path = tmp_path / "prompts.jsonl"
    rows = [
        {
            "prompt_id": "p-1",
            "prompt": "English prompt",
            "source_language": "en",
            "role": "riddler",
        },
        {
            "prompt_id": "p-2",
            "prompt": "中文提示",
            "source_language": "zh",
            "category": "test",
            "system_prompt": "System",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    batch = load_manual_prompts(input_path)

    assert [prompt.prompt_id for prompt in batch.prompts] == ["p-1", "p-2"]
    assert batch.prompts[0].role == "riddler"
    assert batch.prompts[1].role is None
    assert "中文提示" in batch.snapshot_jsonl


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"prompt_id": "same", "prompt": "One", "source_language": "en"},
                {"prompt_id": "same", "prompt": "Two", "source_language": "en"},
            ],
            "duplicate prompt_id",
        ),
        (
            [{"prompt_id": "p", "prompt": "One", "source_language": "fr"}],
            "source_language",
        ),
        (
            [
                {
                    "prompt_id": "p",
                    "prompt": "One",
                    "source_language": "en",
                    "unexpected": True,
                }
            ],
            "unexpected",
        ),
    ],
)
def test_jsonl_rejects_invalid_contract(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises((ValueError, ValidationError), match=message):
        load_manual_prompts(input_path)


def test_manual_input_rejects_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "prompts.csv"
    input_path.write_text("prompt_id,prompt\np,Prompt\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.txt or \.jsonl"):
        load_manual_prompts(input_path, source_language="en")


def test_manual_run_builds_four_by_five_matrix_and_resumes(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt.txt"
    input_path.write_text("Test prompt", encoding="utf-8")
    models_path = _write_fake_models(tmp_path)
    runs_dir = tmp_path / "runs"
    arguments = [
        "manual-run",
        str(input_path),
        "--source-language",
        "en",
        "--translator",
        "fake",
        "--jailbreak",
        "gra_v1",
        "--role",
        "riddler",
        "--models-config",
        str(models_path),
        "--runs-dir",
        str(runs_dir),
    ]

    first = runner.invoke(app, arguments)
    second = runner.invoke(app, arguments)

    assert first.exit_code == second.exit_code == 0, first.output
    assert "planned_jobs=20 processed_jobs=20" in first.output
    assert "planned_jobs=20 processed_jobs=0" in second.output
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    required = {
        "input_snapshot.jsonl",
        "translations.jsonl",
        "variants.jsonl",
        "results.jsonl",
        "report.md",
        "run_manifest.json",
    }
    assert required <= {path.name for path in run_dir.iterdir()}
    translations = [
        json.loads(line)
        for line in (run_dir / "translations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    variants = [
        json.loads(line)
        for line in (run_dir / "variants.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(translations) == len(variants) == 4
    assert len(results) == 20
    assert {row["language"] for row in variants} == {"en", "zh", "vi", "my"}
    assert {row["model_name"] for row in results} == set(DEFAULT_MODELS)
    assert {json.loads(row["attack_metadata_json"])["role_id"] for row in variants} == {"riddler"}
    manifest_text = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    assert "API_KEY" not in manifest_text
    assert "secret" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["contract"]["generation"]["max_tokens"] == 4096
    assert manifest["template_sha256s"]
    assert manifest["catalog_sha256s"]
    assert "response from llama31_8b" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_jsonl_role_overrides_cli_role(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "prompt_id": "p-1",
                "prompt": "Test prompt",
                "source_language": "en",
                "role": "lex_luthor",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "manual-run",
            str(input_path),
            "--translator",
            "fake",
            "--jailbreak",
            "gra_v1",
            "--role",
            "scarecrow",
            "--models-config",
            str(_write_fake_models(tmp_path)),
            "--models",
            "llama31_8b",
            "--runs-dir",
            str(runs_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    variants_path = next(runs_dir.iterdir()) / "variants.jsonl"
    variants = [json.loads(line) for line in variants_path.read_text(encoding="utf-8").splitlines()]
    assert {json.loads(variant["attack_metadata_json"])["role_id"] for variant in variants} == {
        "lex_luthor"
    }


def test_ultra_is_opt_in_addition_to_default_models(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt.txt"
    input_path.write_text("Test prompt", encoding="utf-8")
    runs_dir = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "manual-run",
            str(input_path),
            "--source-language",
            "en",
            "--translator",
            "fake",
            "--models-config",
            str(_write_fake_models(tmp_path)),
            "--add-model",
            "nemotron_3_ultra_550b",
            "--runs-dir",
            str(runs_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    results_path = next(runs_dir.iterdir()) / "results.jsonl"
    results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(results) == 24
    assert {row["model_name"] for row in results} == {
        *DEFAULT_MODELS,
        "nemotron_3_ultra_550b",
    }


def test_txt_source_language_changes_run_identity(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt.txt"
    input_path.write_text("Test prompt", encoding="utf-8")
    runs_dir = tmp_path / "runs"
    common = [
        "manual-run",
        str(input_path),
        "--translator",
        "fake",
        "--models-config",
        str(_write_fake_models(tmp_path)),
        "--models",
        "llama31_8b",
        "--runs-dir",
        str(runs_dir),
    ]

    english = runner.invoke(app, [*common, "--source-language", "en"])
    chinese = runner.invoke(app, [*common, "--source-language", "zh"])

    assert english.exit_code == chinese.exit_code == 0
    assert len(list(runs_dir.iterdir())) == 2


def test_gra_config_change_changes_run_identity(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt.txt"
    input_path.write_text("Test prompt", encoding="utf-8")
    config_path = tmp_path / "jailbreaks.yaml"
    config = yaml.safe_load(Path("configs/jailbreaks.yaml").read_text(encoding="utf-8"))
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs"
    arguments = [
        "manual-run",
        str(input_path),
        "--source-language",
        "en",
        "--translator",
        "fake",
        "--jailbreak",
        "gra_v1",
        "--models-config",
        str(_write_fake_models(tmp_path)),
        "--models",
        "llama31_8b",
        "--jailbreaks-config",
        str(config_path),
        "--runs-dir",
        str(runs_dir),
    ]

    first = runner.invoke(app, arguments)
    config["personas"]["gra_v1"]["joker"]["expertise"]["en"] = "changed expertise"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    second = runner.invoke(app, arguments)

    assert first.exit_code == second.exit_code == 0
    assert len(list(runs_dir.iterdir())) == 2


def test_add_model_only_accepts_ultra(tmp_path: Path) -> None:
    input_path = tmp_path / "prompt.txt"
    input_path.write_text("Test prompt", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "manual-run",
            str(input_path),
            "--source-language",
            "en",
            "--translator",
            "fake",
            "--models-config",
            str(_write_fake_models(tmp_path)),
            "--add-model",
            "llama_guard_3_8b",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code != 0
    assert "--add-model only supports nemotron_3_ultra_550b" in result.output


def test_preparation_lock_serializes_same_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first_entered = threading.Event()
    release_first = threading.Event()
    timeline: list[str] = []

    def first() -> None:
        with _preparation_lock(run_dir):
            timeline.append("first")
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second() -> None:
        assert first_entered.wait(timeout=2)
        with _preparation_lock(run_dir):
            timeline.append("second")

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=2)
    time.sleep(0.1)
    assert timeline == ["first"]
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert timeline == ["first", "second"]
