from pathlib import Path

import yaml
from pydantic import BaseModel


class LanguageConfig(BaseModel):
    display_name: str
    nllb_code: str
    family: str
    script: str


def load_languages(path: Path) -> dict[str, LanguageConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("languages"), dict):
        raise ValueError(f"invalid language configuration: {path}")
    return {
        code: LanguageConfig.model_validate(config) for code, config in payload["languages"].items()
    }
