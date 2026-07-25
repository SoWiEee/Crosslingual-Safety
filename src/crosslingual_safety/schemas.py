from typing import Literal

from pydantic import BaseModel

Intent = Literal["harmful", "benign"]


class PromptCase(BaseModel):
    case_id: str
    content_id: str
    dataset: str
    intent: Intent
    category: str | None
    source_language: str
    source_text: str
    behavior_description: str | None
    success_criteria: str | None
    context_text: str | None
    canonical_payload: str
    payload_format: str
    pair_group_id: str | None = None


class SourceRecord(BaseModel):
    source_record_id: str
    case_id: str
    dataset: str
    source_id: str
    split: str
    source_path: str
    source_row: int
    source_file_sha256: str
    metadata: dict[str, object]


class NativeTranslation(BaseModel):
    translation_id: str
    case_id: str
    language: str
    source_text: str
    translated_text: str
    method: Literal["native_dataset"] = "native_dataset"
    source_record_id: str


class CasePair(BaseModel):
    pair_group_id: str
    pair_kind: Literal["jbb_harmful_benign"]
    source_index: str
    harmful_case_id: str
    benign_case_id: str
    validation_warning: str | None = None


class VariantCaseSelection(BaseModel):
    selection_id: str
    selection_scope: Literal["dataset_exact_content"]
    dataset: str
    content_id: str
    selected_case_id: str
    excluded_case_ids: list[str]
