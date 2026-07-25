from typing import Literal

from pydantic import BaseModel, Field

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


ReviewStatus = Literal["pending", "accepted", "rejected", "needs_revision"]


class TranslationRecord(BaseModel):
    translation_id: str
    case_id: str
    source_language: str
    target_language: str
    source_text: str
    raw_translated_text: str
    normalized_translated_text: str
    method: str
    translator_id: str
    translator_version: str
    decoding_config: dict[str, object]
    source_text_sha256: str
    translated_text_sha256: str
    provider_request_id: str | None
    provider_cache_key: str
    candidate_set: str
    revision_id: str | None = None
    frozen: bool
    created_at: str
    review_status: ReviewStatus


class TranslationReview(BaseModel):
    review_id: str
    translation_id: str
    reviewer_id: str
    rubric_version: str
    intent_preserved: Literal["yes", "no", "uncertain"]
    requested_action_preserved: Literal["yes", "no", "uncertain"]
    target_preserved: Literal["yes", "no", "uncertain"]
    constraints_preserved: Literal["yes", "no", "uncertain"]
    fluency: Literal["good", "acceptable", "poor", "uncertain"]
    ambiguity_added: Literal["yes", "no", "uncertain"]
    notes: str | None
    created_at: str


class TranslationManifest(BaseModel):
    experiment_id: str
    created_at: str
    translation_hashes: dict[str, str]
    provider_language_support: dict[str, list[str]] = Field(default_factory=dict)
