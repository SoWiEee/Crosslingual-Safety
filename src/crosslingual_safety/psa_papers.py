"""Versioned PDF sources for formal Paper Summary Attack conditions."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader


class PsaPaperSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str
    summary_id: str
    title: str
    source_path: Path
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summarizer_model: str = "ais3/gemma-4-12b"


class PaperChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_index: int
    page_number: int
    text: str
    text_sha256: str
    word_count: int = Field(gt=0, le=1000)


class ExtractedPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str
    title: str
    source_path: str
    source_sha256: str
    text_sha256: str
    page_count: int = Field(gt=0)
    chunks: tuple[PaperChunk, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(chunk.text for chunk in self.chunks)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_pdf_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def load_psa_papers(path: Path) -> dict[str, PsaPaperSpec]:
    if not path.is_file():
        raise ValueError(f"PSA paper configuration does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    papers = raw.get("papers") if isinstance(raw, dict) else None
    if not isinstance(papers, dict) or not papers:
        raise ValueError("PSA paper configuration must contain papers")
    result: dict[str, PsaPaperSpec] = {}
    for condition_id, value in papers.items():
        if not isinstance(value, dict):
            raise ValueError(f"invalid PSA paper configuration: {condition_id}")
        spec = PsaPaperSpec.model_validate({"condition_id": str(condition_id), **value})
        if spec.condition_id in result:
            raise ValueError(f"duplicate PSA paper condition: {spec.condition_id}")
        result[spec.condition_id] = spec
    return result


def extract_paper(spec: PsaPaperSpec, *, max_chunk_words: int = 1000) -> ExtractedPaper:
    if max_chunk_words <= 0 or max_chunk_words > 1000:
        raise ValueError("PDF chunk word limit must be between 1 and 1000")
    if not spec.source_path.is_file():
        raise ValueError(f"PSA source PDF does not exist: {spec.source_path}")
    source_sha256 = sha256_file(spec.source_path)
    if source_sha256 != spec.expected_sha256:
        raise ValueError(f"PSA source PDF hash mismatch: {spec.condition_id}")
    try:
        reader = PdfReader(spec.source_path)
    except Exception as error:
        raise ValueError(f"PSA source PDF cannot be read: {spec.condition_id}") from error
    if reader.is_encrypted:
        raise ValueError(f"PSA source PDF must not be encrypted: {spec.condition_id}")

    chunks: list[PaperChunk] = []
    normalized_pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = _normalized_pdf_text(page.extract_text() or "")
        except Exception as error:
            raise ValueError(
                f"PSA source PDF text extraction failed: {spec.condition_id}"
            ) from error
        if not page_text:
            continue
        normalized_pages.append(page_text)
        words = page_text.split()
        for start in range(0, len(words), max_chunk_words):
            text = " ".join(words[start : start + max_chunk_words])
            chunks.append(
                PaperChunk(
                    chunk_index=len(chunks),
                    page_number=page_number,
                    text=text,
                    text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    word_count=len(text.split()),
                )
            )
    if not chunks:
        raise ValueError(f"PSA source PDF contains no extractable text: {spec.condition_id}")
    normalized_text = "\n\n".join(normalized_pages)
    return ExtractedPaper(
        condition_id=spec.condition_id,
        title=spec.title,
        source_path=spec.source_path.as_posix(),
        source_sha256=source_sha256,
        text_sha256=hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
        page_count=len(reader.pages),
        chunks=tuple(chunks),
    )
