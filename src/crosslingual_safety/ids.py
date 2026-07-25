import hashlib
import unicodedata


def stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def canonicalize_text(value: str) -> str:
    """Normalize only reversible transport differences used by the raw contract."""
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()
