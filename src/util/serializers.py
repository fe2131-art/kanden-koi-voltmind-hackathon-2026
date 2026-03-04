"""Serialization utilities for Safety View Agent."""

from typing import Any


def serialize_pydantic_or_dict(obj: Any) -> Any:
    """Serialize Pydantic model or dict-like object to serializable format.

    Args:
        obj: Object to serialize (Pydantic model, dict, or other)

    Returns:
        Serialized object (dict or string)
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
