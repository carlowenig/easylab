from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import getpass
from typing import Any, Union
from ..util import is_wildcard


def get_current_username():
    try:
        return getpass.getuser()
    except OSError:
        return None


MetadataLike = Union["Metadata", dict[str, Any], None]


@dataclass
class Metadata:
    @staticmethod
    def interpret(input: MetadataLike) -> Metadata:
        if input is None:
            return Metadata()
        elif isinstance(input, Metadata):
            return input
        elif isinstance(input, dict):
            return Metadata(**input)
        else:
            raise TypeError(f"Cannot interpret {input} as Metadata.")

    created_at: datetime = field(default_factory=datetime.now)
    created_by: str | None = field(default_factory=get_current_username)
    updated_at: datetime | None = None
    updated_by: str | None = None
    source: str = "defined"

    def update(self, at: datetime | None = None, by: str | None = None):
        self.updated_at = at or datetime.now()
        self.updated_by = by or get_current_username()

    @property
    def was_updated(self):
        return self.updated_at is not None

    def matches(self, query: Any):
        if is_wildcard(query):
            return True

        if isinstance(query, Metadata):
            return self == query

        if isinstance(query, dict):
            return all(getattr(self, key) == value for key, value in query.items())

        if isinstance(query, str) and ":" in query:
            key, value = query.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "created_at":
                # TODO: Implement before/after
                return self.created_at == datetime.fromisoformat(value)
            elif key == "created_by":
                return self.created_by == value
            elif key == "updated_at":
                return self.updated_at == datetime.fromisoformat(value)
            elif key == "updated_by":
                return self.updated_by == value
            elif key == "source":
                return self.source == value

            raise ValueError(
                f"Cannot match Metadata with {query}. Metadata has no attribute {key}."
            )

        raise ValueError(f"Cannot match Metadata with {query}.")
