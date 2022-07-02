from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool:
        ...

    def __gt__(self, other: Any) -> bool:
        ...
