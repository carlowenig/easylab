from __future__ import annotations
from .data import *


@overload
def data() -> Data:
    ...


@overload
def data(input: DataLike, /) -> Data:
    ...


@overload
def data(
    file_path: str,
    column_vars: Iterable[Var | None | Literal["infer"] | EllipsisType] | None = None,
    **kwargs,
) -> Data:
    ...


def data(*args, **kwargs) -> Data:
    if len(args) == 0 and len(kwargs) == 0:
        return ListData([])
    elif len(args) >= 1 and isinstance(args[0], str):
        return load_data(*args, **kwargs)
    elif len(args) == 1 and len(kwargs) == 0:
        return Data.interpret(*args, **kwargs)
    else:
        raise TypeError("Invalid arguments for data().")
