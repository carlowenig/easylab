from typing import Callable, Optional, cast
from varname import varname

_unnamed_indices: dict[type, int] = {}


class AutoNamed:
    _name: str

    _name_find_depth = 0

    @property
    def name(self) -> str:
        if hasattr(self, "_name"):
            return self._name
        else:
            raise ValueError(
                f"Name of {type(self)} was not set. Please call __autoset_name__ inside your __init__ method."
            )

    def __init_auto_named__(
        self,
        name: Optional[str] = None,
        *,
        fallback: Optional[Callable[[], Optional[str]]] = None,
        find_name_depth_offset: int = 0,
        auto_name: bool = True,
    ):
        def get_fallback():
            fallback_result = fallback() if fallback is not None else None
            if fallback_result is not None:
                return fallback_result
            else:
                t = type(self)
                index = _unnamed_indices.get(t, 0)
                _unnamed_indices[t] = index + 1
                return f"unnamed_{t.__name__}_{index}"

        if name is not None:
            self._name = name
            return

        if not auto_name:
            self._name = get_fallback()
            return

        try:
            self._name = cast(
                str,
                varname(
                    frame=1 + type(self)._name_find_depth + find_name_depth_offset,
                    strict=False,
                ),
            )
        except Exception as e:
            self._name = get_fallback()
            # print(
            #     f"[i] Could not infer name of {type(self).__name__}. Using {self._name}."
            # )
            # print(f"    Exception was: {e}")

    def __init_subclass__(cls) -> None:
        cls._name_find_depth += 1
