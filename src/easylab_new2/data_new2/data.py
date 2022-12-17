from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
import inspect
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Sized,
    SupportsIndex,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import final, Self
import dateutil.parser

import pandas as pd

T = TypeVar("T")

I = TypeVar("I", bound=Union[str, int])


D = TypeVar("D", bound="Data")


class Args(Generic[I, T]):
    __slots__ = ("args", "kwargs")

    @overload
    def __new__(cls: type[Self], *args: T) -> Args[int, T]:
        ...

    @overload
    def __new__(cls: type[Self], **kwargs: T) -> Args[str, T]:
        ...

    @overload
    def __new__(cls: type[Self], *args: T, **kwargs: T) -> Args[int | str, T]:
        ...

    def __new__(cls: type[Self], *args: T, **kwargs: T) -> Args[Any, T]:
        return super().__new__(cls, *args, **kwargs)

    def __init_subclass__(cls) -> None:
        raise TypeError("Args cannot be subclassed")

    def __init__(self, *args: T, **kwargs: T):
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def untyped(*args, **kwargs) -> Args[int | str, object]:
        return Args(*args, **kwargs)

    @overload
    @staticmethod
    def compact() -> None:
        ...

    @overload
    @staticmethod
    def compact(value: T, /) -> T:
        ...

    @overload
    @staticmethod
    def compact(arg1: T, arg2: T, *args: T) -> Args[int, T]:
        ...

    @overload
    @staticmethod
    def compact(arg1: T, arg2: T, *args: T, **kwargs: T) -> Args[int | str, T]:
        ...

    @overload
    @staticmethod
    def compact(**kwargs: T) -> Args[str, T]:
        ...

    @overload
    @staticmethod
    def compact(*args, **kwargs) -> Args[int | str, object]:
        ...

    @staticmethod
    def compact(*args, **kwargs) -> object:
        if len(args) == 0 and len(kwargs) == 0:
            return None
        elif len(args) == 1 and len(kwargs) == 0:
            return args[0]
        else:
            return Args(*args, **kwargs)

    @property
    def has_args(self) -> bool:
        return len(self.args) > 0

    @property
    def has_kwargs(self) -> bool:
        return len(self.kwargs) > 0

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.args), len(self.kwargs)

    def __getitem__(self, index: I) -> object:
        if isinstance(index, int):
            return self.args[index]
        elif isinstance(index, str):
            return self.kwargs[index]
        else:
            raise TypeError(f"Invalid key type: {type(index)}")

    def __repr__(self) -> str:
        args_str = ", ".join(f"{arg!r}" for arg in self.args)
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        sep = ", " if len(args_str) > 0 and len(kwargs_str) > 0 else ""
        return f"Args({args_str}{sep}{kwargs_str})"

    def __str__(self) -> str:
        args_str = ", ".join(f"{arg}" for arg in self.args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        sep = ", " if len(args_str) > 0 and len(kwargs_str) > 0 else ""
        return f"({args_str}{sep}{kwargs_str})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Args)
            and self.args == other.args
            and self.kwargs == other.kwargs
        )

    def __hash__(self) -> int:
        return hash((self.args, self.kwargs))

    def __len__(self) -> int:
        return len(self.args) + len(self.kwargs)

    def __iter__(self) -> Iterator[tuple[int | str, object]]:
        yield from enumerate(self.args)
        yield from self.kwargs.items()


class Data(ABC):
    extensions: list[type[Data]] = []
    extension_options: list[str]

    _extension_instances: list[Data]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if not hasattr(cls, "extension_options"):
            cls.extension_options = []
            params = list(inspect.signature(cls.__init__).parameters.values())

            for param in params[1:]:  # Skip self
                if param.kind not in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    cls.extension_options.append(
                        param.name
                        + ("?" if param.default is not inspect.Parameter.empty else "")
                    )

    def __init__(self, **options):
        self._owner: Data | None = None
        self.is_extension: bool = False
        self._init_extensions(**options)

    def _set_owner(self, owner: Data | None):
        self._owner = owner

    @property
    def owner(self) -> Data:
        if self._owner is None:
            return RootData()
        else:
            return self._owner

    def _init_extensions(self, **all_options):
        self._extension_instances: list[Data] = []
        for extension in type(self).extensions:
            ext_options = {}

            for option in extension.extension_options:
                required = True
                if option[-1] == "?":
                    option = option[:-1]
                    required = False

                if option in all_options:
                    ext_options[option] = all_options[option]
                elif extension.type_name() + "_" + option in all_options:
                    ext_options[option] = all_options[
                        extension.type_name() + "_" + option
                    ]
                elif not required:
                    ext_options[option] = None
                else:
                    raise TypeError(
                        f"Missing required option {option!r} for extension {extension.type_name()}"
                    )

            instance = extension._new_extension(**ext_options)
            instance._set_owner(self)
            instance.is_extension = True
            self._extension_instances.append(instance)

    @classmethod
    def _new_extension(cls, **options) -> Self:
        return cls(**options)

    @classmethod
    def type_name(cls) -> str:
        return cls.__name__

    @staticmethod
    def interpret(input: object, owner: Data | None = None) -> Data:
        if input is None:
            data = EmptyData()
        elif isinstance(input, Data):
            data = input
        elif isinstance(input, Sized) and len(input) == 0:
            data = EmptyData()
            # TODO: Add support for other types
        elif isinstance(input, list):
            data = ListData(input)
        elif isinstance(input, dict):
            data = DictData(input)
        else:
            data = ValueData(input)

        if owner is not None:
            data._set_owner(owner)
        return data

    def _matches(self, pattern: object) -> bool:
        return False

    def _get_subdata(self, selector: object) -> Data | None:
        pass

    def _set_subdata(self, selector: object, data: Data) -> bool:
        return False

    def _set_self(self, data: Data) -> bool:
        return False

    def _delete_subdata(self, selector: object) -> bool:
        return False

    def _delete_self(self) -> bool:
        return False

    def _get_timestamp(self) -> datetime | None:
        pass

    def _get_size(self) -> int:
        return 0

    def simplify(self) -> Any:
        return self

    @property
    def timestamp(self) -> datetime | None:
        t = self._get_timestamp()
        if t is not None:
            return t

        for extension in self._extension_instances:
            t = extension.timestamp
            if t is not None:
                return t

    @property
    @final
    def size(self) -> int:
        return self._get_size() + sum(
            extension.size for extension in self._extension_instances
        )

    @property
    @final
    def is_empty(self) -> bool:
        return self.size == 0

    @final
    def matches(self, *args: object, **kwargs: object) -> bool:
        """Checks if the data matches the given pattern."""
        pattern = Args.compact(*args, **kwargs)

        if self._matches(pattern):
            return True

        for extension in self._extension_instances:
            if extension.matches(pattern):
                return True

        return False

    @final
    def get_subdata(self, selector: object) -> Data:
        data = self._get_subdata(selector)

        if data is None:
            for extension in self._extension_instances:
                data = extension.get_subdata(selector)
                if data is not None:
                    break

        if data is None:
            data = EmptyData()

        data._set_owner(self)
        return data

    @final
    def __getitem__(self, item: object) -> Data:
        if isinstance(item, tuple):
            item = Args(*item)

        return self.get_subdata(item)

    @final
    def set_subdata(self, selector: object, data: Data) -> bool:
        if self._set_subdata(selector, data):
            return True

        for extension in self._extension_instances:
            if extension.set_subdata(selector, data):
                return True

        return False

    @final
    def __setitem__(self, selector: object, value: object) -> None:
        if isinstance(selector, tuple):
            selector = Args(*selector)

        self.set_subdata(selector, Data.interpret(value))

    @final
    def delete_subdata(self, selector: object) -> bool:
        if self._delete_subdata(selector):
            return True

        for extension in self._extension_instances:
            if extension.delete_subdata(selector):
                return True

        return False

    @final
    def __delitem__(self, item: object) -> None:
        if isinstance(item, tuple):
            item = Args(*item)

        self.delete_subdata(item)

    @final
    def unpack(self, *selectors: object):
        return tuple(self[selector] for selector in selectors)

    @final
    def update(self, selector: object, f: Callable[[Any], Any] | None = None):
        if f is None:
            return self.update("*", selector)  # type: ignore

        self[selector] = f(self[selector])
        return self

    def __repr__(self) -> str:
        return f"{self.type_name()}()"

    def __str__(self) -> str:
        return f"{self.type_name()}()"

    def __len__(self) -> int:
        return self.size


class SupportsWildcard(Data):
    def _matches(self, pattern: object) -> bool:
        if pattern == "*":
            return True
        return False

    def _get_subdata(self, selector: object) -> Data | None:
        if selector == "*":
            return self

    def _set_subdata(self, selector: object, data: Data) -> bool:
        if selector == "*":
            return self._set_self(data)
        return False

    def _delete_subdata(self, selector: object) -> bool:
        if selector == "*":
            return self._delete_self()
        return False


class HasTimestamp(Data):
    def __init__(self, timestamp: datetime | None = None) -> None:
        if timestamp is None:
            self._timestamp = datetime.now()
        elif isinstance(timestamp, datetime):
            self._timestamp = timestamp
        elif isinstance(timestamp, (int, float)):
            self._timestamp = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            self._timestamp = dateutil.parser.parse(timestamp)
        else:
            raise TypeError(f"Cannot parse timestamp {timestamp!r}")
        super().__init__()

    def _matches(self, pattern: object) -> bool:
        if self._timestamp is None:
            return False
        elif isinstance(pattern, datetime):
            return self._timestamp == pattern
        elif isinstance(pattern, Args) and pattern.shape == (2, 0):
            op, date = pattern.args
            if op in ["=", "=="]:
                return self._timestamp == date
            elif op == ">":
                return self._timestamp > date
            elif op == ">=":
                return self._timestamp >= date
            elif op == "<":
                return self._timestamp < date
            elif op == "<=":
                return self._timestamp <= date
            elif op == "!=":
                return self._timestamp != date
            else:
                raise ValueError(f"Unknown operator {op!r}")
        else:
            return False

    def _get_timestamp(self) -> datetime | None:
        return self._timestamp


class RootData(Data):
    extensions = [HasTimestamp]

    __instance: RootData | None = None

    def __new__(cls) -> RootData:
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    def _set_owner(self, owner: Data):
        raise RuntimeError("Cannot set owner of RootData")

    def __repr__(self) -> str:
        return "RootData()"

    def __str__(self) -> str:
        return "RootData()"


class ValueData(Data, Generic[T]):
    extensions = [HasTimestamp]

    def __init__(self, value: T, **options) -> None:
        self.value: T = value
        super().__init__(**options)

    def _matches(self, pattern: object) -> bool:
        return self.value == pattern

    def _get_subdata(self, selector: object) -> Data | None:
        if isinstance(selector, str) and hasattr(self.value, selector):
            return Data.interpret(getattr(self.value, selector))

        try:
            return Data.interpret(self.value[selector])  # type: ignore
        except (KeyError, IndexError):
            pass

    def _set_subdata(self, selector: object, data: Data) -> bool:
        if not isinstance(data, ValueData):
            return False

        if isinstance(selector, str) and hasattr(self.value, selector):
            setattr(self.value, selector, data.value)
            return True

        try:
            self.value[selector] = data.value  # type: ignore
            return True
        except (KeyError, IndexError):
            return False

    def _set_self(self, data: Data) -> bool:
        self.value = data.simplify()
        return True

    def _delete_subdata(self, selector: object) -> bool:
        if isinstance(selector, str) and hasattr(self.value, selector):
            delattr(self.value, selector)
            return True

        try:
            del self.value[selector]  # type: ignore
            return True
        except (KeyError, IndexError):
            return False

    def _get_size(self) -> int:
        return 1

    def simplify(self) -> T:
        return self.value

    def __repr__(self) -> str:
        return f"{self.type_name()}({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)


class EmptyData(Data):
    extensions = [HasTimestamp]

    def _matches(self, pattern: object) -> bool:
        return False

    def _get_subdata(self, selector: object) -> Data:
        return EmptyData()

    def _set_subdata(self, selector: object, data: Data) -> bool:
        return False

    def _delete_subdata(self, selector: object) -> bool:
        return False

    def _get_timestamp(self) -> datetime | None:
        return None

    def _get_size(self) -> int:
        return 0

    def simplify(self) -> None:
        return None


class Var(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name

    def matches(self, pattern: object):
        return pattern == self or pattern == self.name


class ListData(Data, Generic[T]):
    extensions = [HasTimestamp]

    def __init__(self, list_: list[T], **options) -> None:
        self.list = list_
        super().__init__(**options)

    def _matches(self, pattern: object) -> bool:
        if self.list == pattern:
            return True
        elif isinstance(pattern, Args) and pattern.shape == (2, 0):
            op, value = pattern.args
            if op == "contains":
                return value in self.list
            else:
                raise ValueError(f"Unknown operator {op!r}")
        else:
            return False

    def _get_subdata(self, selector: object) -> Data:
        if isinstance(selector, slice):
            return ListData(self.list[selector])
        elif isinstance(selector, int):
            return Data.interpret(self.list[selector])
        else:
            raise TypeError(f"Cannot index list with {selector!r}")

    def _set_subdata(self, selector: object, data: Data) -> bool:
        if not isinstance(data, ValueData):
            return False

        if isinstance(selector, SupportsIndex):
            self.list[selector] = data.value
            return True
        else:
            raise TypeError(f"Cannot index list with {selector!r}")

    def _set_self(self, data: Data) -> bool:
        if isinstance(data, ListData):
            self.list = data.list
            return True
        else:
            return False

    def _delete_subdata(self, selector: object) -> bool:
        if isinstance(selector, SupportsIndex):
            del self.list[selector]
            return True
        else:
            raise TypeError(f"Cannot index list with {selector!r}")

    def _delete_self(self) -> bool:
        self.list = []
        return True

    def _get_size(self) -> int:
        return len(self.list)

    def simplify(self) -> list[T]:
        return self.list

    def __repr__(self) -> str:
        return f"{self.type_name()}({self.list!r})"

    def __str__(self) -> str:
        return str(self.list)


K = TypeVar("K")
V = TypeVar("V")


class DictData(Data, Generic[K, V]):
    extensions = [HasTimestamp]

    def __init__(self, dict_: dict[K, V], **options) -> None:
        self.dict = dict_
        super().__init__(**options)

    def _matches(self, pattern: object) -> bool:
        return self.dict == pattern

    def _get_subdata(self, selector: object) -> Data:
        return Data.interpret(self.dict.get(selector, None))  # type: ignore

    def _set_subdata(self, selector: object, data: Data) -> bool:
        self.dict[selector] = data  # type: ignore
        return True

    def _set_self(self, data: Data) -> bool:
        if isinstance(data, DictData):
            self.dict = data.dict
            return True
        else:
            return False

    def _delete_subdata(self, selector: object) -> bool:
        if selector in self.dict:
            del self.dict[selector]  # type: ignore
            return True
        else:
            return False

    def _delete_self(self) -> bool:
        self.dict = {}
        return True

    def _get_size(self) -> int:
        return len(self.dict)

    def __repr__(self) -> str:
        return f"{self.type_name()}({self.dict!r})"

    def __str__(self) -> str:
        return str(self.dict)

    def simplify(self) -> dict[K, V]:
        return self.dict


class Record(Data):
    extensions = [HasTimestamp, SupportsWildcard]

    def __init__(self, dict_: dict[Var, object], **options) -> None:
        self.dict = dict_
        super().__init__(**options)

    def _matches(self, pattern: object) -> bool:
        return self.dict == pattern

    def _get_var(self, input: object):
        if isinstance(input, Var):
            return input

        for var in self.dict:
            if var.matches(input):
                return var

    def _get_subdata(self, selector: object) -> Data:
        var = self._get_var(selector)
        if var is None:
            return EmptyData()
        else:
            return Data.interpret(self.dict[var])

    def _set_subdata(self, selector: object, data: Data) -> bool:
        var = self._get_var(selector)
        if var is None:
            return False
        else:
            self.dict[var] = data
            return True

    def _set_self(self, data: Data) -> bool:
        if isinstance(data, Record):
            self.dict = data.dict
            return True
        else:
            return False

    def _delete_subdata(self, selector: object) -> bool:
        var = self._get_var(selector)
        if var is None:
            return False
        else:
            del self.dict[var]
            return True

    def _delete_self(self) -> bool:
        self.dict = {}
        return True

    def _get_size(self) -> int:
        return len(self.dict)

    def __repr__(self) -> str:
        return f"{self.type_name()}({self.dict!r})"

    def __str__(self) -> str:
        return str(self.dict)
