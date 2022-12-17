from __future__ import annotations
from abc import ABC, abstractmethod
import builtins
from dataclasses import dataclass, field
from datetime import datetime
import sys
from typing import (
    Any,
    Generic,
    Iterable,
    Literal,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    TypeVar,
    Union,
    cast,
    final,
    overload,
)
from typing_extensions import Self, TypeGuard

import dateutil.parser


I = TypeVar("I")
J = TypeVar("J")
T = TypeVar("T")
S = TypeVar("S")


MetadataLike = Union["Metadata", dict[str, Any], None]


@dataclass
class Metadata:
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "code"
    owner: Data | None = None

    @staticmethod
    def interpret(input: MetadataLike) -> Metadata:
        if input is None:
            return Metadata()
        elif isinstance(input, Metadata):
            return input
        elif isinstance(input, dict):
            props = {}

            for key, value in input.items():
                if key == "timestamp":
                    if value is None or isinstance(value, datetime):
                        pass
                    elif isinstance(value, str):
                        value = dateutil.parser.parse(value)
                    else:
                        raise TypeError(f"Invalid type for timestamp: {type(value)}")

                if key in ("timestamp", "source", "owner"):
                    props[key] = value

                if key == "metadata":
                    if isinstance(value, Metadata):
                        props.update(value.__dict__)
                    elif isinstance(value, dict):
                        props.update(value)
                    else:
                        raise TypeError(f"Invalid type for metadata: {type(value)}")

            return Metadata(**props)
        else:
            raise TypeError(f"Cannot interpret metadata from {input!r}")

    def touch(self, source: str | None = None):
        self.timestamp = datetime.now()
        if source is not None:
            self.source = source


class Data(Generic[I]):
    __slots__ = ("_internal", "_dtype", "_metadata", "_props")

    _internal: I
    _dtype: DataType[I]
    _metadata: Metadata
    _props: dict[str, object]

    @overload
    def __new__(cls, *, metadata: MetadataLike = None, **props) -> Data[None]:
        ...

    @overload
    def __new__(cls, input: I, *, metadata: MetadataLike = None, **props) -> Data[I]:
        ...

    @overload
    def __new__(
        cls, input: Data[I], *, metadata: MetadataLike = None, **props
    ) -> Data[I]:
        ...

    @overload
    def __new__(
        cls,
        input: object,
        dtype: DataTypeLike[I],
        metadata: MetadataLike = None,
        **props,
    ) -> Data[I]:
        ...

    @overload
    def __new__(
        cls,
        input: I | object = None,
        dtype: DataTypeLike[I] | Literal["infer"] = "infer",
        metadata: MetadataLike = None,
        **props,
    ) -> Data:
        ...

    def __new__(
        cls,
        input: I | object = None,
        dtype: DataTypeLike[I] | Literal["infer"] = "infer",
        metadata: MetadataLike = None,
        **props,
    ) -> Data:
        _dtype: DataType[Any]

        if dtype == "infer":
            if input is None:
                _dtype = EmptyDataType()
            elif isinstance(input, Data):
                _dtype = input.dtype
            else:
                raise TypeError(f"Cannot infer data type from input {input!r}")
        else:
            _dtype = get_dtype(dtype)

        metadata = Metadata.interpret(metadata)

        if _dtype.accepts_internal(input):
            data: Data[Any] = super().__new__(cls)
            data._init_data(input, _dtype, metadata, **props)
            return data
        else:
            return _dtype.interpret(input, metadata, **props)

    def _init_data(
        self,
        internal: I,
        dtype: DataType[I],
        metadata: Metadata,
        **props,
    ):
        self._internal = internal
        self._dtype = dtype
        self._metadata = metadata
        self._props = props

        for name, value in props.items():
            setattr(self, name, value)

    @property
    def internal(self) -> I:
        return self._internal

    @property
    def dtype(self) -> DataType[I]:
        return self._dtype

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    def touch(self, source: str | None = None):
        self._metadata.touch(source)

    @overload
    def copy(
        self,
        *,
        metadata: MetadataLike | None = None,
        **props,
    ) -> Data[I]:
        ...

    @overload
    def copy(
        self,
        *,
        internal: J | None = None,
        dtype: DataType[J] | None = None,
        metadata: MetadataLike | None = None,
        **props,
    ) -> Data[J]:
        ...

    def copy(
        self,
        *,
        internal: J | None = None,
        dtype: DataType[J] | None = None,
        metadata: MetadataLike | None = None,
        **props,
    ) -> Data:
        return Data(
            internal if internal is not None else self.internal,
            dtype if dtype is not None else self.dtype,
            metadata if metadata is not None else self.metadata,
            **(self._props | props),
        )

    def convert(self, to: DataType[J]) -> Data[J]:
        converted = to.convert_from(self)
        if converted is NotImplemented:
            converted = self._dtype.convert_to(self, to)
        if converted is NotImplemented:
            raise Exception(f"Cannot convert {self} to {to}")

        return self.copy(internal=converted, dtype=to)  # type: ignore # TODO: what's wrong here?

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._internal!r}, dtype={self._dtype}, metadata={self._metadata})"

    def __str__(self) -> str:
        return f"{self.dtype.name}({self.internal})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Data) and self.internal == other.internal

    def __hash__(self) -> int:
        return hash(self.internal)

    def get_subdata(self, selector: object) -> Data:
        result = self._dtype.get_subdata(self, selector)

        if result is NotImplemented:
            if isinstance(selector, str) and hasattr(self.internal, selector):
                return Data(getattr(self.internal, selector), self._dtype)

            try:
                return Data(self.internal[selector], self._dtype)  # type: ignore
            except (KeyError, TypeError):
                return Data()

        return result

    def set_subdata(self, selector: object, value: object) -> bool:
        result = self._dtype.set_subdata(self, selector, value)

        if result is NotImplemented:
            if isinstance(selector, str) and hasattr(self.internal, selector):
                setattr(self.internal, selector, value)
                return True

            try:
                self.internal[selector] = value  # type: ignore
                return True
            except (KeyError, TypeError):
                return False

        return result

    def delete_subdata(self, selector: object) -> bool:
        result = self._dtype.delete_subdata(self, selector)

        if result is NotImplemented:
            if isinstance(selector, str) and hasattr(self.internal, selector):
                delattr(self.internal, selector)
                return True

            try:
                del self.internal[selector]  # type: ignore
                return True
            except (KeyError, TypeError):
                return False

        return result

    def __getitem__(self, selector: object) -> Data:
        return self.get_subdata(selector)

    def __setitem__(self, selector: object, value: object) -> None:
        if not self.set_subdata(selector, value):
            raise Exception(f"Cannot set subdata {selector} of {self} to {value}")

    def __delitem__(self, selector: object) -> None:
        if not self.delete_subdata(selector):
            raise Exception(f"Cannot delete subdata {selector} of {self}")

    def get_prop(self, name: str):
        if name in self._props:
            return self._props[name]

        raise Exception(f"Cannot get property {name!r} of {self}")

    def set_prop(self, name: str, value: object):
        self._props[name] = value


class DataType(ABC, Generic[I]):
    def __init__(self, name: str | None = None, data_cls: type[Data[I]] = Data) -> None:
        self.name = name if name is not None else type(self).__name__
        self.data_cls = data_cls

    @abstractmethod
    def validate_internal(self, internal: object) -> Iterable[str]:
        ...

    def parse(self, input: object) -> I:
        return NotImplemented

    def get_subdata(self, data: Data[I], selector: object) -> Data:
        return NotImplemented

    def set_subdata(self, data: Data[I], selector: object, value: object) -> bool:
        return NotImplemented

    def delete_subdata(self, data: Data[I], selector: object) -> bool:
        return NotImplemented

    def convert_from(self, data: Data) -> Data[I]:
        return NotImplemented

    def convert_to(self, data: Data[I], new_dtype: DataType[J]) -> Data[J]:
        return NotImplemented

    def get_size(self, data: Data[I]) -> int:
        return 1

    def accepts_internal(self, internal: object) -> bool:
        for _ in self.validate_internal(internal):
            return False
        return True

    def check_internal(self, internal: object, msg: str | None):
        reasons = list(self.validate_internal(internal))
        if len(reasons) > 0:
            raise Exception(
                (msg or f"Value {internal!r} is not accepted by {self}. ")
                + " Reasons: "
                + ", ".join(reasons)
            )

    def interpret(
        self,
        input: object,
        metadata: MetadataLike = None,
        **props,
    ) -> Data[I]:
        if self.accepts_internal(input):
            return self.data_cls(input, self, metadata)

        elif isinstance(input, Data):
            data = input.convert(self)

            if metadata is not None:
                data._metadata = Metadata.interpret(metadata)
            data._props.update(props)
        else:
            parsed = self.parse(input)
            if parsed is NotImplemented:
                raise ValueError(
                    f"Cannot interpret {input!r} as {self}. Try implementing a {self}.parse method."
                )
            self.check_internal(
                parsed, f"Parsed value {parsed!r} is not accepted by {self}"
            )
            data = self.data_cls(parsed, self, metadata, **props)

        return data

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DataType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class Singleton:
    _instance: Self | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


def is_data(obj: object, dtype: DataTypeLike[T] | None = None) -> TypeGuard[Data[T]]:
    return isinstance(obj, Data) and (dtype is None or obj.dtype == get_dtype(dtype))


class EmptyDataType(DataType[None], Singleton):
    def __init__(self) -> None:
        super().__init__("empty")

    def validate_internal(self, internal: object):
        if internal is not None:
            yield "Input is not None"

    def parse(self, input: object) -> None:
        return None

    def get_subdata(self, data: Data[None], selector: object) -> Data:
        return Data()

    def set_subdata(self, data: Data[None], selector: object, value: object) -> bool:
        return False

    def delete_subdata(self, data: Data[None], selector: object) -> bool:
        return False

    def get_size(self, data: Data[None]) -> int:
        return 0


class AnyDataType(DataType[Any], Singleton):
    def __init__(self) -> None:
        super().__init__("any")

    def validate_internal(self, internal: object) -> Iterable[str]:
        return []

    def parse(self, input: object) -> object:
        return input


class ValueDataType(DataType[T]):
    def __init__(self, *types: type) -> None:
        self.types = types

        super().__init__("value[" + "|".join(t.__name__ for t in types) + "]")

    def validate_internal(self, internal: object) -> bool:
        return isinstance(internal, self.types)


class IntDataType(DataType[int], Singleton):
    def __init__(self) -> None:
        super().__init__("int")

    def validate_internal(self, internal: object) -> Iterable[str]:
        if not isinstance(internal, int):
            yield "Input is not an int"

    def parse(self, input: object) -> int:
        if isinstance(input, int):
            return input
        elif isinstance(input, (str, SupportsInt, SupportsIndex)):
            return int(input)
        else:
            return NotImplemented


class FloatDataType(DataType[float], Singleton):
    def __init__(self) -> None:
        super().__init__("float")

    def validate_internal(self, internal: object) -> Iterable[str]:
        if not isinstance(internal, float):
            yield "Input is not a float"

    def parse(self, input: object) -> float:
        if isinstance(input, float):
            return input
        elif isinstance(input, SupportsFloat):
            return float(input)
        else:
            return NotImplemented


class StringDataType(DataType[str], Singleton):
    def __init__(self) -> None:
        super().__init__("str")

    def validate_internal(self, internal: object) -> Iterable[str]:
        if not isinstance(internal, str):
            yield "Input is not a string"

    def parse(self, input: object) -> str:
        return str(input)


class dtypes:
    empty = EmptyDataType()
    any = AnyDataType()
    int = IntDataType()
    float = FloatDataType()
    string = StringDataType()

    @staticmethod
    def value(*types: type) -> DataType:
        return ValueDataType(*types)

    @staticmethod
    def record(
        required_vars: tuple[Var, ...] = (),
        optional_vars: tuple[Var, ...] | None = None,
    ) -> DataType:
        return RecordDataType(required_vars, optional_vars)


def get_python_type(name: str) -> type:
    name = name.strip()

    if name in builtins.__dict__:
        t = builtins.__dict__[name]
        if isinstance(t, type):
            return t

    raise TypeError(f"{name!r} is not a type")


def get_dtype_for_type(type_: type) -> DataType:
    if type_ is int:
        return dtypes.int
    elif type_ is float:
        return dtypes.float
    elif type_ is str:
        return dtypes.string
    else:
        return ValueDataType(type_)


DataTypeLike = Union[DataType[T], type[T], str, tuple]


def get_dtype(input: DataTypeLike[T]) -> DataType[T]:
    if isinstance(input, DataType):
        return input
    elif isinstance(input, str):
        input = input.strip()

        if input == "empty":
            return dtypes.empty  # type: ignore
        elif input == "any":
            return dtypes.any
        elif input == "record":
            return RecordDataType()  # type: ignore
        elif input.startswith("value[") and input.endswith("]"):
            type_strs = input[6:-1].split("|")
            types = tuple(get_python_type(type_str) for type_str in type_strs)
            return ValueDataType(*types)
        else:
            parts = input.split("|")
            if len(parts) == 1:
                return get_dtype_for_type(get_python_type(parts[0]))
            else:
                return ValueDataType(*(get_python_type(part) for part in parts))
    elif isinstance(input, type):
        return get_dtype_for_type(input)
    elif isinstance(input, tuple):
        return ValueDataType(*(get_python_type(part) for part in input))
    else:
        raise TypeError(f"{input!r} is not a data type")


class VarDataType(DataType[str], Generic[T]):
    def __init__(self, value_dtype: DataTypeLike[T] = dtypes.any) -> None:
        self.value_dtype = get_dtype(value_dtype)

        super().__init__("var[" + self.value_dtype.name + "]", Var)

    def validate_internal(self, internal: object) -> Iterable[str]:
        if not isinstance(internal, str):
            yield "Input (name) is not a string"

    def parse(self, input: object) -> str:
        return str(input)


class Var(Data[str], Generic[T]):
    name: str
    value_dtype: DataType[T]

    def matches(self, pattern: object):
        return pattern == self or pattern == self.name

    def data(self, input: object) -> Data[T]:
        return self.value_dtype.interpret(input, source="var:" + self.name)


def var(
    name: str, value_dtype: DataTypeLike[T] = dtypes.any, metadata: MetadataLike = None
) -> Var[T]:
    value_dtype = get_dtype(value_dtype)
    return Var(
        name, VarDataType(value_dtype), metadata, name=name, value_dtype=value_dtype
    )


@overload
def vars(names_str: str) -> tuple[Var, ...]:
    ...


@overload
def vars(
    names_str: str, value_dtype: DataTypeLike[T] = dtypes.any
) -> tuple[Var[T], ...]:
    ...


def vars(names_str: str, value_dtype: DataTypeLike[T] = dtypes.any):
    parts = names_str.split(",")
    vars: list[Var] = []
    for part in parts:
        if ":" in part:
            name, _value_dtype = part.split(":", 1)
        else:
            name = part
            _value_dtype = value_dtype
        vars.append(var(name, _value_dtype))
    return tuple(vars)


# class Var(Generic[T]):
#     def __init__(self, name: str, dtype: DataTypeLike[T] = dtypes.any) -> None:
#         self.name = name
#         self.dtype: DataType[T] = get_dtype(dtype)

#     def data(self, input: object) -> Data[T]:
#         return self.dtype.interpret(input, source="var:" + self.name)

#     def matches(self, pattern: object):
#         return pattern == self or pattern == self.name or pattern == self.dtype

#     def __repr__(self) -> str:
#         return f"Var({self.name!r}, {self.dtype!r})"

#     def __str__(self) -> str:
#         return self.name


RecordInternal = dict[Var, object]


class RecordDataType(DataType[RecordInternal]):
    def __init__(
        self,
        required_vars: tuple[Var, ...] = (),
        optional_vars: tuple[Var, ...] | None = None,
    ) -> None:
        self.required_vars = required_vars
        self.optional_vars = optional_vars

        name = "record[" + ", ".join(var.name for var in required_vars)

        if optional_vars is not None:
            if len(required_vars) > 0:
                name += ", "
            name += ", ".join(var.name + "?" for var in optional_vars)
        else:
            if len(required_vars) > 0:
                name += ", "
            name += "..."

        name += "]"

        super().__init__(name, Record)

    @property
    def vars(self) -> tuple[Var, ...]:
        return self.required_vars + (self.optional_vars or ())

    def is_required(self, var: Var) -> bool:
        return var in self.required_vars

    def accepts_var(self, var: Var) -> bool:
        return (
            self.optional_vars is None
            or var in self.optional_vars
            or var in self.required_vars
        )

    def get_var(
        self, selector: object, data: Data[RecordInternal] | None = None
    ) -> Var | None:
        if isinstance(selector, Var):
            if self.accepts_var(selector):
                return selector
            else:
                return None

        for var in self.vars:
            if var.matches(selector):
                return var

        if data is not None:
            for var in data.internal:
                if var.matches(selector):
                    return var

        return None

    def validate_internal(self, internal: object) -> Iterable[str]:
        if not isinstance(internal, dict):
            yield f"Input {internal} is not a dict"
            return

        for key, value in internal.items():
            if not isinstance(key, Var):
                yield f"Key {key} is not a Var"
                continue

            if not self.accepts_var(key):
                yield f"Var {key.name} is not accepted"
                continue

            for reason in key.value_dtype.validate_internal(value):
                yield f"Var {key.name}: {reason}"

        for var in self.required_vars:
            if var not in internal:
                yield f"Missing required var {var}"

    def parse(self, input: object) -> RecordInternal:
        if not isinstance(input, dict):
            raise ValueError(f"Expected dict, got {type(input).__name__}")

        internal: RecordInternal = {}

        for key, value in input.items():
            var = self.get_var(key)
            if var is None:
                raise ValueError(f"Unknown variable {key!r}")

            internal[var] = var.value_dtype.parse(value)

        for var in self.required_vars:
            if var not in internal:
                raise ValueError(f"Missing required variable {var.name!r}")

        return internal

    def get_subdata(self, data: Data[RecordInternal], selector: object) -> Data:
        var = self.get_var(selector, data)
        if var is None:
            return Data()

        return var.data(data.internal.get(var, None))

    def set_subdata(
        self, data: Data[RecordInternal], selector: object, value: object
    ) -> bool:
        var = self.get_var(selector, data)
        if var is None:
            return False

        data.internal[var] = value
        return True

    def delete_subdata(self, data: Data[RecordInternal], selector: object) -> bool:
        var = self.get_var(selector, data)
        if var is None:
            return False

        del data.internal[var]
        return True

    def get_size(self, data: Data[RecordInternal]) -> int:
        return len(data.internal)

    def convert_from(self, data: Data) -> Data[RecordInternal]:
        if isinstance(data.internal, Iterable):
            entries: list[tuple[Var, object]] = []
            for entry in data.internal:
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and is_data(entry[0], VarDataType)
                ):
                    entries.append(entry)
                else:
                    return NotImplemented

            return data.copy(internal=dict(entries), dtype=self)
        else:
            return NotImplemented


class Record(Data[RecordInternal]):
    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(
            {
                var.name: list(value) if isinstance(value, Iterable) else [value]
                for var, value in self.internal.items()
            }
        )

    def __ipython_display__(self) -> None:
        from IPython.display import display

        display(self.to_pandas())


def record(input: object, metadata: MetadataLike = None) -> Record:
    return Record(input, RecordDataType(), metadata=metadata)
