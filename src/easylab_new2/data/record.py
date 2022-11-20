from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
import getpass
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import TypeGuard
from .var import (
    DerivedVar,
    Var,
    VarQuery,
    RecordEntryCondition,
    VarType,
    VarTypeLike,
    is_var_query,
)
from .metadata import Metadata, MetadataLike
from . import data as m_data
from ..lang import is_text_target, Text, lang
from ..internal_util import Undefined, undefined, Wildcard, is_wildcard

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")


def gt(other: Any):
    return lambda x: x > other


def lt(other: Any):
    return lambda x: x < other


def ge(other: Any):
    return lambda x: x >= other


def le(other: Any):
    return lambda x: x <= other


def eq(other: Any):
    return lambda x: x == other


def ne(other: Any):
    return lambda x: x != other


ValueQuery = Union[T, Callable[[T], bool], Wildcard, str]


def is_value_query(input: Any) -> TypeGuard[ValueQuery[T]]:
    return True


comparison_operators = (">", ">=," "<", "<=", "==", "!=", "is", "in")


def value_matches(value: T, query: ValueQuery[T]) -> bool:
    if is_wildcard(query) or query == value:
        return True
    elif callable(query):
        return query(value)
    elif isinstance(query, str):
        query = query.strip()

        if query.startswith(comparison_operators):
            query = "$ " + query
        elif query.endswith(comparison_operators):
            query = query + " $"

        return eval(query.replace("$", "__value__"), globals(), {"__value__": value})

    return False


RecordEntryLike = Union["RecordEntry[T]", tuple[Var[T], T]]


def is_record_entry_like(input: Any) -> TypeGuard[RecordEntryLike[T]]:
    return isinstance(input, RecordEntry) or (
        isinstance(input, tuple) and len(input) == 2 and isinstance(input[0], Var)
    )


RecordEntryQuery = Union[
    VarQuery[T], tuple[VarQuery[T], ValueQuery[T]], RecordEntryCondition[T]
]


def is_record_entry_query(input: Any) -> TypeGuard[RecordEntryQuery[T]]:
    return (
        is_var_query(input)
        or (
            isinstance(input, tuple)
            and len(input) == 2
            and is_var_query(input[0])
            and is_value_query(input[1])
        )
        or isinstance(input, RecordEntryCondition)
    )


class RecordEntry(Generic[T]):
    @staticmethod
    def interpret(
        input: RecordEntryLike[T], *, metadata_hint: Metadata | None = None
    ) -> RecordEntry[T]:
        if isinstance(input, RecordEntry):
            return input
        elif isinstance(input, tuple) and len(input) == 2:
            return RecordEntry(*input, metadata=metadata_hint)
        else:
            raise TypeError(f"Cannot interpret {input!r} as RecordEntry.")

    _var: Var[T]
    _value_input: Any
    _value: T
    metadata: Metadata

    def __init__(
        self,
        var: Var[T],
        value: Any,
        *,
        metadata: MetadataLike = None,
        _parsed_value: T | Undefined = undefined,
    ) -> None:
        self._var = var
        self._value_input = value

        self._value = (
            var.parse(value) if isinstance(_parsed_value, Undefined) else _parsed_value
        )

        self.metadata = Metadata.interpret(metadata)

    def get_formatted_value(self):
        return self._var.format(self._value)

    @property
    def text(self):
        return self._var.label + " = " + self.get_formatted_value()

    @property
    def var(self) -> Var[T]:
        return self._var

    @var.setter
    def var(self, var: Var[T]) -> None:
        self._var = var
        self.value = self._value_input  # Update value (parse with new var)
        self.metadata.update()

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        self._value_input = value
        self._value = self.var.parse(value)
        self.metadata.update()

    def __str__(self) -> str:
        return self.text.ascii

    def __repr__(self) -> str:
        return self.text.ascii

    def value_equals(self, other: Any):
        return self._var.equal(self._value, other)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, RecordEntry)
            and self._var is other._var
            and self.value_equals(other._value)
        )

    def matches(self, query: RecordEntryQuery[T]) -> bool:
        if isinstance(query, tuple):
            var_query, value_query = query
            return self._var.matches(var_query) and value_matches(
                self._value, value_query
            )
        elif isinstance(query, RecordEntryCondition) and len(query.vars) == 1:
            return query.check(self._var)
        else:
            return self._var.matches(query)

    def copy(self, copy_value: bool = False) -> RecordEntry[T]:
        if copy_value:
            return RecordEntry(self._var, copy(self._value))
        else:
            return RecordEntry(
                self._var,
                self._value_input,
                _parsed_value=self._value,  # Pass parsed value to avoid re-parsing
            )

    def __copy__(self):
        return self.copy()


RecordInput = Union[Iterable[RecordEntryLike[T]], dict[Var[T], T]]


def is_record_input(input: Any) -> TypeGuard[RecordInput[T]]:
    return isinstance(input, Iterable) or isinstance(input, dict)


RecordLike = Union["Record[T]", RecordInput[T], "m_data.DataLike"]


def is_record_like(input: Any) -> TypeGuard[RecordLike[T]]:
    return (
        isinstance(input, Record)
        or is_record_input(input)
        or m_data.is_data_like(input)
    )


RecordQuery = Union[
    RecordEntryQuery[T],
    tuple[RecordEntryQuery[T], ...],
    list[RecordEntryQuery[T]],
    Wildcard,
]


def is_record_query(input: Any) -> TypeGuard[RecordQuery[T]]:
    return (
        is_wildcard(input)
        or isinstance(input, tuple)
        or isinstance(input, list)
        or is_record_entry_query(input)
    )


ComparisonLike = Union["Comparison[T]", tuple[T, T]]


class Comparison(Generic[T]):
    @staticmethod
    def interpret(input: ComparisonLike[T]) -> Comparison[T]:
        if isinstance(input, Comparison):
            return input
        elif isinstance(input, tuple) and len(input) == 2:
            return Comparison(*input)
        else:
            raise TypeError(f"Cannot interpret {input!r} as Comparison.")

    def __init__(self, a: T, b: T):
        self.a = a
        self.b = b

    def is_equality(self) -> bool:
        return self.a == self.b

    def __str__(self) -> str:
        return f"{self.a} <-> {self.b}"

    def __repr__(self) -> str:
        return f"Comparison({self.a}, {self.b})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Comparison) and self.a == other.a and self.b == other.b

    def __hash__(self) -> int:
        return hash((self.a, self.b))


class ComparisonVar(Var[Comparison[Union[T, Undefined]]]):
    def __init__(self, var: Var[T]):
        self.var = var
        super().__init__("compare" + lang.par(var.label), Comparison)

    def parse(self, value: Any):
        comparison = Comparison.interpret(value)

        return Comparison(
            undefined
            if isinstance(comparison.a, Undefined)
            else self.var.parse(comparison.a),
            undefined
            if isinstance(comparison.b, Undefined)
            else self.var.parse(comparison.b),
        )

    def check(self, value: Comparison[T | Undefined]):
        if not isinstance(value.a, Undefined):
            self.var.check(value.a)
        if not isinstance(value.b, Undefined):
            self.var.check(value.b)

    def format(self, value: Comparison[T]):
        a, b = value.a, value.b

        if isinstance(a, Undefined) and isinstance(b, Undefined):
            return lang.left_right_arrow
        elif isinstance(a, Undefined):
            return lang.left_right_arrow + lang.space + self.var.format(b)
        elif isinstance(b, Undefined):
            return self.var.format(a) + lang.space + lang.left_right_arrow
        else:
            return (
                self.var.format(a)
                + lang.space
                + lang.left_right_arrow
                + lang.space
                + self.var.format(b)
            )


def take_first(a, b):
    return a


def take_last(a, b):
    return b


class RecordEntryNotFoundException(Exception):
    def __init__(self, query: RecordEntryQuery):
        self.query = query

    def __str__(self) -> str:
        return f"No record entry found for query {self.query!r}."


class VarNotFoundException(Exception):
    def __init__(self, query: VarQuery):
        self.query = query

    def __str__(self) -> str:
        if isinstance(self.query, DerivedVar):
            return f"Not all dependencies of {self.query!r} were found."

        return f"No var found for query {self.query!r}."


S = TypeVar("S")


class Record(Generic[T]):
    @staticmethod
    def interpret(input: RecordLike[T]) -> Record[T]:
        if isinstance(input, Record):
            return input
        elif is_record_input(input):
            return Record(input)
        elif m_data.is_data_like(input):
            return cast(Record[T], m_data.Data.interpret(input).to_record())
        else:
            raise TypeError(f"Cannot interpret {input!r} as Record.")

    _entries: list[RecordEntry[T]]

    def __init__(
        self, entries: RecordInput[T] = [], *, metadata_hint: Metadata | None = None
    ) -> None:
        if isinstance(entries, dict):
            entries_ = ((cast(Var, var), value) for var, value in entries.items())
        elif isinstance(entries, Iterable):
            entries_ = entries
        else:
            raise TypeError(f"Cannot create Record from {entries!r}.")

        self._entries = list(
            RecordEntry.interpret(entry, metadata_hint=metadata_hint)
            for entry in entries_
        )

    @property
    def size(self):
        return len(self._entries)

    @property
    def entries(self) -> list[RecordEntry[T]]:
        return self._entries

    def get_entry_or_none(self, query: RecordEntryQuery[S]) -> RecordEntry[S] | None:
        for entry in self._entries:
            if entry.matches(query):
                return cast(RecordEntry[S], entry)

        if isinstance(query, DerivedVar):
            return RecordEntry(
                query, query.get_value(self), metadata={"source": "derived"}
            )

    def get_entry(self, query: RecordEntryQuery[S]) -> RecordEntry[S]:
        entry = self.get_entry_or_none(query)

        if entry is None:
            raise RecordEntryNotFoundException(query)

        return entry

    def delete_entry(self, query: RecordEntryQuery) -> None:
        for i, entry in enumerate(self._entries):
            if entry.matches(query):
                del self._entries[i]
                return

        # raise NoRecordEntryFoundException(query)

    def get_value(self, query: RecordEntryQuery[S]) -> S:
        return self.get_entry(query).value

    def get_value_or_undefined(self, query: RecordEntryQuery[S]) -> S | Undefined:
        entry = self.get_entry_or_none(query)
        if entry is None:
            return undefined
        else:
            return entry.value

    def get_derived_value(self, derived_var: DerivedVar[S]) -> S:
        return derived_var.get_value(self)

    def set_value(self, query: RecordEntryQuery, value: Any) -> None:
        self.get_entry(query).value = value

    def get_formatted_value(self, query: RecordEntryQuery) -> Text:
        return self.get_entry(query).get_formatted_value()

    @overload
    def get_vars(self) -> list[Var[T]]:
        ...

    @overload
    def get_vars(self, query: RecordEntryQuery[S]) -> list[Var[S]]:
        ...

    def get_vars(self, query: RecordEntryQuery[S] = "*") -> list[Var[S]]:
        vars = []
        for entry in self._entries:
            if entry.matches(query):
                vars.append(entry.var)
        return vars

    def get_var_or_none(self, query: RecordEntryQuery[S]) -> Var[S] | None:
        for entry in self._entries:
            if entry.matches(query):
                return cast(Var[S], entry.var)

        if isinstance(query, DerivedVar):
            # and all(d in self for d in query.get_dependencies()):
            return query

    def get_var(self, query: RecordEntryQuery[S]) -> Var[S]:
        var = self.get_var_or_none(query)

        if var is None:
            raise VarNotFoundException(query)

        return var

    def set_var(self, query: RecordEntryQuery, var: Var) -> None:
        self.get_entry(query).var = var

    def __getitem__(self, query: RecordEntryQuery[S]) -> S:
        if query in self:
            return self.get_value(query)
        elif isinstance(query, DerivedVar):
            return self.get_derived_value(query)

        if isinstance(query, Var):
            default = query.default()
            if default is not undefined:
                return cast(S, default)

        raise RecordEntryNotFoundException(query)

    def __setitem__(self, query: RecordEntryQuery, value: Any) -> None:
        self.set_value(query, value)

    def __delitem__(self, query: RecordEntryQuery) -> None:
        self.delete_entry(query)

    def __contains__(self, query: RecordEntryQuery) -> bool:
        return any(entry.matches(query) for entry in self._entries)

    def add_entry(self, entry: RecordEntryLike) -> None:
        self._entries.append(RecordEntry.interpret(entry))

    def add_derived_var(self, derived_var: DerivedVar) -> None:
        self.add_entry(RecordEntry(derived_var, derived_var.get_value(self)))

    def get_entries_text(self, *, sep: Any = ", ", start: Any = "", end: Any = ""):
        return Text.interpret(sep).join(
            start + entry.text + end for entry in self._entries
        )

    @property
    def text(self) -> Text:
        if self.size < 5:
            return lang.curly_brackets(
                lang.space + self.get_entries_text() + lang.space
            )
        else:
            return lang.curly_brackets(
                lang.newline
                + self.get_entries_text(sep="," + lang.newline, start=lang.large_space)
                + lang.newline
            )

    def __str__(self) -> str:
        return self.text.ascii

    def __repr__(self) -> str:
        return self.text.ascii

    def __iter__(self) -> Iterable[RecordEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Record) and self._entries == other._entries

    def __hash__(self) -> int:
        return hash(tuple(self._entries))

    def matches(self, query: RecordQuery) -> bool:
        if is_wildcard(query) or (isinstance(query, Record) and query == self):
            return True
        elif isinstance(query, list):
            return all(self.matches(q) for q in query)
        elif isinstance(query, dict):
            return all(self.matches(q) for q in query.items())
        elif isinstance(query, RecordEntryCondition):
            values = [self[var] for var in query.vars]
            return query.check(*values)
        else:
            return any(entry.matches(query) for entry in self._entries)

    # Transform all entries to some new type S
    @overload
    def map(
        self,
        query: Wildcard,
        transform_matched_entry: Callable[[RecordEntry[T]], RecordEntryLike[S]],
    ) -> Record[S]:
        ...

    # Transform to same type T
    @overload
    def map(
        self,
        query: RecordEntryQuery[S],
        transform_matched_entry: Callable[[RecordEntry[S]], RecordEntryLike[T]],
    ) -> Record[T]:
        ...

    # Transform to unknown type
    @overload
    def map(
        self,
        query: RecordEntryQuery[S],
        transform_matched_entry: Callable[[RecordEntry[S]], RecordEntryLike],
    ) -> Record:
        ...

    def map(
        self,
        query: RecordEntryQuery[S],
        transform_matched_entry: Callable[[RecordEntry[S]], RecordEntryLike],
    ) -> Record:
        return Record(
            transform_matched_entry(cast(RecordEntry[S], entry))
            if entry.matches(query)
            else entry
            for entry in self._entries
        )

    # Pluck all entries to some new type S
    @overload
    def pluck(self, query: Wildcard, attr: str, type_: VarTypeLike[S]) -> Record[S]:
        ...

    # Pluck to unknown type
    @overload
    def pluck(self, query: RecordEntryQuery, attr: str) -> Record:
        ...

    # General signature
    @overload
    def pluck(
        self, query: RecordEntryQuery, attr: str, type_: VarTypeLike | None = None
    ) -> Record:
        ...

    def pluck(
        self, query: RecordEntryQuery, attr: str, type_: VarTypeLike | None = None
    ) -> Record:
        if type_ is not None:
            if not is_wildcard(query):
                raise ValueError(
                    "Specifying an attr type does only make sense, when all entries are being plucked. If you want to specify a type, use a wildcard query, i.e. '*' or Ellipsis."
                )

            type_ = VarType.interpret(type_)

        def transform_matched_entry(
            entry: RecordEntry, type_: VarType | None
        ) -> RecordEntryLike:
            value = getattr(entry.value, attr)
            if type_ is not None:
                type_.check(value)

            plucked_var = Var(entry.var.label + "." + attr, type_ or object)
            return RecordEntry(plucked_var, value)

        return self.map(query, lambda entry: transform_matched_entry(entry, type_))

    @overload
    def to_dict(self, keys: Literal["vars"]) -> dict[Var[T], T]:
        ...

    @overload
    def to_dict(
        self,
        keys: Literal[
            "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> dict[str, T]:
        ...

    @overload
    def to_dict(
        self,
        keys: Literal[
            "vars", "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> dict[Any, T]:
        ...

    def to_dict(
        self,
        keys: Literal[
            "vars", "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> dict[Any, T]:
        if keys == "var":
            return {entry.var: entry.value for entry in self._entries}
        elif keys.startswith("labels."):
            target = keys[7:]

            if not is_text_target(target):
                raise ValueError(f"Invalid text target {target!r}")

            return {entry.var.label.get(target): entry.value for entry in self._entries}

        raise ValueError(f"Invalid keys argument: {keys!r}")

    def copy(self, copy_entries: bool = True) -> Record[T]:
        if copy_entries:
            return Record(entry.copy() for entry in self._entries)
        else:
            return Record(self._entries)

    def __copy__(self) -> Record[T]:
        return self.copy()

    def compare(
        self, other: RecordLike[T], include_equal: bool = False
    ) -> Record[Comparison[T | Undefined]]:
        """Returns a record containing the comparisons between the two records."""
        other = Record.interpret(other)

        entries: dict[Var[Comparison[T | Undefined]], Comparison[T | Undefined]] = {}

        self_vars = set(self.get_vars())
        other_vars = set(other.get_vars())

        for var in self_vars | other_vars:
            comp_var = ComparisonVar(var)

            if var in self_vars and var in other_vars:
                self_entry = self.get_entry(var)
                other_entry = other.get_entry(var)

                if include_equal or not self_entry.value_equals(other_entry.value):
                    entries[comp_var] = Comparison(self_entry.value, other_entry.value)

            elif var in self_vars:
                entries[comp_var] = Comparison(self.get_value(var), undefined)
            elif var in other_vars:
                entries[comp_var] = Comparison(undefined, other.get_value(var))

        return Record(entries)

    def union(
        self, other: RecordLike, combine: Callable[[Any, Any], Any] = take_first
    ) -> Record:
        """Returns a record containing the union of the two records."""
        other = Record.interpret(other)

        entries: dict[Var[Any], Any] = {}

        for entry in self._entries:
            entries[entry.var] = entry.value

        for entry in other._entries:
            if entry.var in entries:
                entries[entry.var] = combine(entries[entry.var], entry.value)
            else:
                entries[entry.var] = entry.value

        return Record(entries)

    def __or__(self, other: RecordLike) -> Record:
        return self.union(other)

    def intersect(
        self, other: RecordLike, combine: Callable[[Any, Any], Any] = take_first
    ) -> Record:
        """Returns a record containing the intersection of the two records."""
        other = Record.interpret(other)

        entries: dict[Var[Any], Any] = {}

        for entry in self._entries:
            if entry.var in other:
                entries[entry.var] = combine(entry.value, other[entry.var])

        return Record(entries)

    def __and__(self, other: RecordLike) -> Record:
        return self.intersect(other)

    def extract(self) -> Union[None, RecordEntry, Record]:
        if self.size == 0:
            return None
        elif self.size == 1:
            return self._entries[0]
        else:
            return self


_dummy_var = Var("dummy")
_dummy_entry = RecordEntry(_dummy_var, None)


class ObserverRecord(Record[T]):
    def __init__(self):
        super().__init__([])
        self.accessed_vars: list[Var] = []

    def get_entry_or_none(self, query: RecordEntryQuery[S]) -> RecordEntry[S] | None:
        if isinstance(query, Var):
            self.accessed_vars.append(query)

        return cast(RecordEntry[S], _dummy_entry)
