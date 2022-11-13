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
from .var import DerivedVar, Var, VarQuery, RecordEntryCondition
from .metadata import Metadata, MetadataLike
from ..lang import is_text_target, Text, lang
from ..util import Undefined, undefined, Wildcard, is_wildcard

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

RecordEntryQuery = Union[
    VarQuery[T], tuple[VarQuery[T], ValueQuery[T]], RecordEntryCondition
]


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


RecordInput = Union[Iterable[RecordEntryLike], dict[Var[Any], Any]]

RecordLike = Union["Record", RecordInput]

RecordQuery = Union[
    RecordEntryQuery, tuple[RecordEntryQuery, ...], list[RecordEntryQuery], Wildcard
]

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


class Record:
    @staticmethod
    def interpret(input: RecordLike) -> Record:
        if isinstance(input, Record):
            return input
        else:
            return Record(input)

    _entries: list[RecordEntry[Any]]

    def __init__(
        self, entries: RecordInput, *, metadata_hint: Metadata | None = None
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
    def entries(self) -> list[RecordEntry[Any]]:
        return self._entries

    def get_entry_or_none(self, query: RecordEntryQuery[T]) -> RecordEntry[T] | None:
        for entry in self._entries:
            if entry.matches(query):
                return entry

        if isinstance(query, DerivedVar):
            return RecordEntry(
                query, query.get_value(self), metadata={"source": "derived"}
            )

    def get_entry(self, query: RecordEntryQuery[T]) -> RecordEntry[T]:
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

    def get_value(self, query: RecordEntryQuery[T]) -> T:
        return self.get_entry(query).value

    def get_value_or_undefined(self, query: RecordEntryQuery[T]) -> T | Undefined:
        entry = self.get_entry_or_none(query)
        if entry is None:
            return undefined
        else:
            return entry.value

    def get_derived_value(self, derived_var: DerivedVar[T]) -> T:
        return derived_var.get_value(self)

    def set_value(self, query: RecordEntryQuery, value: Any) -> None:
        self.get_entry(query).value = value

    def get_formatted_value(self, query: RecordEntryQuery[T]) -> Text:
        return self.get_entry(query).get_formatted_value()

    def get_vars(self, query: RecordEntryQuery[T] = "*") -> list[Var[T]]:
        vars = []
        for entry in self._entries:
            if entry.matches(query):
                vars.append(entry.var)
        return vars

    def get_var_or_none(self, query: RecordEntryQuery[T]) -> Var[T] | None:
        for entry in self._entries:
            if entry.matches(query):
                return entry.var

        if isinstance(query, DerivedVar):
            # and all(d in self for d in query.get_dependencies()):
            return query

    def get_var(self, query: RecordEntryQuery[T]) -> Var[T]:
        var = self.get_var_or_none(query)

        if var is None:
            raise VarNotFoundException(query)

        return var

    def set_var(self, query: RecordEntryQuery, var: Var) -> None:
        self.get_entry(query).var = var

    def __getitem__(self, query: RecordEntryQuery[T]) -> T:
        if query in self:
            return self.get_value(query)
        elif isinstance(query, DerivedVar):
            return self.get_derived_value(query)

        if isinstance(query, Var):
            default = query.default()
            if default is not undefined:
                return cast(T, default)

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

    @overload
    def to_dict(self, keys: Literal["vars"]) -> dict[Var, Any]:
        ...

    @overload
    def to_dict(
        self,
        keys: Literal[
            "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> dict[str, Any]:
        ...

    @overload
    def to_dict(
        self,
        keys: Literal[
            "vars", "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> dict[Any, Any]:
        ...

    def to_dict(
        self,
        keys: Literal[
            "vars", "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> dict[Any, Any]:
        if keys == "var":
            return {entry.var: entry.value for entry in self._entries}
        elif keys.startswith("labels."):
            target = keys[7:]

            if not is_text_target(target):
                raise ValueError(f"Invalid text target {target!r}")

            return {entry.var.label.get(target): entry.value for entry in self._entries}

        raise ValueError(f"Invalid keys argument: {keys!r}")

    def copy(self, copy_entries: bool = True) -> Record:
        if copy_entries:
            return Record(entry.copy() for entry in self._entries)
        else:
            return Record(self._entries)

    def __copy__(self) -> Record:
        return self.copy()

    def compare(self, other: RecordLike, include_equal: bool = False) -> Record:
        """Returns a record containing the comparisons between the two records."""
        other = Record.interpret(other)

        entries: dict[ComparisonVar[Any], Any] = {}

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

        return Record(entries)  # type: ignore # TODO: fix typing

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
