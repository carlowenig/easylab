from __future__ import annotations
from abc import ABC, abstractmethod
import itertools
from typing import (
    Any,
    Iterable,
    Literal,
    Sized,
    SupportsIndex,
    TypeVar,
    Union,
    cast,
    overload,
)
from .record import (
    Record,
    RecordEntry,
    RecordEntryQuery,
    RecordQuery,
    ValueQuery,
    RecordLike,
)
from .var import Computed, DerivedVar, VarQuery, Var
from ..lang import Text, lang


T = TypeVar("T")

DataLike = Union["Data", Iterable[RecordLike]]


T_Extracted = TypeVar("T_Extracted", bound=Union[None, RecordEntry, Record, "Data"])


class Data(ABC):
    @staticmethod
    def interpret(input: DataLike) -> Data:
        if isinstance(input, Data):
            return input
        elif isinstance(input, Iterable):
            return ListData(input)
        else:
            raise TypeError(f"Cannot interpret {input!r} as Data.")

    @staticmethod
    def from_data_frame(data_frame, column_vars: list[Var | None]) -> ListData:
        import pandas

        if not isinstance(data_frame, pandas.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(data_frame).__name__}."
            )

        if len(column_vars) > len(data_frame.columns):
            raise ValueError(
                f"Number of column vars ({len(column_vars)}) exceeds number of columns in data frame ({len(data_frame.columns)})."
            )

        records = []

        for row in data_frame.itertuples(index=False):
            record = {}

            for i, var in enumerate(column_vars):
                if var is not None:
                    record[var] = row[i]

            records.append(record)

        return ListData(records)

    @abstractmethod
    def get_records(self) -> Iterable[Record]:
        ...

    @abstractmethod
    def add_record(self, record: RecordLike) -> None:
        ...

    @property
    def size(self) -> int:
        records = self.get_records()
        if isinstance(records, Sized):
            return len(records)
        else:
            return sum(1 for _ in records)

    def add_derived_var(self, derived_var: DerivedVar):
        for record in self.get_records():
            record.add_derived_var(derived_var)

    def add(self, item: RecordLike | DerivedVar) -> None:
        if isinstance(item, DerivedVar):
            self.add_derived_var(item)
        else:
            self.add_record(item)

    @overload
    def where(self, query: RecordQuery, /) -> WhereData:
        ...

    @overload
    def where(self, var: VarQuery, value: ValueQuery, /) -> WhereData:
        ...

    @overload
    def where(
        self, arg1: RecordQuery | VarQuery, arg2: ValueQuery | None = None, /
    ) -> WhereData:
        ...

    def where(
        self, arg1: RecordQuery | VarQuery, arg2: ValueQuery | None = None, /
    ) -> WhereData:
        if arg2 is None:
            query = arg1
        else:
            query = (arg1, arg2)

        return WhereData(self, query)

    def cache(self) -> CachedData:
        return CachedData(self)

    @overload
    def extract(self, type_: type[T_Extracted] | None = None) -> T_Extracted:
        ...

    @overload
    def extract(self) -> Union[None, RecordEntry, Record, Data]:
        ...

    def extract(self, type_: type[T_Extracted] | None = None) -> T_Extracted:
        if self.size == 0:
            extracted = None
        elif self.size == 1:
            extracted = self[0].extract()
        else:
            extracted = self

        if type_ is not None and not isinstance(extracted, type_):
            raise TypeError(
                f"Extracted value has an invalid type. Expected {type_.__name__}, got {type(extracted).__name__}."
            )

        return cast(T_Extracted, extracted)

    def __iter__(self):
        return iter(self.get_records())

    def __len__(self):
        if self.size is not None:
            return self.size
        else:
            return len(list(self.get_records()))

    def __contains__(self, query: RecordQuery) -> bool:
        return any(record.matches(query) for record in self)

    def __getitem__(self, index: SupportsIndex) -> Record:
        # TODO: Support slices, var queries, etc.
        return list(self.get_records())[index]

    def get_entries(self, query: RecordEntryQuery[T]) -> Iterable[RecordEntry[T]]:
        for record in self.get_records():
            entry = record.get_entry_or_none(query)
            if entry is not None:
                yield entry

    def get_values(self, query: RecordEntryQuery[T]) -> Iterable[T]:
        for entry in self.get_entries(query):
            yield entry.value

    def to_list(self):
        return list(self.get_records())

    @overload
    def to_list_of_dicts(self, keys: Literal["vars"]) -> list[dict[Var, Any]]:
        ...

    @overload
    def to_list_of_dicts(
        self,
        keys: Literal[
            "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> list[dict[str, Any]]:
        ...

    @overload
    def to_list_of_dicts(
        self,
        keys: Literal[
            "vars", "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> list[dict[Any, Any]]:
        ...

    def to_list_of_dicts(
        self,
        keys: Literal[
            "vars", "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ):
        return [record.to_dict(keys=keys) for record in self.get_records()]

    def to_data_frame(
        self,
        keys: Literal[
            "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ):
        import pandas

        return pandas.DataFrame(self.to_list_of_dicts(keys=keys))

    def get_records_text(
        self,
        *,
        sep: Any = ", ",
        start: Any = "",
        end: Any = "",
        limit: int | None = None,
    ) -> Text:
        records = self.get_records()

        if limit is not None:
            records = itertools.islice(records, limit)

        return Text.interpret(sep).join(start + record.text + end for record in records)

    @property
    def text(self):
        limit = 5
        if self.size is not None and self.size < 3:
            return lang.brackets(lang.space + self.get_records_text() + lang.space)
        else:
            return lang.brackets(
                lang.newline
                + self.get_records_text(
                    sep=lang.newline, start=lang.large_space, limit=limit
                )
                + lang.newline
                + (
                    lang.large_space
                    + f"...{self.size - limit} more records ({self.size} total)"
                    + lang.newline
                    if self.size > limit
                    else ""
                )
            )

    def __str__(self) -> str:
        return self.text.ascii

    def __repr__(self) -> str:
        return self.text.ascii

    def __add__(self, other: DataLike) -> Data:
        return CombinedData(self, other)


class ListData(Data):
    def __init__(self, records: Iterable[RecordLike] = []) -> None:
        self.records = list(Record.interpret(record) for record in records)

    def get_records(self) -> Iterable[Record]:
        return self.records

    def add_record(self, record: RecordLike) -> None:
        self.records.append(Record.interpret(record))

    @property
    def size(self) -> int:
        return len(self.records)


class WhereData(Data):
    def __init__(self, data: Data, query: RecordQuery):
        self.data = data
        self.query = query

    def get_records(self) -> Iterable[Record]:
        for record in self.data.get_records():
            if record.matches(self.query):
                yield record

    def add_record(self, record: RecordLike) -> None:
        return self.data.add_record(record)


class CachedData(Data):
    def __init__(self, data: Data):
        self.data = data
        self._records: list[Record] | None = None

    def get_records(self) -> Iterable[Record]:
        if self._records is None:
            self._records = list(self.data.get_records())

        return self._records

    def add_record(self, record: RecordLike) -> None:
        self.data.add_record(record)

        record = Record.interpret(record)
        if self._records is not None:
            self._records.append(record)

    @property
    def size(self) -> int:
        if self._records is None:
            self.get_records()  # Trigger caching
        return len(cast(list, self._records))

    def clear_cache(self) -> None:
        self._records = None


class CombinedData(Data):
    def __init__(self, *items: DataLike):
        self.items = list(Data.interpret(item) for item in items)

    def get_records(self) -> Iterable[Record]:
        for item in self.items:
            yield from item.get_records()

    def add_record(self, record: RecordLike, pos: SupportsIndex = -1) -> None:
        self.items[pos].add_record(record)

    @property
    def size(self) -> int:
        return sum(item.size for item in self.items)


_default_read_csv_kwargs = dict(
    sep=None,
    encoding="ISO-8859-1",
    header="infer",
    decimal=".",
    engine="python",
)


def load_data(file_path: str, column_vars: list[Var | None], **kwargs) -> Data:
    import pandas

    data_frame = pandas.read_csv(file_path, **(_default_read_csv_kwargs | kwargs))

    return Data.from_data_frame(data_frame, column_vars)
