from __future__ import annotations
from abc import ABC, abstractmethod
import itertools

from ..internal_util import EllipsisType, Wildcard, undefined
from typing import (
    Any,
    Callable,
    Generic,
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
    VarNotFoundException,
    is_record_like,
    is_record_query,
)
from .var import Computed, DerivedVar, VarQuery, Var, VarTypeLike
from ..lang import Text, lang, TextTarget
from ..expr import Expr
from .metadata import Metadata
from tabulate import tabulate
from typing_extensions import TypeGuard

T = TypeVar("T")
S = TypeVar("S")

DataLike = Union["Data[T]", Iterable[RecordLike[T]]]


def is_data_like(input: Any) -> TypeGuard[DataLike]:
    if isinstance(input, Data):
        return True
    if isinstance(input, Iterable):
        first = next(iter(input), undefined)
        return first is undefined or is_record_like(first)
    return False


T_Extracted = TypeVar("T_Extracted", bound=Union[None, RecordEntry, Record, "Data"])


def retype_var(var: Var, label: Any, type_: type[T]) -> Var[T]:
    return Var(Text.interpret(label) + lang.par(var.label), type_)


class Data(ABC, Generic[T]):
    @staticmethod
    def interpret(input: DataLike[T]) -> Data[T]:
        if isinstance(input, Data):
            return input
        elif isinstance(input, Iterable):
            return ListData(input)
        else:
            raise TypeError(f"Cannot interpret {input!r} as Data.")

    @staticmethod
    def from_data_frame(data_frame, column_vars: Iterable[Var | None]) -> ListData:
        column_vars = list(column_vars)
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
    def get_records(self) -> Iterable[Record[T]]:
        ...

    @abstractmethod
    def add_record(self, record: RecordLike[T]) -> None:
        ...

    @abstractmethod
    def remove(self, query: RecordQuery[T]) -> None:
        ...

    @abstractmethod
    def copy(self) -> Data[T]:
        ...

    @property
    def size(self) -> int:
        records = self.get_records()
        if isinstance(records, Sized):
            return len(records)
        else:
            return sum(1 for _ in records)

    def add_derived_var(self, derived_var: DerivedVar[T]):
        for record in self.get_records():
            record.add_derived_var(derived_var)

    def add(self, item: RecordLike[T] | DerivedVar[T]) -> None:
        if isinstance(item, DerivedVar):
            self.add_derived_var(item)
        else:
            self.add_record(item)

    def _where(self, query: RecordQuery[T]) -> Data[T]:
        return WhereData(self, query)

    @overload
    def where(self, query: RecordQuery[T], /) -> Data[T]:
        ...

    @overload
    def where(self, var: VarQuery[T], value: ValueQuery[T], /) -> Data[T]:
        ...

    @overload
    def where(
        self, arg1: RecordQuery[T] | VarQuery[T], arg2: ValueQuery[T] | None = None, /
    ) -> Data[T]:
        ...

    def where(
        self, arg1: RecordQuery[T] | VarQuery[T], arg2: ValueQuery[T] | None = None, /
    ) -> Data[T]:
        if arg2 is None:
            query = arg1
        else:
            query = (arg1, arg2)

        return self._where(query)

    def cache(self) -> Data[T]:
        return CachedData(self)

    def child(self, data: DataLike) -> Data:
        return ChildData(Data.interpret(data), parent_data=self)

    def map_records(
        self, transform_record: Callable[[Record[T]], Record[S]]
    ) -> Data[S]:
        return MappedData(self, transform_record)

    def map_entries(
        self,
        query: RecordEntryQuery,
        transform_entry: Callable[[RecordEntry], RecordEntry],
    ):
        return self.map_records(lambda record: record.map(query, transform_entry))

    @overload
    def pluck(self, entry_query: Wildcard, attr: str, type_: VarTypeLike[S]) -> Data[S]:
        ...

    @overload
    def pluck(self, entry_query: RecordEntryQuery, attr: str) -> Data:
        ...

    @overload
    def pluck(
        self, entry_query: RecordEntryQuery, attr: str, type_: VarTypeLike | None = None
    ) -> Data:
        ...

    def pluck(
        self, entry_query: RecordEntryQuery, attr: str, type_: VarTypeLike | None = None
    ):
        return self.map_records(lambda record: record.pluck(entry_query, attr, type_))

    def format(self) -> Data[Text]:
        def format_entry_as_text(entry: RecordEntry) -> RecordEntry[Text]:
            var = retype_var(entry.var, "formatted", Text)
            return RecordEntry(var, entry.var.format(entry.value))

        return self.map_records(lambda record: record.map(..., format_entry_as_text))

    def format_for_target(self, target: TextTarget):
        return self.format().pluck(..., target, str)

    def _extract(self, type_: type[T_Extracted] | None = None) -> T_Extracted:
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

    @overload
    def extract(self, type_: type[T_Extracted] | None = None) -> T_Extracted:
        ...

    @overload
    def extract(self) -> Union[None, RecordEntry[T], Record[T], Data[T]]:
        ...

    def extract(self, type_: type[T_Extracted] | None = None) -> T_Extracted:
        return self._extract(type_)

    def __iter__(self):
        return iter(self.get_records())

    def __len__(self):
        if self.size is not None:
            return self.size
        else:
            return len(list(self.get_records()))

    def __contains__(self, query: RecordQuery) -> bool:
        return any(record.matches(query) for record in self)

    def __getitem__(self, index: SupportsIndex) -> Record[T]:
        # TODO: Support slices, var queries, etc.
        return list(self.get_records())[index]

    def get_entries(self, query: RecordEntryQuery[S]) -> Iterable[RecordEntry[S]]:
        for record in self.get_records():
            entry = record.get_entry_or_none(query)
            if entry is not None:
                yield entry

    def get_values(self, query: RecordEntryQuery[S]) -> Iterable[S]:
        for entry in self.get_entries(query):
            yield entry.value

    def get_value_list(self, query: RecordEntryQuery[S]) -> list[S]:
        return list(self.get_values(query))

    @overload
    def get_vars(self) -> list[Var[T]]:
        ...

    @overload
    def get_vars(self, query: RecordEntryQuery[S]) -> list[Var[S]]:
        ...

    def get_vars(self, query: RecordEntryQuery[S] = "*") -> list[Var[S]]:
        vars = []
        for record in self.get_records():
            for var in record.get_vars(query):
                # Cannot use "in", since it relies on __eq__.
                if not any(other is var for other in vars):
                    vars.append(var)

        return vars

    def get_var_or_none(self, query: RecordEntryQuery[S]) -> Var[S] | None:
        for record in self.get_records():
            var = record.get_var_or_none(query)
            if var is not None:
                return var

    def get_var(self, query: RecordEntryQuery[S]) -> Var[S]:
        var = self.get_var_or_none(query)
        if var is None:
            raise VarNotFoundException(query)
        return var

    def to_list(self):
        return list(self.get_records())

    @overload
    def to_list_of_dicts(self, keys: Literal["vars"]) -> list[dict[Var[T], T]]:
        ...

    @overload
    def to_list_of_dicts(
        self,
        keys: Literal[
            "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> list[dict[str, T]]:
        ...

    @overload
    def to_list_of_dicts(
        self,
        keys: Literal[
            "vars", "labels.ascii", "labels.unicode", "labels.latex"
        ] = "labels.ascii",
    ) -> list[dict[Any, T]]:
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

    def to_record(self):
        return Record(
            {
                var.wrap("collected", var.type.list()): self.get_value_list(var)
                for var in self.get_vars()
            }
        )

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

    @overload
    def __add__(self, other: DataLike[T]) -> Data[T]:
        ...

    @overload
    def __add__(self, other: DataLike) -> Data:
        ...

    def __add__(self, other: DataLike) -> Data:
        return CombinedData(self, other)

    # TODO: Somehow outsource this to the plot module
    def plot(
        self,
        x: VarQuery,
        y: VarQuery,
        ax=None,
        axes_labels: bool = True,
        method: str | Callable = "plot",
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        x = self.get_var(x)
        y = self.get_var(y)

        if ax is None:
            ax = cast(plt.Axes, plt.gca())
        elif not isinstance(ax, plt.Axes):
            raise TypeError("ax must be a matplotlib Axes object.")

        x_values = [x.get_plot_value(v) for v in self.get_values(x)]
        y_values = [y.get_plot_value(v) for v in self.get_values(y)]

        if isinstance(method, str):
            result = getattr(ax, method)(x_values, y_values, **kwargs)
        else:
            result = method(ax, x_values, y_values, **kwargs)
        if axes_labels:
            ax.set_xlabel(x.label.latex)
            ax.set_ylabel(y.label.latex)

        return result

    def plot_all(self, *, single_plot=False):
        vars = self.get_vars()
        import matplotlib.pyplot as plt

        if single_plot:
            fig, axes = plt.subplots(len(vars), len(vars), sharex=True, sharey=True)

            for i, x in enumerate(vars):
                for j, y in enumerate(vars):
                    self.plot(x, y, ax=axes[i][-1 - j], axes_labels=False)

            for i, x in enumerate(vars):
                axes[i][0].set_ylabel(x.label.latex)

            for j, y in enumerate(vars):
                axes[-1][-1 - j].set_xlabel(y.label.latex)

            plt.subplots_adjust(wspace=0, hspace=0)

            return fig, axes
        else:
            figs = []
            axes_list = []
            for i, x in enumerate(vars):
                fig, axes = plt.subplots(
                    len(vars) - 1, 1, figsize=(6, 2 * (len(vars) - 1)), sharex=True
                )
                j = 0
                for y in vars:
                    if y is x:
                        continue
                    self.plot(x, y, ax=axes[j], axes_labels=False)
                    axes[j].set_ylabel(y.label.latex)
                    j += 1

                axes[-1].set_xlabel(x.label.latex)
                figs.append(fig)
                axes_list.append(axes)
                plt.subplots_adjust(hspace=0)
                plt.show()

            return figs, axes_list

    def inspect(self):
        from IPython.display import display
        from ..plot import DataInspector

        display(DataInspector(self))


class ListData(Data[T]):
    def __init__(self, records: Iterable[RecordLike[T]] = []) -> None:
        self.records = list(Record.interpret(record) for record in records)

    def get_records(self) -> Iterable[Record[T]]:
        return self.records

    def add_record(self, record: RecordLike[T]) -> None:
        self.records.append(Record.interpret(record))

    def remove(self, query: RecordQuery[T]) -> None:
        self.records = [record for record in self.records if not record.matches(query)]

    @property
    def size(self) -> int:
        return len(self.records)

    def cache(self) -> Data[T]:
        # Return self since we're already cached.
        return self

    def copy(self) -> Data:
        return ListData(self.records)


class PandasData(Data):
    def __init__(self, data_frame, column_vars: Iterable[Var | None]) -> None:
        import pandas

        if not isinstance(data_frame, pandas.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(data_frame).__name__}."
            )

        self.data_frame: pandas.DataFrame = data_frame

        self.column_vars = list(column_vars)
        if len(self.column_vars) > len(data_frame.columns):
            raise ValueError(
                f"Number of column vars ({len(self.column_vars)}) exceeds number of columns in data frame ({len(data_frame.columns)})."
            )

    def _record_from_pandas_row(self, row) -> Record:
        return Record(
            [
                RecordEntry(var=var, value=row[i])
                for i, var in enumerate(self.column_vars)
                if var is not None
            ]
        )

    def get_records(self) -> Iterable[Record]:
        for row in self.data_frame.itertuples(index=False):
            yield self._record_from_pandas_row(row)

    def add_record(self, record: RecordLike) -> None:
        record_dict = Record.interpret(record).to_dict()
        self.data_frame[-1] = record_dict

    def remove(self, query: RecordQuery) -> None:
        # TODO: Make more efficient by checking the query directly on the row.
        self.data_frame = self.data_frame[
            ~self.data_frame.apply(
                lambda row: self._record_from_pandas_row(row).matches(query),
                axis=1,
            )
        ]

    # OPTIMIZED METHODS

    @property
    def size(self) -> int:
        return len(self.data_frame)

    def to_data_frame(self):
        return self.data_frame

    def where(self, query: RecordQuery) -> Data:
        # TODO: Make more efficient by checking the query directly on the row.
        return PandasData(
            self.data_frame[
                self.data_frame.apply(
                    lambda row: self._record_from_pandas_row(row).matches(query),
                    axis=1,
                )
            ],
            self.column_vars,
        )

    def copy(self) -> Data:
        return PandasData(self.data_frame.copy(), self.column_vars)


class ProxyData(Data):
    def __init__(self, data: DataLike) -> None:
        self.data = Data.interpret(data)

    def get_records(self) -> Iterable[Record]:
        return self.data.get_records()

    def add_record(self, record: RecordLike) -> None:
        self.data.add_record(record)

    def remove(self, query: RecordQuery) -> None:
        self.data.remove(query)

    @property
    def size(self) -> int:
        return self.data.size


class ChildData(ProxyData):
    def __init__(self, data: DataLike, parent_data: DataLike) -> None:
        self.parent_data = Data.interpret(parent_data)
        super().__init__(data)

    def get_records(self) -> Iterable[Record]:
        for parent_record in self.parent_data.get_records():
            for own_record in self.data.get_records():
                yield own_record | parent_record

    @property
    def size(self) -> int:
        return self.data.size * self.parent_data.size

    def copy(self) -> Data:
        return ChildData(self.data, self.parent_data)


class WhereData(ProxyData):
    def __init__(self, data: DataLike, query: RecordQuery):
        if not is_record_query(query):
            raise TypeError(f"Invalid record query: {query!r}")

        self.query = query
        super().__init__(data)

    def get_records(self) -> Iterable[Record]:
        for record in self.data.get_records():
            if record.matches(self.query):
                yield record

    @property
    def size(self) -> int:
        return sum(1 for _ in self.get_records())

    def copy(self) -> Data:
        return WhereData(self.data.copy(), self.query)


class CachedData(Data):
    def __init__(self, data: DataLike):
        self.data = Data.interpret(data)
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

    def remove(self, query: RecordQuery) -> None:
        self.data.remove(query)

        if self._records is not None:
            self._records = [
                record for record in self._records if not record.matches(query)
            ]

    @property
    def size(self) -> int:
        if self._records is None:
            self.get_records()  # Trigger caching
        return len(cast(list, self._records))

    def cache(self) -> Data:
        # Return self since we're already cached.
        return self

    def clear_cache(self) -> None:
        self._records = None

    def copy(self) -> Data:
        return CachedData(self.data.copy())


class CombinedData(Data):
    def __init__(self, *items: DataLike):
        self.items = list(Data.interpret(item) for item in items)

    def get_records(self) -> Iterable[Record]:
        for item in self.items:
            yield from item.get_records()

    def add_record(self, record: RecordLike, pos: SupportsIndex = -1) -> None:
        self.items[pos].add_record(record)

    def remove(self, query: RecordQuery) -> None:
        for item in self.items:
            item.remove(query)

    @property
    def size(self) -> int:
        return sum(item.size for item in self.items)

    def copy(self) -> Data:
        return CombinedData(*(item.copy() for item in self.items))


class MappedData(Data[T], Generic[S, T]):
    def __init__(
        self, data: Data[S], transform_record: Callable[[Record[S]], Record[T]]
    ):
        self.data = data
        self.transform_record = transform_record

    def get_records(self) -> Iterable[Record[T]]:
        for record in self.data.get_records():
            yield self.transform_record(record)

    def add_record(self, record: RecordLike[T]) -> None:
        raise TypeError("FormattedData is read-only.")

    def remove(self, query: RecordQuery[T]) -> None:
        raise TypeError("FormattedData is read-only.")

    def copy(self) -> Data[T]:
        return MappedData(self.data.copy(), self.transform_record)


_default_read_csv_kwargs = dict(
    sep=None,
    encoding="ISO-8859-1",
    header="infer",
    decimal=".",
    engine="python",
)


def type_from_dtype(dtype):
    import numpy as np

    return type(np.zeros(1, type).tolist()[0])


def infer_variable(name: Any, dtype):
    return Var(
        f"V_{name}" if isinstance(name, int) else str(name),
        type_from_dtype(dtype),
        metadata=Metadata(source="inferred"),
    )


def load_data(
    file_path: str,
    column_vars: Iterable[Var | None | Literal["infer"] | EllipsisType] | None = None,
    **kwargs,
) -> Data:
    if column_vars is not None:
        column_vars = list(column_vars)

        if len(column_vars) == 0:
            return ListData([])

    import pandas

    data_frame: pandas.DataFrame = pandas.read_csv(
        file_path, **(_default_read_csv_kwargs | kwargs)
    )

    dtypes = list(data_frame.dtypes.items())

    complete_column_vars: list[Var | None] = []
    # inferred_column_vars: list[Var] = []

    if column_vars is None:
        # Infer all variables
        for name, dtype in dtypes:
            complete_column_vars.append(infer_variable(name, dtype))
    else:
        # Column vars are provided -> only infer specified variables

        for var in column_vars[:-1]:
            if var is ...:
                raise ValueError("Ellipsis can only be used as the last column var.")

        if column_vars[-1] is ...:
            column_vars.pop(-1)
            infer_rest = True
        else:
            infer_rest = False

        # At this point, the ellipsis has been removed if it was present.
        column_vars = cast(list[Union[Var, None]], column_vars)

        for i, var in enumerate(column_vars):
            # First check for string, since var could be a Var object with non-standard __eq__.
            if isinstance(var, str) and var == "infer":
                name, dtype = dtypes[i]
                var = infer_variable(name, dtype)

            complete_column_vars.append(var)

        if infer_rest and len(complete_column_vars) < len(dtypes):
            # Infer the rest of the variables
            for name, dtype in dtypes[len(complete_column_vars) :]:
                var = infer_variable(name, dtype)
                complete_column_vars.append(var)

    return Data.from_data_frame(data_frame, complete_column_vars)
