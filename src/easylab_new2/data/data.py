from __future__ import annotations
from abc import ABC, abstractmethod
import itertools
from ..util import EllipsisType
from typing import (
    Any,
    Callable,
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
)
from .var import Computed, DerivedVar, VarQuery, Var
from ..lang import Text, lang
from .metadata import Metadata


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
    def get_records(self) -> Iterable[Record]:
        ...

    @abstractmethod
    def add_record(self, record: RecordLike) -> None:
        ...

    @abstractmethod
    def remove(self, query: RecordQuery) -> None:
        ...

    @abstractmethod
    def copy(self) -> Data:
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

    def _where(self, query: RecordQuery) -> Data:
        return WhereData(self, query)

    @overload
    def where(self, query: RecordQuery, /) -> Data:
        ...

    @overload
    def where(self, var: VarQuery, value: ValueQuery, /) -> Data:
        ...

    @overload
    def where(
        self, arg1: RecordQuery | VarQuery, arg2: ValueQuery | None = None, /
    ) -> Data:
        ...

    def where(
        self, arg1: RecordQuery | VarQuery, arg2: ValueQuery | None = None, /
    ) -> Data:
        if arg2 is None:
            query = arg1
        else:
            query = (arg1, arg2)

        return self._where(query)

    def cache(self) -> Data:
        return CachedData(self)

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
    def extract(self) -> Union[None, RecordEntry, Record, Data]:
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

    def get_value_list(self, query: RecordEntryQuery[T]) -> list[T]:
        return list(self.get_values(query))

    def get_vars(self, query: RecordEntryQuery[T] = "*") -> list[Var[T]]:
        vars = []
        for record in self.get_records():
            for var in record.get_vars(query):
                # Cannot use "in", since it relies on __eq__.
                if not any(other is var for other in vars):
                    vars.append(var)

        return vars

    def get_var_or_none(self, query: RecordEntryQuery[T]) -> Var[T] | None:
        for record in self.get_records():
            var = record.get_var_or_none(query)
            if var is not None:
                return var

    def get_var(self, query: RecordEntryQuery[T]) -> Var[T]:
        var = self.get_var_or_none(query)
        if var is None:
            raise VarNotFoundException(query)
        return var

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
        from .. import plot  # Setup plotting

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

    def inspect(self):
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        import pandas as pd

        vars = self.get_vars()

        x_input = widgets.Text(
            description="x", value=vars[0].label.ascii, continuous_update=False
        )
        y_input = widgets.Text(
            description="y", value=vars[1].label.ascii, continuous_update=False
        )

        def create_plot_var(input_str: str, label: Any):
            return Computed(
                label, vars, input_str, vars[0].type
            )  # TODO: Better type inference

        plot_output = widgets.Output()
        table_output = widgets.Output()

        # fig, ax = plt.subplots()
        def update(x_str, y_str):
            table_output.clear_output(True)
            plot_output.clear_output(True)

            try:
                x = create_plot_var(x_str, "x")
                y = create_plot_var(y_str, "y")
            except Exception as e:
                with plot_output:
                    _, ax = plt.subplots(figsize=(8, 6))
                    ax.text(
                        0.5,
                        0.5,
                        str(e),
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        color="red",
                        fontdict={"size": 8},
                    )
                    plt.show()
                return

            table_data = self.copy()
            table_data.add(x)
            table_data.add(y)

            with plot_output:
                _, ax = plt.subplots(figsize=(8, 6))
                try:
                    table_data.plot(x, y, ax=ax)
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        str(e),
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        color="red",
                        fontdict={"size": 8},
                    )
                plt.show()

            with table_output:
                display(table_data.to_data_frame())

        update(vars[0].label.ascii, vars[1].label.ascii)

        x_input.observe(lambda change: update(change.new, y_input.value), "value")  # type: ignore
        y_input.observe(lambda change: update(x_input.value, change.new), "value")  # type: ignore

        controls = widgets.HBox([x_input, y_input])

        return widgets.VBox([controls, widgets.HBox([plot_output, table_output])])


class ListData(Data):
    def __init__(self, records: Iterable[RecordLike] = []) -> None:
        self.records = list(Record.interpret(record) for record in records)

    def get_records(self) -> Iterable[Record]:
        return self.records

    def add_record(self, record: RecordLike) -> None:
        self.records.append(Record.interpret(record))

    def remove(self, query: RecordQuery) -> None:
        self.records = [record for record in self.records if not record.matches(query)]

    @property
    def size(self) -> int:
        return len(self.records)

    def cache(self) -> Data:
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

    def remove(self, query: RecordQuery) -> None:
        return self.data.remove(query)

    def copy(self) -> Data:
        return WhereData(self.data.copy(), self.query)


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
