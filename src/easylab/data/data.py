from abc import ABC, abstractmethod
from itertools import chain
import glob

from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import Self
from matplotlib import pyplot as plt
import pandas as pd
from IPython.display import display

from ..util import AutoNamed
from ..util import Comparable, all_fulfill_type_guard
from ..plot import plot
from ..lang import lang
from .var import Var
from .constraint import (
    AndConstraint,
    ConstraintInput,
    Constraint,
    is_constraint_input,
    AnyConstraint,
)
from .value import ValueVar


_T = TypeVar("_T")
_C = TypeVar("_C", bound=Comparable)

DataEntryInput = Union[
    "DataEntry",
    dict[Var, Any],
    ConstraintInput,
    Iterable[ConstraintInput],
]


class DataEntry:
    _constraints: list[Constraint]
    _vars: set[Var]

    def __init__(self, *constraints: ConstraintInput):
        self._constraints = []
        self._vars = set()
        for constraint in constraints:
            constraint = Constraint.parse(constraint)
            for other in [*self._constraints]:
                if other == constraint or other.includes(constraint):
                    # Constraint is redundant, since another constraint includes it
                    break
                if constraint.includes(other):
                    # A previous constraint is made redundant by this constraint
                    self._constraints.remove(other)
            else:
                # Constraint should be added if no other constraint already includes it
                self._constraints.append(constraint)
                self._vars.add(constraint.var)

    @staticmethod
    def parse(input: DataEntryInput):
        if isinstance(input, DataEntry):
            return input
        elif isinstance(input, dict):
            return DataEntry(
                *(Constraint.parse((cast(Var, var), val)) for var, val in input.items())
            )
        elif is_constraint_input(input):
            return DataEntry(input)
        elif isinstance(input, Iterable) and all_fulfill_type_guard(
            input, is_constraint_input
        ):
            return DataEntry(*input)
        else:
            raise ValueError(f"Cannot parse DataEntry from {input}.")

    @property
    def vars(self) -> set[Var]:
        return self._vars.copy()

    @property
    def constraints(self):
        return self._constraints

    @property
    def size(self):
        return len(self._constraints)

    @property
    def is_empty(self) -> bool:
        return len(self._constraints) == 0

    def has_var(self, var: Var):
        return var in self._vars

    def __contains__(self, var: Var):
        return self.has_var(var)

    def var_constraints(
        self,
        var: Var[_T],
        constraint_type: Optional[type] = None,
    ) -> list[Constraint[_T]]:
        return [
            c
            for c in self._constraints
            if c.var == var
            and (constraint_type is None or isinstance(c, constraint_type))
        ]

    def constraint(
        self,
        var: Var[_T],
        constraint_type: Optional[type] = None,
    ) -> Constraint[_T]:
        constraints = self.var_constraints(var, constraint_type)

        if len(constraints) == 0:
            return AnyConstraint(var)
        elif len(constraints) > 1:
            return AndConstraint(*constraints)
        else:
            return constraints[0]

    def __getitem__(self, var: Var[_T]) -> Constraint[_T]:
        return self.constraint(var)

    def value(self, var: Var[_T]) -> Optional[_T]:
        if var not in self._vars and var.is_computed:
            return var.eval(*(self.value(cast(Var, d)) for d in var.dependencies))

        for constraint in self.var_constraints(var):
            value = constraint.value
            if value is not None:
                return value

    def where(self, condition: Callable[[Constraint], bool]):
        return DataEntry(*(c for c in self._constraints if condition(c)))

    def remove_vars(self, *vars: Var):
        return self.where(lambda c: c.var not in vars)

    def keep_vars(self, *vars: Var):
        return self.where(lambda c: c.var in vars)

    def __len__(self):
        return len(self._constraints)

    def __add__(
        self,
        other: DataEntryInput,
    ):
        other = DataEntry.parse(other)

        return DataEntry(*(self.constraints + other.constraints))

    def vars_str(self, *, separator: str = ", ") -> str:
        return separator.join(str(c) for c in self._constraints)

    def __str__(self) -> str:
        return "( " + self.vars_str() + " )"

    def __repr__(self) -> str:
        return f"DataEntry({self._constraints})"


VarSelector = Union[Var, str]


class AbstractData(ABC):
    @property
    @abstractmethod
    def entries(self) -> Iterable[DataEntry]:
        pass

    @abstractmethod
    def add(self, *entries: DataEntryInput) -> Self:
        pass

    @abstractmethod
    def update(self, f: Callable[[DataEntry], DataEntry]) -> Self:
        pass

    @abstractmethod
    def remove_where(self, condition: Callable[[DataEntry], bool]) -> Self:
        pass

    @abstractmethod
    def copy(self) -> Self:
        pass

    def __len__(self):
        return len(list(self.entries))

    @property
    def is_empty(self):
        return len(self) == 0

    def clear(self) -> Self:
        return self.remove_where(lambda _: True)

    def add_var(self, var: Var):
        return self.update(lambda entry: entry + {var: entry.value(var)})

    def add_rows(self, header: Sequence[Var], *rows: Sequence[float]):
        var_count = len(header)

        for row in rows:
            if len(row) != var_count:
                raise ValueError(
                    f"All rows must have the same number of elements as the header row. Found row {row} for header {header}."
                )

        entries = []
        for row in rows:
            entry = {}
            for i, var in enumerate(header):
                entry[var] = row[i]
            entries.append(entry)

        return self.add(*entries)

    def add_dataframe(
        self,
        dataframe: pd.DataFrame,
        vars: Union[dict[Any, Var], Iterable[Var]],
        transform: Optional[Callable[[DataEntry], DataEntry]] = None,
    ):
        entries = []
        for _, row in dataframe.iterrows():
            entry = {}
            if isinstance(vars, dict):
                for var, name in cast(dict[Any, Var], vars.items()):
                    entry[var] = row[name]
            else:
                for i, var in enumerate(vars):
                    entry[var] = row[i]
            entries.append(
                transform(DataEntry.parse(entry)) if transform is not None else entry
            )

        return self.add(*entries)

    def add_csv(
        self,
        path_glob: str,
        vars: Union[dict[Any, Var], Iterable[Var]],
        transform: Optional[Callable[[DataEntry, str], DataEntry]] = None,
        **kwargs,
    ):
        default_kwargs = dict(
            sep=None,
            encoding="ISO-8859-1",
            header="infer",
            decimal=".",
            engine="python",
        )

        result = self

        for path in glob.glob(path_glob):
            df = cast(pd.DataFrame, pd.read_csv(path, **default_kwargs, **kwargs))
            result = result.add_dataframe(
                df,
                vars,
                transform=(lambda entry: transform(entry, path))
                if transform is not None
                else None,
            )
        # dataframe = cast(pd.DataFrame, pd.read_csv(path, **(default_kwargs | kwargs)))

        return result

    # @overload
    # def set(self, var: Var, value: Any, /) -> Self:
    #     ...

    # @overload
    # def set(self, entry: DataEntryInput, /) -> Self:
    #     ...

    # def set(self, *args) -> Self:
    #     if len(args) == 1:
    #         return self.update(lambda entry: entry + args[0])
    #     elif len(args) == 2:
    #         return self.update(lambda entry: entry + {args[0]: args[1]})
    #     else:
    #         raise ValueError(f"Invalid arguments: {args}")

    @property
    def vars(self):
        return set(chain(*(entry.vars for entry in self.entries)))

    def constraints(self, var: Var[_T]) -> Iterable[Constraint[_T]]:
        for entry in self.entries:
            if var in entry:
                yield entry[var]

    def constraint(self, var: Var[_T]) -> Constraint[_T]:
        return AndConstraint(*self.constraints(var))

    @property
    def values_dict(self):
        result: dict[Var, list] = {}
        for entry in self.entries:
            for var in entry.vars:
                if var not in result:
                    result[var] = []
                result[var].append(entry.value(var))
        return result

    def values(self, var: Var[_T]) -> Iterable[_T]:
        for entry in self.entries:
            value = entry.value(var)
            if value is not None:
                yield value

    def value(self, var: Var[_T]) -> _T:
        values = list(set(self.values(var)))
        if len(values) == 0:
            raise ValueError(f"No value found for variable {var}.")
        elif len(values) > 1:
            raise ValueError(f"Multiple values found for variable {var}.")
        else:
            return values[0]

    def __contains__(self, entry: DataEntryInput) -> bool:
        entry = DataEntry.parse(entry)
        return entry in self.entries

    @overload
    def select(self, selection: DataEntryInput, /) -> "SelectedData":
        ...

    @overload
    def select(self, var: Var, value: Any, /) -> "SelectedData":
        ...

    @overload
    def select(
        self, var: Var, operator: Literal["<=", ">=", "<", ">"], value: Any, /
    ) -> "SelectedData":
        ...

    @overload
    def select(self, var: Var, min: Any, max: Any, /) -> "SelectedData":
        ...

    def select(self, *args) -> "SelectedData":
        if len(args) == 1:
            if isinstance(self, SelectedData):
                # Merge selections if possible
                return SelectedData(self.inner, self.selection + args[0])
            else:
                return SelectedData(self, args[0])
        elif len(args) == 2 or len(args) == 3:
            return self.select(args)
        else:
            raise ValueError(
                f"Invalid number of arguments. Expected 1-3, got {len(args)}."
            )

    __getitem__ = select

    def group_by(self, *vars: Var) -> tuple["AbstractData"]:
        if len(vars) == 0:
            return (self,)

        if len(vars) > 1:
            return sum(
                (g.group_by(*vars[1:]) for g in self.group_by(vars[0])),
                cast(tuple["AbstractData"], ()),
            )

        var = vars[0]
        constraints = []
        for entry in self.entries:
            if var in entry:
                constraints.append(entry[var])

        return tuple(self.select(constraint) for constraint in set(constraints))
        # return DataCollection(*(self.select({var: value}) for value in set(values)))

    @property
    def constants(self):
        const_constraints: list[Constraint] = []

        for i, entry in enumerate(self.entries):
            for constraint in entry.constraints:
                if i == 0:
                    const_constraints.append(constraint)
                else:
                    for other in const_constraints:
                        if other.var == constraint.var and other != constraint:
                            const_constraints.remove(other)

        return DataEntry(*const_constraints)

    def __or__(self, condition_entry: DataEntryInput):
        return self.select(condition_entry)

    def graph(self, x: "Var[_X]", y: "Var[_Y]") -> "GraphData[_X, _Y]":
        return GraphData(self, x, y)

    @property
    def name(self) -> str:
        return "Data"

    @property
    def hidden_vars(self) -> Set[Var]:
        return set()

    @property
    def info(self):
        title = self.name

        entries = list(self.entries)

        if len(entries) == 0:
            return f"{title}: empty"

        line_length = len(title) + 2
        sections: list[str] = [f"  {title}"]

        hidden_vars = self.hidden_vars

        constants = self.constants.remove_vars(*hidden_vars)
        if not constants.is_empty:
            constants_str = f"    const: {constants.vars_str()}"
            line_length = max(line_length, len(constants_str))
            sections.append(constants_str)

        entries_lines: list[str] = []
        for entry in entries:
            entry = entry.remove_vars(*constants.vars, *hidden_vars)

            if not entry.is_empty:
                entry_line = "    " + entry.vars_str()

                line_length = max(line_length, len(entry_line))
                entries_lines.append(entry_line)

        if len(entries_lines) > 0:
            sections.append("\n".join(entries_lines))

        line_length += 2

        return (
            "=" * line_length
            + "\n"
            + ("\n" + "-" * line_length + "\n").join(sections)
            + "\n"
            + "=" * line_length
        )

    def to_dataframe(
        self,
        *,
        text_target: Optional[str] = None,
        format: bool = False,
        unit_on_top: bool = False,
    ):
        dict = {}

        for var, values in self.values_dict.items():
            if unit_on_top and isinstance(var, ValueVar):
                var = var.remove_unit()

            key = var.label.string(text_target) if text_target is not None else var.name

            if format:
                values = [
                    var.format(value).string(text_target or "default")
                    for value in values
                ]

            dict[key] = values

        return pd.DataFrame(dict)

    def to_array(self):
        return self.to_dataframe().to_numpy()

    def show(
        self,
        *,
        text_target: Optional[str] = "unicode",
        unit_on_top: bool = True,
    ):
        display(
            self.to_dataframe(
                text_target=text_target, format=True, unit_on_top=unit_on_top
            ),
        )

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"


class Data(AbstractData, AutoNamed):
    _name: Optional[str] = None
    _entries: list[DataEntry]

    def __init__(self, name: Optional[str] = None, entries: Iterable[DataEntry] = ()):
        self.__init_auto_named__(name)
        self._entries = list(entries)

    @property
    def name(self) -> str:
        return self._name or super().name

    @property
    def entries(self) -> Iterable[DataEntry]:
        return iter(self._entries)

    def add(self, *entries: DataEntryInput):
        self._entries.extend(DataEntry.parse(entry) for entry in entries)
        return self

    def update(self, f: Callable[[DataEntry], DataEntry]) -> Self:
        self._entries = [f(entry) for entry in self._entries]
        return self

    def remove_where(self, condition: Callable[[DataEntry], bool]) -> Self:
        self._entries = [entry for entry in self._entries if not condition(entry)]
        return self

    def clear(self) -> Self:
        self._entries = []
        return self

    def copy(self):
        return Data(self._name, self._entries)


class ForwardingData(AbstractData):
    inner: AbstractData

    def __init__(self, inner: AbstractData):
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def entries(self) -> Iterable[DataEntry]:
        return self.inner.entries

    def add(self, *entries: DataEntryInput):
        self.inner.add(*entries)
        return self

    def update(self, f: Callable[[DataEntry], DataEntry]) -> Self:
        self.inner.update(f)
        return self

    def remove_where(self, condition: Callable[[DataEntry], bool]) -> Self:
        self.inner.remove_where(condition)
        return self

    def clear(self) -> Self:
        self.inner.clear()
        return self

    def copy(self):
        return ForwardingData(self.inner.copy())

    @property
    def hidden_vars(self) -> Set[Var]:
        return self.inner.hidden_vars


class ConditionalData(ForwardingData):
    @abstractmethod
    def __contains__(self, entry: DataEntry) -> bool:
        pass

    @property
    def entries(self) -> Iterable[DataEntry]:
        for entry in self.inner.entries:
            if entry in self:
                yield entry

    def update(self, f: Callable[[DataEntry], DataEntry]) -> Self:
        self.inner.update(lambda entry: f(entry) if entry in self else entry)
        return self

    def remove_where(self, condition: Callable[[DataEntry], bool]) -> Self:
        self.inner.remove_where(lambda entry: condition(entry) and entry in self)
        return self


class SelectedData(ConditionalData):
    selection: DataEntry

    def __init__(self, inner: AbstractData, selection: DataEntryInput):
        self.selection = DataEntry.parse(selection)
        super().__init__(inner)

    @property
    def name(self) -> str:
        return f"{self.inner.name}[{self.selection.vars_str()}]"

    def __contains__(self, entry: DataEntry):
        return all(
            c.var not in entry or c.includes(entry.constraint(c.var))
            for c in self.selection.constraints
        )

    def add(self, *entries: DataEntryInput):
        self.inner.add(*(DataEntry.parse(entry) + self.selection for entry in entries))
        return self

    def copy(self):
        return SelectedData(self.inner.copy(), self.selection)


_X = TypeVar("_X")
_Y = TypeVar("_Y")


class GraphData(ConditionalData, Generic[_X, _Y]):
    x: Var[_X]
    y: Var[_Y]

    def __init__(self, inner: AbstractData, x: Var[_X], y: Var[_Y]):
        super().__init__(inner)
        self.x = x
        self.y = y

    def __contains__(self, entry: DataEntry) -> bool:
        x_val = entry.value(self.x)
        y_val = entry.value(self.y)
        return x_val is not None and y_val is not None

    @property
    def name(self) -> str:
        return f"{self.inner.name}[{self.x}, {self.y}]"

    @property
    def value_pairs(self) -> Iterable[tuple[_X, _Y]]:
        for entry in self.inner.entries:
            x_val = entry.value(self.x)
            y_val = entry.value(self.y)
            if x_val is not None and y_val is not None:
                yield (x_val, y_val)

    def plot(
        self, *args, axes: Optional[plt.Axes] = None, method: str = "plot", **kwargs
    ):
        param_vars = self.vars - self.constants.vars - {self.x, self.y}
        # Filter out variables, that depend on x or y
        param_vars = {
            var
            for var in param_vars
            if not (self.x in var.dependencies or self.y in var.dependencies)
        }
        grouped_data = self.inner.group_by(*param_vars)

        results = []

        for data in grouped_data:
            if data.is_empty:
                continue

            label = None
            for var in param_vars:
                if label is None:
                    label = ""
                else:
                    label += "," + lang.space
                label += var.text + " = " + var.format(data.value(var))

            value_pairs = []

            for entry in data.entries:
                x_val = entry.value(self.x)
                y_val = entry.value(self.y)
                if x_val is not None and y_val is not None:
                    value_pairs.append((x_val, y_val))

            results.append(
                plot(
                    self.x,
                    self.y,
                    value_pairs,
                    *args,
                    axes=axes,
                    method=method,
                    label=label,
                    **kwargs,
                )
            )
        return results

    def __str__(self) -> str:
        return f"GraphData({self.inner.name}, {self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"GraphData({self.inner.name}, {self.x}, {self.y})"


# class DataCollection:
#     _data_tuple: tuple[AbstractData]

#     def __init__(self, *data_tuple: AbstractData):
#         self._data_tuple = data_tuple

#     def graph(self, x: "Var", y: "Var"):
#         return DataCollection(*(d.graph(x, y) for d in self))

#     def plot(
#         self,
#         *args,
#         axes: Optional[Union[plt.Axes, Callable[[AbstractData], plt.Axes]]] = None,
#         method: Union[str, Callable[[AbstractData], str]] = "plot",
#         **kwargs,
#     ):
#         for data in self:
#             if isinstance(data, GraphData):
#                 args = (arg(data) if callable(arg) else arg for arg in args)
#                 kwargs = {k: v(data) if callable(v) else v for k, v in kwargs.items()}
#                 axes = axes(data) if callable(axes) else axes  # type: ignore
#                 method = method(data) if callable(method) else method  # type: ignore
#                 data.plot(*args, axes=axes, method=method, **kwargs)

#     def __iter__(self):
#         return iter(self._data_tuple)

#     def __str__(self) -> str:
#         return f"DataCollection{self._data_tuple}"

#     def __repr__(self) -> str:
#         return f"DataCollection{self._data_tuple}"
