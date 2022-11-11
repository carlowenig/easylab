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
from matplotlib.axes import Axes
import pandas as pd
from IPython.display import display


from ..util import Comparable, all_fulfill_type_guard, list_unique, AutoNamed
from ..plot import plot
from ..lang import lang, Text, TextInput
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

    def __getattr__(self, name: str) -> Constraint:
        for constraint in self._constraints:
            if constraint.var.name == name:
                return constraint

        raise AttributeError(f"DataEntry has no var named '{name}'.")

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

    def vars_text(self, *, separator: TextInput = ", "):
        return Text.parse(separator).join(c.text for c in self._constraints)

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
    def update(self, f: Callable[[DataEntry], DataEntryInput]) -> Self:
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
        transform: Optional[Callable[[DataEntry], DataEntryInput]] = None,
    ):
        entries: list[DataEntryInput] = []
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
        transform: Optional[Callable[[DataEntry, str], DataEntryInput]] = None,
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

    def _get_grouped_data_items(self, *vars: Var) -> tuple["SelectedData", ...]:
        if len(vars) == 0:
            return (SelectedData(self, {}),)

        if len(vars) > 1:
            return sum(
                (
                    g._get_grouped_data_items(*vars[1:])
                    for g in self._get_grouped_data_items(vars[0])
                ),
                cast(tuple["SelectedData"], ()),
            )

        var = vars[0]
        constraints = []
        for entry in self.entries:
            if var in entry:
                constraints.append(entry[var])

        return tuple(self.select(constraint) for constraint in set(constraints))
        # return DataCollection(*(self.select({var: value}) for value in set(values)))

    def group_by(self, *vars: Var) -> "GroupedData":
        return GroupedData(*self._get_grouped_data_items(*vars))

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

    def graph(self, x: "Var[_X]", y: "Var[_Y]") -> "GraphData[Self, _X, _Y]":
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
                values = [v.remove_unit() for v in values]

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

    def __init__(
        self, name: Optional[str] = None, entries: Iterable[DataEntryInput] = ()
    ):
        self.__init_auto_named__(name)
        self._entries = [DataEntry.parse(entry) for entry in entries]

    @property
    def name(self) -> str:
        return self._name or super().name

    @property
    def entries(self) -> Iterable[DataEntry]:
        return iter(self._entries)

    def add(self, *entries: DataEntryInput):
        self._entries.extend(DataEntry.parse(entry) for entry in entries)
        return self

    def update(self, f: Callable[[DataEntry], DataEntryInput]) -> Self:
        self._entries = [DataEntry.parse(f(entry)) for entry in self._entries]
        return self

    def remove_where(self, condition: Callable[[DataEntry], bool]) -> Self:
        self._entries = [entry for entry in self._entries if not condition(entry)]
        return self

    def clear(self) -> Self:
        self._entries = []
        return self

    def copy(self):
        return Data(self._name, self._entries)


_D = TypeVar("_D", bound=AbstractData)


class ForwardingData(AbstractData, Generic[_D]):
    inner: _D

    def __init__(self, inner: _D):
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def entries(self) -> Iterable[DataEntry]:
        return self.inner.entries

    def add(self, *entries: DataEntryInput) -> Self:
        self.inner.add(*entries)
        return self

    def update(self, f: Callable[[DataEntry], DataEntryInput]) -> Self:
        self.inner.update(f)
        return self

    def remove_where(self, condition: Callable[[DataEntry], bool]) -> Self:
        self.inner.remove_where(condition)
        return self

    def clear(self) -> Self:
        self.inner.clear()
        return self

    @property
    def hidden_vars(self) -> Set[Var]:
        return self.inner.hidden_vars


class ConditionalData(ForwardingData[_D]):
    @abstractmethod
    def condition(self, entry: DataEntry) -> bool:
        pass

    def __contains__(self, entry: DataEntryInput) -> bool:
        return self.condition(DataEntry.parse(entry))

    @property
    def entries(self) -> Iterable[DataEntry]:
        for entry in self.inner.entries:
            if entry in self:
                yield entry

    def update(self, f: Callable[[DataEntry], DataEntryInput]) -> Self:
        self.inner.update(lambda entry: f(entry) if entry in self else entry)
        return self

    def remove_where(self, condition: Callable[[DataEntry], bool]) -> Self:
        self.inner.remove_where(lambda entry: condition(entry) and entry in self)
        return self


class SelectedData(ConditionalData[_D]):
    selection: DataEntry

    def __init__(self, inner: _D, selection: DataEntryInput):
        self.selection = DataEntry.parse(selection)
        super().__init__(inner)

    @property
    def name(self) -> str:
        return f"{self.inner.name}[{self.selection.vars_str()}]"

    def condition(self, entry: DataEntry):
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


class GraphData(ConditionalData[_D], Generic[_D, _X, _Y]):
    x: Var[_X]
    y: Var[_Y]

    def __init__(self, inner: _D, x: Var[_X], y: Var[_Y]):
        super().__init__(inner)
        self.x = x
        self.y = y

    def condition(self, entry: DataEntry) -> bool:
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
        self,
        *args,
        ax: Optional[plt.Axes] = None,
        method: str = "plot",
        **kwargs,
    ):
        return plot(
            self.x,
            self.y,
            self.value_pairs,
            *args,
            ax=ax,
            method=method,
            **kwargs,
        )

    def copy(self) -> Self:
        return GraphData(self.inner.copy(), self.x, self.y)

    def __str__(self) -> str:
        return f"GraphData({self.inner.name}, {self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"GraphData({self.inner.name}, {self.x}, {self.y})"


def swap_nested_forwarding_data(data: ForwardingData[ForwardingData]):
    copy = data.copy()
    deep_inner = copy.inner.inner

    result = copy.inner
    result.inner = copy
    copy.inner = deep_inner
    return result


class GroupedData(AbstractData, Generic[_D]):
    items: list[SelectedData[_D]]

    def __init__(self, *items: SelectedData[_D]):
        self.items = list(items)

    @property
    def name(self) -> str:
        return "Group(" + ", ".join(group.name for group in self.items) + ")"

    @property
    def entries(self) -> Iterable[DataEntry]:
        for item in self.items:
            yield from item.entries

    def add(self, *entries: DataEntryInput):
        for entry in entries:
            for item in self.items:
                if entry in item:
                    item.add(entry)
                    break
        return self

    def update(self, f: Callable[[DataEntry], DataEntryInput]) -> Self:
        for item in self.items:
            item.update(f)
        return self

    def remove_where(self, condition: Callable[[DataEntry], bool]) -> Self:
        for item in self.items:
            item.remove_where(condition)
        return self

    def clear(self) -> Self:
        for item in self.items:
            item.clear()
        return self

    def copy(self):
        return GroupedData(*(item.copy() for item in self.items))

    def plot(
        self,
        method: str = "plot",
        subplots_kw: dict = {},
        **kwargs,
    ):
        """
        Creates a plot for the contained graphs.

        The plot will be created by the following rules:
        - For each x variable, a separate figure will be created.
        - For each y variable connected to the same x variable, a subplot sharing the same x axis will be created.
        - For each graph with the same x and y variables, a line will be created.
        """

        # Swap items from SelectedData[GraphData] to GraphData[SelectedData], to make them easier to handle
        graphs = [
            cast(
                GraphData[SelectedData, Any, Any],
                swap_nested_forwarding_data(cast(SelectedData[ForwardingData], item)),
            )
            for item in self.items
            if isinstance(item.inner, GraphData)
        ]

        x_graphs: dict[Var, dict[Var, list[GraphData[SelectedData, Any, Any]]]] = {}

        for graph in graphs:
            if graph.x not in x_graphs:
                x_graphs[graph.x] = {}
            y_graphs = x_graphs[graph.x]

            if graph.y not in y_graphs:
                y_graphs[graph.y] = []

            y_graphs[graph.y].append(graph)

        figures = []

        for x, y_graphs in x_graphs.items():
            fig, ax_or_axs = plt.subplots(len(y_graphs), 1, sharex=True, **subplots_kw)
            axs = [ax_or_axs] if isinstance(ax_or_axs, plt.Axes) else ax_or_axs

            for (y, graphs), ax in zip(y_graphs.items(), axs):
                for graph in graphs:
                    graph_kwargs = kwargs | dict(
                        label=graph.inner.selection.vars_text()
                    )
                    graph.plot(ax=ax, method=method, **graph_kwargs)

            figures.append(fig)

        return figures
