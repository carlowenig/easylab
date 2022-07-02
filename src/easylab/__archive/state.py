from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import Self

from ..util import empty


T = TypeVar("T")
S = TypeVar("S")


# def merge_functions(
#     functions: Iterable[Optional[Callable[..., T]]],
#     merge_results: Callable[[T, T], T] = lambda a, b: b,
# ) -> Optional[Callable[..., T]]:
#     result: Optional[Callable[..., T]] = None

#     for function in functions:
#         if function is not None:
#             result = (
#                 function
#                 if result is None
#                 else lambda *args: merge_results(
#                     cast(Any, result)(*args), cast(Any, function)(*args)
#                 )
#             )

#     return result


# @dataclass
# class VarBlueprint(Generic[T]):
#     default: Optional[T] = None
#     type: Optional[Type[T]] = None
#     check: Optional[Callable[[T], None]] = None
#     parse: Optional[Callable[[Any], T]] = None
#     format: Optional[Callable[[T], str]] = None
#     equal: Optional[Callable[[T, T], bool]] = None

#     def __add__(self, other: "VarBlueprint[T]"):
#         return VarBlueprint(
#             default=other.default or self.default,
#             type=other.type or self.type,
#             check=merge_functions([self.check, other.check]),
#             parse=merge_functions([self.parse, other.parse], lambda a, b: b or a),
#             format=merge_functions([self.format, other.format]),
#             equal=merge_functions([self.equal, other.equal], lambda a, b: a and b),
#         )

#     def __call__(self, name: str) -> "Var[T]":
#         return Var.from_blueprint(self, name)


class Var(Generic[T]):
    name: str
    default: Optional[T] = None
    type: Optional[Type[T]] = None
    _check_func: Optional[Callable[[T], None]] = None
    _parse_func: Optional[Callable[[Any], T]] = None
    _format_func: Optional[Callable[[T], str]] = None
    _equal_func: Optional[Callable[[T, T], bool]] = None
    _merge_func: Optional[Callable[[T, T], T]] = None

    def __init__(
        self,
        name: str,
        *,
        type: Optional[Type[T]] = None,
        default: Optional[T] = None,
        check: Optional[Callable[[T], None]] = None,
        parse: Optional[Callable[[Any], T]] = None,
        format: Optional[Callable[[T], str]] = None,
        equal: Optional[Callable[[T, T], bool]] = None,
        merge: Optional[Callable[[T, T], T]] = None,
    ):
        self.name = name

        # Only set fields if not None to avoid overriding default values in subclass.
        if type is not None:
            self.type = type
        if default is not None:
            self.default = default

        self._check_func = check
        self._parse_func = parse
        self._format_func = format
        self._equal_func = equal
        self._merge_func = merge

    # @staticmethod
    # def from_blueprint(blueprint: VarBlueprint[T], name: str) -> "Var[T]":
    #     return Var(
    #         name=name,
    #         type=blueprint.type,
    #         default=blueprint.default,
    #         check=blueprint.check,
    #         parse=blueprint.parse,
    #         format=blueprint.format,
    #         equal=blueprint.equal,
    #     )

    # def to_blueprint(self):
    #     return VarBlueprint(
    #         default=self.default,
    #         type=self.type,
    #         check=self._check_func,
    #         parse=self._parse_func,
    #         format=self._format_func,
    #         equal=self._equal_func,
    #     )

    def check(self, value: T, *, check_type: bool = True) -> None:
        if check_type and self.type is not None and not isinstance(value, self.type):
            raise TypeError(
                f"Value of variable {self.name} must be of type {self.type}."
            )

        if self._check_func is not None:
            self._check_func(value)

    def parse(self, input: Any, *, check: bool = True, use_default: bool = True):
        if use_default and input is None and self.default is not None:
            return self.default

        if self._parse_func is not None:
            value = self._parse_func(input)
        elif self.type is not None and isinstance(input, self.type):
            value = cast(T, input)
        elif callable(self.type):
            value = cast(T, self.type(input))
        else:
            raise ValueError(
                f"Cannot parse input {input} of type {type(input)} as value of variable {self.name}."
            )

        if check:
            self.check(value)

        return value

    def format(self, value: T) -> str:
        if self._format_func is not None:
            return self._format_func(value)
        else:
            return str(value)

    def equal(self, a: T, b: T):
        if self._equal_func is not None:
            return self._equal_func(a, b)
        else:
            return a == b

    def merge(self, a: T, b: T):
        if self._merge_func is not None:
            return self._merge_func(a, b)
        elif self.equal(a, b):
            return a
        else:
            raise ValueError(
                f"Values {a} and {b} of variable {self.name} cannot be merged."
            )

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class Unknown:
    _instance: Optional["Unknown"] = None

    def __new__(cls: Type[Self]) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


unknown = Unknown()


class State:
    __slots__ = ["_values"]

    _values: dict[Var, Any]

    def __init__(self, values: dict[Var, Any] = {}):
        self._values = {var: var.parse(input) for var, input in values.items()}

    @property
    def values(self) -> dict[Var, Any]:
        return self._values.copy()

    @property
    def vars(self) -> Iterable[Var]:
        return self._values.keys()

    @property
    def is_empty(self):
        return len(self._values) == 0

    def contains_var(self, var: Union[Var, str]) -> bool:
        for v in self._values:
            if v == var or v.name == var:
                return True
        return False

    def var(self, var: Union[Var, str]) -> Var:
        for v in self._values:
            if v == var or v.name == var:
                return v
        raise KeyError(f"State does not contain variable {var}.")

    @overload
    def __getitem__(self, var: Var[T]) -> Union[T, Unknown]:
        ...

    @overload
    def __getitem__(self, var: str) -> Any:
        ...

    def __getitem__(self, var: Union[Var[T], str]) -> Union[T, Any]:
        if isinstance(var, Var):
            if var not in self._values:
                return unknown
            return cast(T, self._values[var])
        else:
            for var in self._values.keys():
                if var.name == var:
                    return self._values[var]
            return unknown

    def __getattr__(self, name: str) -> Any:
        return self[name]

    def __repr__(self) -> str:
        return f"State(values={self._values})"

    def to_str(
        self, *, relevant_vars: Optional[Iterable[Var]] = None, parens: bool = True
    ):
        ellipsis = relevant_vars is not None and set(relevant_vars) != set(self.vars)

        if relevant_vars is None:
            relevant_vars = self.vars

        str = ", ".join(
            f"{var.name} = {var.format(value)}"
            for var, value in self._values.items()
            if var in relevant_vars
        )
        if ellipsis:
            str += ", ..."
        if parens:
            str = f"( {str} )"
        return str

    def __str__(self) -> str:
        return self.to_str()

    def merge(self, other: "State"):
        values = self._values.copy()

        for var, value in other._values.items():
            if var in values:
                value = var.merge(values[var], value)

            values[var] = value

        return State(values)

    def __add__(self, other: "State") -> "State":
        return self.merge(other)

    def get_connected_vars(self, other: "State") -> Iterable[Var]:
        for var, value in other._values.items():
            if var in self._values and var.equal(value, self._values[var]):
                yield var

    def is_connected(self, other: "State") -> bool:
        return not empty(self.get_connected_vars(other))


StateCondition = Union[
    Callable[[State], bool], Var[T], str, dict[Union[Var, str], Any], None
]


def partition_states(states: Iterable[State]):
    partitions: list[State] = []

    for state in states:
        found_partition = None
        for i, partition in enumerate(partitions):

            if state.is_connected(partition):
                if found_partition is not None:
                    raise ValueError(
                        f"Found multiple partitions for state {state}: {found_partition}, {partition}"
                    )

                partitions[i] = partition + state
                found_partition = partition

        if found_partition is None:
            partitions.append(state)

    return partitions


class Series:
    _states: list[State]

    def __init__(self, *states: Union[State, dict[Var, Any]]):
        self._states = partition_states(
            state if isinstance(state, State) else State(state) for state in states
        )
        # self._check_all()

    def __iter__(self) -> Iterator[State]:
        return iter(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, index: int) -> State:
        return self._states[index]

    def add(self, *states: State) -> None:
        self._states = partition_states([*self._states, *states])

    @overload
    def states_where(self, condition: Callable[[State], bool]) -> Iterable[State]:
        ...

    @overload
    def states_where(self, condition: Var[T], value: T) -> Iterable[State]:
        ...

    @overload
    def states_where(self, condition: str, value: Any) -> Iterable[State]:
        ...

    @overload
    def states_where(self, condition: dict[Union[Var, str], Any]) -> Iterable[State]:
        ...

    @overload
    def states_where(self, **kwargs: Any) -> Iterable[State]:
        ...

    @overload
    def states_where(
        self,
        condition: StateCondition[T] = None,
        value: Optional[T] = None,
        **kwargs: Any,
    ) -> Iterable[State]:
        ...

    def states_where(
        self,
        condition: StateCondition[T] = None,
        value: Optional[T] = None,
        **kwargs: Any,
    ) -> Iterable[State]:

        if isinstance(condition, (Var, str)):
            for state in self._states:
                var = state.var(condition)
                if state.contains_var(condition) and var.equal(
                    state[condition], var.parse(value)
                ):
                    yield state

        elif isinstance(condition, dict):
            for state in self._states:
                if all(
                    state.contains_var(var)
                    and state.var(var).equal(state[var], state.var(var).parse(value))
                    for var, value in condition.items()
                ):
                    yield state

        elif callable(condition):
            for state in self._states:
                if condition(state):
                    yield state

        elif condition is None:
            return self.states_where(cast(dict[Union[Var, str], Any], kwargs))

        else:
            raise TypeError(
                f"Expected a Var, str, callable, dict or None as first argument. Got {type(condition)}."
            )

    @overload
    def where(self, condition: Callable[[State], bool]) -> "Series":
        ...

    @overload
    def where(self, condition: Var[T], value: T) -> "Series":
        ...

    @overload
    def where(self, condition: str, value: Any) -> "Series":
        ...

    @overload
    def where(self, condition: dict[Union[Var, str], Any]) -> "Series":
        ...

    @overload
    def where(self, **kwargs: Any) -> "Series":
        ...

    @overload
    def where(
        self,
        condition: StateCondition[T] = None,
        value: Optional[T] = None,
        **kwargs: Any,
    ) -> "Series":
        ...

    def where(
        self,
        condition: StateCondition[T] = None,
        value: Optional[T] = None,
        **kwargs: Any,
    ) -> "Series":
        return Series(*self.states_where(condition, value, **kwargs))

    @overload
    def find(self, condition: Callable[[State], bool]) -> State:
        ...

    @overload
    def find(self, condition: Union[Var, str], value: Any) -> State:
        ...

    @overload
    def find(self, condition: dict[Union[Var, str], Any]) -> State:
        ...

    @overload
    def find(
        self,
        condition: StateCondition[T] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> State:
        ...

    def find(
        self,
        condition: StateCondition[T] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> State:
        return sum(self.states_where(condition, value, **kwargs), State())

    # def _check_single(self, state: State):
    #     for other in self._states:
    #         if state is not other and not state.is_compatible(other):
    #             relevant_vars = [
    #                 *state.get_connected_vars(other),
    #                 *state.get_incompatible_vars(other),
    #             ]
    #             raise ValueError(
    #                 f"Found conflicting states: {state.to_str(relevant_vars=relevant_vars)} vs {other.to_str(relevant_vars=relevant_vars)}. Reasons:\n  - "
    #                 + "\n  - ".join(state.get_incompatibility_reasons(other))
    #             )

    # def _check_all(self):
    #     for state in self._states:
    #         self._check_single(state)

    def __str__(self) -> str:
        return "Series:\n" + "\n".join(f"  {state}" for state in self._states)
