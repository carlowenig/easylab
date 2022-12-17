from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, TypeVar
import pandas as pd

V = TypeVar("V")


class Query(ABC):
    @abstractmethod
    def execute(self, data: Data) -> Data:
        ...


class AnyQuery(Query):
    def __init__(self, *queries: Any):
        self.queries = queries

    def execute(self, data: Data) -> Data:
        return UnionData(*(data.get(query) for query in self.queries))


def q_any(*queries: Any) -> AnyQuery:
    return AnyQuery(*queries)


class AllQuery(Query):
    def __init__(self, *queries: Any):
        self.queries = queries

    def execute(self, data: Data) -> Data:
        return IntersectionData(*(data.get(query) for query in self.queries))


def q_all(*queries: Any) -> AllQuery:
    return AllQuery(*queries)


# class NotQuery(Query):
#     def __init__(self, query: Any):
#         self.query = query

#     def execute(self, data: Data) -> Data:
#         return data.difference(data.get(self.query))


class ValueConditionQuery(Query):
    def execute(self, data: Data) -> Data:
        if self.check(data.value):
            return data
        else:
            return EmptyData()

    @abstractmethod
    def check(self, value: Any) -> bool:
        ...


class EqualsQuery(ValueConditionQuery):
    def __init__(self, value: Any):
        self.value = value

    def check(self, value: Any) -> bool:
        return value == self.value


class ContainsQuery(ValueConditionQuery):
    def __init__(self, element: Any):
        self.element = element

    def check(self, value: Any) -> bool:
        return self.element in value


class InQuery(ValueConditionQuery):
    def __init__(self, elements: Iterable[Any]):
        self.elements = elements

    def check(self, value: Any) -> bool:
        return value in self.elements


class TransformQuery(Query):
    def __init__(self, query: Any):
        self.query = query

    def execute(self, data: Data) -> Data:
        return self.transform(data.get(self.query))

    @abstractmethod
    def transform(self, data: Data) -> Data:
        ...


class ToListQuery(TransformQuery):
    def transform(self, data: Data) -> Data:
        return ListData(data.value)


class NamedQuery(Query):
    def __init__(self, name: str, query: Any):
        self.name = name
        self.query = query

    def execute(self, data: Data) -> Data:
        raise ValueError(
            f"{type(data).__name__} does not accept named query {self.name}."
        )


_registered_query_factories: dict[str, Callable[[Any], Query]] = {
    "any": AnyQuery,
    "all": AllQuery,
    "contains": ContainsQuery,
    "in": InQuery,
    "equals": EqualsQuery,
    "eq": EqualsQuery,
    "to_list": ToListQuery,
}


def register_query_factory(name: str, factory: Callable[[Any], Query]):
    _registered_query_factories[name] = factory


def deregister_query_factory(name: str):
    del _registered_query_factories[name]


class Data(Generic[V]):
    @staticmethod
    def interpret(input: Any):
        if isinstance(input, Data):
            return input
        elif isinstance(input, dict):
            return DictData(input)
        elif isinstance(input, Iterable):
            return ListData(list(input))
        elif isinstance(input, pd.DataFrame):
            return DataFrameData(input)
        elif isinstance(input, pd.Series):
            return SeriesData(input)
        else:
            return ValueData(input)

    def _get(self, query: Any) -> Data:
        raise NotImplementedError

    def _copy_with_value(self, value: V) -> Data[V]:
        raise NotImplementedError

    @property
    def value(self) -> V:
        raise ValueError(f"Cannont get value of {type(self)}")

    @value.setter
    def value(self, value: V) -> None:
        raise ValueError(f"Cannont set value of {type(self)}")

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def is_not_empty(self) -> bool:
        return self.size > 0

    def get(self, *queries: Any, **named_queries: Any) -> Data:
        for name, query in named_queries.items():
            subnames = name.split("__")
            named_query = query
            for subname in reversed(subnames):
                named_query = NamedQuery(subname, named_query)
            queries += (named_query,)

        if len(queries) == 0:
            # No query, return self
            return self
        elif len(queries) == 1:
            # One query given
            query = queries[0]

            if (
                isinstance(query, NamedQuery)
                and query.name in _registered_query_factories
            ):
                return self.get(_registered_query_factories[query.name](query.query))

            try:
                return self._get(query)
            except NotImplementedError:
                if isinstance(query, Query):
                    return query.execute(self)
                else:
                    return EqualsQuery(query).execute(self)
                    raise ValueError(f"Cannot query {self!r} by {query!r}")
        else:
            # Multiple queries given
            data = self
            for query in queries:
                data = data.get(query)
            return data

    def has(self, *queries: Any, **named_queries: Any) -> bool:
        return self.get(*queries, **named_queries).is_not_empty

    def copy(self, new_value: V | None = None, /) -> Data[V]:
        if new_value is None:
            return self._copy_with_value(self.value)
        else:
            return self._copy_with_value(new_value)

    def __getitem__(self, query: Any):
        if isinstance(query, tuple):
            return self.get(*query).value
        else:
            return self.get(query).value

    def __setitem__(self, query: Any, value: Any):
        if isinstance(query, tuple):
            self.get(*query).value = value
        else:
            self.get(query).value = value

    def __or__(self, other: Any) -> Data:
        return UnionData(self, Data.interpret(other))

    def __ror__(self, other: Any) -> Data:
        return UnionData(Data.interpret(other), self)

    def __and__(self, other: Any) -> Data:
        return IntersectionData(self, Data.interpret(other))

    def __rand__(self, other: Any) -> Data:
        return IntersectionData(Data.interpret(other), self)

    def __repr__(self):
        value_repr = repr(self.value)

        if "\n" in value_repr:
            value_repr = "\n    " + value_repr.replace("\n", "\n    ") + "\n"

        return f"{type(self).__name__}({value_repr})"


class EmptyData(Data[None]):
    def _get(self, query: Any) -> Data:
        return EmptyData()

    @property
    def value(self):
        return None

    @value.setter
    def value(self, value: Any) -> None:
        pass

    @property
    def size(self) -> int:
        return 0

    def __repr__(self):
        return "EmptyData()"


class ValueData(Data[V]):
    def __init__(self, value: V):
        self._value = value

    @property
    def value(self) -> V:
        return self._value

    @value.setter
    def value(self, value: V) -> None:
        self._value = value

    @property
    def size(self) -> int:
        return 1


class CollectedData(Data[tuple[Data[V], ...]]):
    def __init__(self, *items: Data[V]):
        self._items = items
        super().__init__()

    @property
    def value(self) -> tuple[V, ...]:
        return tuple(item.value for item in self._items)

    @value.setter
    def value(self, value: tuple[V, ...]) -> None:
        if len(value) != len(self._items):
            raise ValueError(f"Cannot set value of {self!r}: length mismatch")

        for item, item_value in zip(self._items, value):
            item.value = item_value

    @property
    def size(self) -> int:
        return sum(item.size for item in self._items)


class UnionData(CollectedData[V]):
    def _get(self, query: Any) -> Data:
        return UnionData(*(item.get(query) for item in self._items))


class IntersectionData(CollectedData[V]):
    def _get(self, query: Any) -> Data:
        return IntersectionData(*(item.get(query) for item in self._items))


class ListData(ValueData[list[V]]):
    def __init__(self, values: list[V]):
        super().__init__(values)

    def _get(self, query: Any) -> Data[Any]:
        if isinstance(query, int):
            return ValueData(self.value[query])
        elif isinstance(query, slice):
            return ListData(self.value[query])
        else:
            raise NotImplementedError

    @property
    def size(self) -> int:
        return len(self.value)


class DictData(ValueData[dict[Any, V]]):
    def __init__(self, values: dict[Any, V]):
        super().__init__(values)

    def _get(self, query: Any) -> Data:
        if isinstance(query, NamedQuery):
            if query.name == "key":
                return ValueData(
                    {
                        k: v
                        for k, v in self.value.items()
                        if ValueData(k).has(query.query)
                    }
                )
            elif query.name == "value":
                return ValueData(
                    {
                        k: v
                        for k, v in self._value.items()
                        if ValueData(v).has(query.query)
                    }
                )
            else:
                raise NotImplementedError

        return ValueData(self._value[query])

    @property
    def size(self) -> int:
        return len(self._value)


class DataFrameData(ValueData[pd.DataFrame]):
    def __init__(self, values: Any):
        super().__init__(pd.DataFrame(values))

    def _get(self, query: Any) -> Data:
        if isinstance(query, NamedQuery):
            if query.name in ["col", "column"]:
                return DataFrameData(
                    {
                        col: vals
                        for col, vals in self.value.items()
                        if ValueData(col).has(query.query)
                    }
                )
            elif query.name in ["values", "series"]:
                return DataFrameData(
                    {
                        col: vals
                        for col, vals in self._value.items()
                        if ValueData(vals).has(query.query)
                    }
                )
            else:
                raise NotImplementedError

        result = self._value[query]

        if isinstance(result, pd.DataFrame):
            return DataFrameData(result)
        elif isinstance(result, pd.Series):
            return SeriesData(result)
        else:
            return ValueData(result)

    @property
    def size(self) -> int:
        return self._value.size


class SeriesData(ValueData[pd.Series]):
    def __init__(self, values: Any):
        super().__init__(pd.Series(values))

    def _get(self, query: Any) -> Data:
        return ValueData(self._value[query])

    @property
    def size(self) -> int:
        return self._value.size
