from __future__ import annotations
from typing import Generic, TypeVar
import numpy as np
import numpy.typing as npt

T = TypeVar("T")


class State(Generic[T]):
    def get_base_states(self) -> list[State[T]]:
        return [self]

    def get_base_state_probabilities(self) -> npt.NDArray[np.float64]:
        return np.array([1.0])

    def get_base_state_probability(self, state: State[T]) -> float:
        base_states = self.get_base_states()

        if state not in base_states:
            return 0.0

        probabilities = self.get_base_state_probabilities()
        return probabilities[base_states.index(state)]

    # def get_value_probability(self, value: T) -> float:
    #     return self.get_base_state_probability(self.get_base_state(value))

    def get_entropy(self) -> float:
        return np.sum(
            -self.get_base_state_probabilities()
            * np.log2(self.get_base_state_probabilities()),
            dtype=np.float64,
        )


class Ensemble(Generic[T]):
    def __init__(self, possible_states: list[State[T]]):
        self.possible_states = possible_states

    def state(self, probabilities: npt.ArrayLike):
        return EnsembleState(self, probabilities)


class EnsembleState(State[T]):
    def __init__(self, ensemble: Ensemble[T], probabilities: npt.ArrayLike):
        self.ensemble = ensemble
        self.probabilities = np.array(probabilities)

    def get_base_states(self) -> list[State[T]]:
        return self.ensemble.possible_states

    def get_base_state_probabilities(self) -> npt.NDArray[np.float64]:
        return self.probabilities


e = Ensemble([State(), State()])

entropy = e.state([0.5, 0.5]).get_entropy()
