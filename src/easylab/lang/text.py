from typing import Any, Callable, Union
from typing_extensions import TypeGuard

TextInput = Union[str, dict[str, str], tuple[str, dict[str, str]], "Text", None]


def is_text_input(input: Any) -> TypeGuard[TextInput]:
    return (
        input is None
        or isinstance(input, (str, Text, dict))
        or (
            isinstance(input, tuple)
            and len(input) == 2
            and isinstance(input[0], str)
            and isinstance(input[1], dict)
        )
    )


class Text:
    _default: str
    _target_strings: dict[str, str]

    def __init__(self, default: str = "", **target_strings: str):
        self._default = default
        self._target_strings = target_strings

    @staticmethod
    def wrapper(default: str = "%", **target_strings: str):
        return Text(default, **target_strings)

    @staticmethod
    def parse(input: TextInput) -> "Text":
        if input is None:
            return Text()
        elif isinstance(input, Text):
            return input
        elif isinstance(input, dict):
            return Text(**input)
        elif (
            isinstance(input, tuple)
            and len(input) == 2
            and isinstance(input[0], str)
            and isinstance(input[1], dict)
        ):
            return Text(input[0], **input[1])
        elif isinstance(input, str):
            return Text(input)
        else:
            return Text(str(input))
        # else:
        #     raise TypeError(f"Cannot parse {input} as Text.")

    @property
    def default(self):
        return self._default

    @property
    def unicode(self):
        return self.string("unicode")

    @property
    def latex(self):
        return self.string("latex")

    def has_target(self, target: str) -> bool:
        return target in self._target_strings

    def string(self, target: str):
        return self._target_strings.get(target, self._default)

    def __getattr__(self, target: str):
        return self.string(target)

    def __str__(self) -> str:
        return self._default

    def __repr__(self) -> str:
        return f"Label({self._default}, {self._target_strings})"

    def __eq__(self, other: "Text") -> bool:
        return (
            isinstance(other, Text)
            and self._default == other._default
            and self._target_strings == other._target_strings
        )

    def __hash__(self) -> int:
        return hash((self._default, self._target_strings))

    def __add__(self, other: TextInput) -> "Text":
        other = Text.parse(other)
        targets = {*self._target_strings.keys(), *other._target_strings.keys()}
        return Text(
            default=self._default + other._default,
            **{
                target: self.string(target) + other.string(target) for target in targets
            },
        )

    def __radd__(self, other: TextInput) -> "Text":
        return Text.parse(other) + self

    def superscript(self, superscript: TextInput):
        return self + Text("^", latex="_{") + superscript + Text(latex="}")

    def subscript(self, subscript: TextInput):
        return self + Text("_", latex="_{") + subscript + Text(latex="}")

    def __getitem__(self, other: TextInput) -> "Text":
        return self.subscript(other)

    def __xor__(self, other: TextInput) -> "Text":
        return self.superscript(other)

    def transform(self, f: Callable[[str, str], str]):
        return Text(
            default=f(self._default, "default"),
            **{
                target: f(string, target)
                for target, string in self._target_strings.items()
            },
        )

    def wrap(self, wrapper: TextInput):
        wrapper = Text.parse(wrapper)
        return self.transform(
            lambda s, target: wrapper.string(target).replace("%%", s)
            if wrapper.has_target(target)
            else s
        )

    def __call__(self, *args: TextInput):
        args = tuple(Text.parse(arg) for arg in args)

        def fill_args(s: str, target: str):
            if len(args) >= 1:
                s = s.replace("%", args[0].string(target))

            for i, arg in enumerate(args):
                s = s.replace(f"%{i+1}", arg.string(target))

            return s

        return self.transform(fill_args)

    def matches(self, query: str):
        return self.default == query or query in self._target_strings.values()
