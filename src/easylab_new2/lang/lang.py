from __future__ import annotations
from typing import Any, Callable, Iterable, Literal, Optional, TypeGuard


TextTarget = Literal["ascii", "unicode", "latex"]
text_targets: list[TextTarget] = ["ascii", "unicode", "latex"]


def is_text_target(input: Any) -> TypeGuard[TextTarget]:
    return input in text_targets


class Text:
    @staticmethod
    def interpret(input: Any):
        if isinstance(input, Text):
            return input
        elif isinstance(input, dict):
            return Text(**input)
        elif hasattr(input, "__text__"):
            return Text.interpret(input.__text__())
        elif hasattr(input, "text"):
            return Text.interpret(getattr(input, "text"))
        else:
            return Text(str(input))

    def __init__(self, fallback: str | None = None, **kwargs: str):
        self._target_strings: dict[TextTarget, str] = {}

        if fallback is not None:
            for target in text_targets:
                self._target_strings[target] = fallback

        for target, string in kwargs.items():
            if not is_text_target(target):
                raise ValueError(f"Invalid text target {target}.")
            self._target_strings[target] = string

    @property
    def ascii(self):
        return self.get("ascii")

    @property
    def unicode(self):
        return self.get("unicode")

    @property
    def latex(self):
        return self.get("latex")

    def specifies_target(self, target: TextTarget) -> bool:
        return target in self._target_strings

    @property
    def specified_targets(self):
        return set(self._target_strings.keys())

    def get(self, *targets: TextTarget, fallback: str | None = None) -> str:
        for target in targets:
            if target in self._target_strings:
                return self._target_strings[target]
        if fallback is not None:
            return fallback
        raise ValueError(
            f"{repr(self)} does not specify a string for any of the given targets {targets}."
        )

    def __getattr__(self, target: TextTarget):
        return self.get(target)

    def __repr__(self) -> str:
        return f"Text({self._target_strings})"

    def __eq__(self, other: Text) -> bool:
        return isinstance(other, Text) and self._target_strings == other._target_strings

    def __hash__(self) -> int:
        return hash(self._target_strings)

    def __add__(self, other: Any) -> Text:
        other = text(other)
        targets: set[TextTarget] = {
            *self._target_strings.keys(),
            *other._target_strings.keys(),
        }
        return Text(
            **{target: self.get(target) + other.get(target) for target in targets},
        )

    def __radd__(self, other: Any) -> "Text":
        return text(other) + self

    def superscript(self, superscript: Any):
        return self + Text("^", latex="_{") + text(superscript) + Text("", latex="}")

    def subscript(self, subscript: Any):
        subscript_text = text(subscript)
        subscript_ascii = subscript_text.ascii

        if " " in subscript_ascii and not (
            subscript_ascii.startswith("(") and subscript_ascii.endswith(")")
        ):
            lpar = "("
            rpar = ")"
        else:
            lpar = ""
            rpar = ""
        return (
            self
            + Text("_" + lpar, latex="_{")
            + text(subscript)
            + Text(rpar, latex="}")
        )

    def __getitem__(self, other: Any) -> "Text":
        return self.subscript(other)

    def __xor__(self, other: Any) -> "Text":
        return self.superscript(other)

    def transform(self, f: Callable[[str, TextTarget], str]):
        return Text(
            **{
                target: f(string, target)
                for target, string in self._target_strings.items()
            },
        )

    def __call__(self, *args: Any):
        args = tuple(text(arg) for arg in args)

        def fill_args(s: str, target: TextTarget):
            if len(args) >= 1:
                s = s.replace("%", args[0].get(target))

            for i, arg in enumerate(args):
                s = s.replace(f"%{i+1}", arg.get(target))

            return s

        return self.transform(fill_args)

    def matches(self, query: Any, *, try_text=True, case_sensitive=False):
        # Since query could be a Var, which has a non-standard implementation of __eq__,
        # we need to check the type of query before all equality checks.

        if query is self or (isinstance(query, Text) and query == self):
            return True
        if isinstance(query, str) and query in self._target_strings.values():
            return True

        # Try lower case comparison
        if not case_sensitive and isinstance(query, str):
            for string in self._target_strings.values():
                if query.lower() == string.lower():
                    return True

        if try_text:
            query_text = text(query)
            for string in query_text._target_strings.values():
                return self.matches(string, try_text=False)

        return False

    def join(self, parts: Iterable[Any]) -> Text:
        parts = [text(part) for part in parts]

        targets = set(self.specified_targets)
        for part in parts:
            targets.update(part.specified_targets)

        target_strings: dict[str, str] = {}
        for target in targets:
            target_strings[target] = self.get(target).join(
                part.get(target) for part in parts
            )

        return Text(**target_strings)

    def replace(self, old: str | Text, new: str | Text):
        return Text(
            **{
                target: string.replace(
                    old if isinstance(old, str) else old.get(target),
                    new if isinstance(new, str) else new.get(target),
                )
                for target, string in self._target_strings.items()
            },
        )

    def __contains__(self, s: str | Text):
        if isinstance(s, Text):
            return all(
                string in self._target_strings[target]
                for target, string in s._target_strings.items()
            )
        else:
            return all(s in string for string in self._target_strings.values())


class TextPrinter:
    def __init__(self, name: str, targets: list[TextTarget]):
        self.name = name
        self.targets = targets

    def __call__(self, input: Any, fallback: str | None = None):
        return text(input).get(*self.targets, fallback=fallback)

    def __repr__(self):
        return f"TextPrinter({self.name}, targets={self.targets})"

    def __str__(self):
        return f"TextPrinter({self.name})"


ascii = TextPrinter("ascii", ["ascii", "unicode", "latex"])
unicode = TextPrinter("unicode", ["unicode", "ascii", "latex"])
latex = TextPrinter("latex", ["latex", "ascii", "unicode"])


class lang:
    # GREEK SYMBOLS
    alpha = Text("α", latex="\\alpha ")
    beta = Text("β", latex="\\beta ")
    gamma = Text("γ", latex="\\gamma ")
    delta = Text("δ", latex="\\delta ")
    epsilon = Text("ε", latex="\\epsilon ")
    varepsilon = Text("ε", latex="\\varepsilon ")
    zeta = Text("ζ", latex="\\zeta ")
    eta = Text("η", latex="\\eta ")
    theta = Text("θ", latex="\\theta ")
    iota = Text("ι", latex="\\iota ")
    kappa = Text("κ", latex="\\kappa ")
    lambda_ = Text("λ", latex="\\lambda ")
    mu = Text("μ", latex="\\mu ")
    nu = Text("ν", latex="\\nu ")
    xi = Text("ξ", latex="\\xi ")
    omicron = Text("ο", latex="\\omicron ")
    pi = Text("π", latex="\\pi ")
    rho = Text("ρ", latex="\\rho ")
    sigma = Text("σ", latex="\\sigma ")
    tau = Text("τ", latex="\\tau ")
    upsilon = Text("υ", latex="\\upsilon ")
    phi = Text("φ", latex="\\phi ")
    varphi = Text("φ", latex="\\varphi ")
    chi = Text("χ", latex="\\chi ")
    psi = Text("ψ", latex="\\psi ")
    omega = Text("ω", latex="\\omega ")

    Alpha = Text("Α", latex="\\Alpha ")
    Beta = Text("Β", latex="\\Beta ")
    Gamma = Text("Γ", latex="\\Gamma ")
    Delta = Text("Δ", latex="\\Delta ")
    Epsilon = Text("Ε", latex="\\Epsilon ")
    Zeta = Text("Ζ", latex="\\Zeta ")
    Eta = Text("Η", latex="\\Eta ")
    Theta = Text("Θ", latex="\\Theta ")
    Iota = Text("Ι", latex="\\Iota ")
    Kappa = Text("Κ", latex="\\Kappa ")
    Lambda = Text("Λ", latex="\\Lambda ")
    Mu = Text("Μ", latex="\\Mu ")
    Nu = Text("Ν", latex="\\Nu ")
    Xi = Text("Ξ", latex="\\Xi ")
    Omicron = Text("Ο", latex="\\Omicron ")
    Pi = Text("Π", latex="\\Pi ")
    Rho = Text("Ρ", latex="\\Rho ")
    Sigma = Text("Σ", latex="\\Sigma ")
    Tau = Text("Τ", latex="\\Tau ")
    Upsilon = Text("Υ", latex="\\Upsilon ")
    Phi = Text("Φ", latex="\\Phi ")
    Chi = Text("Χ", latex="\\Chi ")
    Psi = Text("Ψ", latex="\\Psi ")
    Omega = Text("Ω", latex="\\Omega ")

    # UTILITY LABELS
    latex_div = Text(latex=" ")
    math = Text(latex=" $%$ ")
    display_math = Text(latex=" $$%$$ ")
    space = Text(" ", latex="\\: ")
    small_space = Text(" ", latex="\\, ")
    large_space = Text("    ", latex="\\quad ")

    mathrm = Text(latex="\\mathrm{%} ")
    mathbf = Text(latex="\\mathbf{%} ")
    mathit = Text(latex="\\mathit{%} ")
    mathsf = Text(latex="\\mathsf{%} ")
    mathtt = Text(latex="\\mathtt{%} ")
    mathcal = Text(latex="\\mathcal{%} ")
    mathfrak = Text(latex="\\mathfrak{%} ")
    mathbb = Text(latex="\\mathbb{%} ")
    emph = Text(latex="\\emph{%} ")

    # SYMBOLS
    pm = plus_minus = Text("+/-", unicode="±", latex="\\pm ")
    cdot = Text("⋅", latex="\\cdot ")
    times = Text("×", latex="\\times ")
    frac = Text("%1 / %2", latex="\\frac{%1}{%2} ")
    degree = Text("deg", unicode="°", latex="^\\circ ")

    d = Text("d", latex="\\mathrm{d} ")
    pd = partial = Text("∂", latex="\\partial ")

    dv = derivative = Text("d%1/d%2", latex="\\frac{%1}{%2}")
    pdv = partial_derivative = Text(
        "∂%1/∂%2", latex="\\frac{\\partial %1}{\\partial %2}"
    )

    substack = Text("(%1 / %2)", latex="\\substack{%1 \\ %2} ")

    par = parentheses = Text("(%)", latex="\\left( % \\right) ")
    brack = brackets = Text("[%]", latex="\\left[ % \\right] ")
    curly = curly_brackets = Text("{%}", latex="\\left\\{ % \\right\\} ")

    lt = less_than = Text("<", latex="\\lt ")
    gt = greater_than = Text(">", latex="\\gt ")
    leq = less_than_or_equal = Text("<=", latex="\\leq ")
    geq = greater_than_or_equal = Text(">=", latex="\\geq ")

    newline = Text("\n", latex="\\\\ ")

    left_arrow = larrow = Text("←", latex="\\leftarrow ")
    right_arrow = rarrow = Text("→", latex="\\rightarrow ")
    left_right_arrow = lrarrow = Text("↔", latex="\\leftrightarrow ")
    left_double_arrow = ldarrow = Text("⇐", latex="\\Leftarrow ")
    right_double_arrow = rdarrow = Text("⇒", latex="\\Rightarrow ")
    left_right_double_arrow = lrdarrow = Text("⇔", latex="\\Leftrightarrow ")

    @staticmethod
    def number(value: Any, precision: Optional[int] = None, decimal: str = ",") -> Text:
        if precision is not None:
            s = f"{value:.{precision}f}"
        else:
            s = f"{value:.4g}"

        return Text(s.replace(".", decimal), latex=s.replace(".", "{" + decimal + "}"))


def text(input: Any) -> Text:
    if isinstance(input, Text):
        if "%{lang" in input:
            # TODO: Make this more efficient using regex matching
            for name, value in lang.__dict__.items():
                if isinstance(value, Text):
                    input = input.replace("%{lang." + name + "}", value)
        return input
    elif input is None:
        return text("")
    elif isinstance(input, str):
        return Text(input)
    elif hasattr(input, "__text__"):
        return text(getattr(input, "__text__")())
    elif isinstance(input, list):
        return text(lang.brack(Text(", ").join(input)))
    elif isinstance(input, set):
        return text(lang.curly(Text(", ").join(input)))
    elif isinstance(input, dict):
        return text(
            lang.curly(
                Text(", ").join(
                    text(k) + ":" + lang.space + text(v) for k, v in input.items()
                )
            )
        )
    else:
        return text(str(input))


__all__ = [
    "TextTarget",
    "text_targets",
    "is_text_target",
    "Text",
    "TextPrinter",
    "ascii",
    "unicode",
    "latex",
    "lang",
    "text",
]
