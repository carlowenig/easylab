from typing import Any, Optional
from .text import Text

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
math = Text.wrapper(latex=" $%$ ")
display_math = Text.wrapper(latex=" $$%$$ ")
space = Text(" ", latex="\\: ")
small_space = Text(" ", latex="\\, ")

mathrm = Text.wrapper(latex="\\mathrm{%} ")
mathbf = Text.wrapper(latex="\\mathbf{%} ")
mathit = Text.wrapper(latex="\\mathit{%} ")
mathsf = Text.wrapper(latex="\\mathsf{%} ")
mathtt = Text.wrapper(latex="\\mathtt{%} ")
mathcal = Text.wrapper(latex="\\mathcal{%} ")
mathfrak = Text.wrapper(latex="\\mathfrak{%} ")
mathbb = Text.wrapper(latex="\\mathbb{%} ")
emph = Text.wrapper(latex="\\emph{%} ")

# SYMBOLS
pm = plus_minus = Text("+/-", unicode="±", latex="\\pm ")
cdot = Text("⋅", latex="\\cdot ")
times = Text("×", latex="\\times ")
frac = Text("%1 / %2", latex="\\frac{%1}{%2} ")
degree = Text("deg", unicode="°", latex="^\\circ ")

d = Text("d", latex="\\mathrm{d} ")
pd = partial = Text("∂", latex="\\partial ")

dv = derivative = frac(d + "%1", d + "%2")
pdv = partial_derivative = frac(partial + latex_div + "%1", d + latex_div + "%2")

substack = Text("(%1 / %2)", latex="\\substack{%1 \\ %2} ")

par = parentheses = Text("(%)", latex="\\left( % \\right) ")
brack = brackets = Text("[%]", latex="\\left[ % \\right] ")
curly = curly_brackets = Text("{%}", latex="\\left\\{ % \\right\\} ")

lt = less_than = Text("<", latex="\\lt ")
gt = greater_than = Text(">", latex="\\gt ")
leq = less_than_or_equal = Text("<=", latex="\\leq ")
geq = greater_than_or_equal = Text(">=", latex="\\geq ")


def number(value: Any, precision: Optional[int] = None, decimal: str = ",") -> Text:
    if precision is not None:
        s = f"{value:.{precision}f}"
    else:
        s = f"{value:.4g}"

    return Text(s.replace(".", decimal), latex=s.replace(".", "{" + decimal + "}"))
