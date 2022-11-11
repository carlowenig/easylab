from ..dispatch import *

diff = Dispatcher("diff")
anti_diff = Dispatcher("anti_diff")

integrate = Dispatcher(
    "integrate",
    [
        Handler(
            lambda f, a, b: anti_diff(f)(b) - anti_diff(f)(a),
            name="evaluate_anti_diff",
        )
    ],
)

select_arg = Dispatcher(
    "select_arg",
    [
        Handler(
            lambda f, i: lambda arg: lambda *args, **kwargs: f(
                *args[:i], arg, *args[i + 1 :], **kwargs
            ),
            name="select_arg",
        )
    ],
)
select_kwarg = Dispatcher(
    "select_kwarg",
    [
        Handler(
            lambda f, key: lambda arg: lambda *args, **kwargs: f(
                *args, **(kwargs | {key: arg})
            ),
            name="select_kwarg",
        )
    ],
)
