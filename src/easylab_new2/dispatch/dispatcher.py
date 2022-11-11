from __future__ import annotations
import operator
import time
from typing import Any, Callable, Generic, Iterable, TypeVar, cast, overload

T = TypeVar("T")


class Handler(Generic[T]):
    def __init__(
        self,
        function: Callable[..., T],
        priority: float = 0.0,
        name: str | None = None,
        catch: Iterable[type[Exception]] = (),
    ):
        self.function = function
        self.name = str(name) if name is not None else function.__name__
        self.priority = float(priority)
        self.catch = tuple(catch)

    def __call__(self, *args, **kwargs) -> T:
        try:
            return self.function(*args, **kwargs)
        except NoHandlerFoundException as e:
            raise StopHandlerException(
                f"nested dispatcher {e.dispatcher.name} did not found handler for args {format_args(*e.args, **e.kwargs)}: "
                + e.get_formatted_stop_reasons()
            )
        except Exception as e:
            if len(self.catch) > 0 and isinstance(e, self.catch):
                raise StopHandlerException(f"caught {type(e).__name__}: {e}")
            else:
                raise

    def __repr__(self):
        return f"Handler(function={self.function!r}, name={self.name!r}, priority={self.priority!r})"

    def __str__(self):
        return self.name


def operator_handler(op: str):
    return Handler(
        lambda *args: getattr(operator, op)(*args),
        name=op + "_operator",
        catch=(TypeError,),
    )


commute_args_handler = Handler(
    lambda a, b: get_dispatcher()(b, a),
    name="commute_args",
)


def __associative_handler_func(*args):
    if len(args) == 0:
        return NotImplemented

    if len(args) == 1:
        return args[0]

    dispatcher = get_dispatcher()

    return dispatcher(args[0], dispatcher(*args[1:]))


associative_handler = Handler(__associative_handler_func, name="associative")


class StopHandlerException(Exception):
    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason

    def __str__(self):
        return f"Handler was stopped." + (
            f" Reason: {self.reason}" if self.reason else ""
        )


def stop_handler(reason: str | None = None):
    raise StopHandlerException(reason)


def expect(condition: bool, reason: str | None = None):
    if not condition:
        raise StopHandlerException(reason)


def expect_type(value: Any, type_: type[T], reason: str | None = None) -> T:
    if not isinstance(value, type_):
        raise StopHandlerException(
            f"expected type {type_.__name__}, got {type(value).__name__}"
            + (f": {reason}" if reason else "")
        )
    return cast(T, value)


class NoHandlerFoundException(Exception):
    def __init__(
        self, dispatcher: Dispatcher, args: tuple, kwargs: dict, stop_reasons: dict
    ) -> None:
        self.dispatcher = dispatcher
        self.args = args
        self.kwargs = kwargs
        self.stop_reasons = stop_reasons

    def __str__(self):
        return (
            f"No handler for {self.dispatcher.name}({format_args(*self.args, **self.kwargs)}) found. "
            + self.get_formatted_stop_reasons()
        )

    def get_formatted_stop_reasons(self):
        return "Stop reasons:" + (
            "".join(
                f"\n  - \u001b[38;5;248m[{handler.priority:.2f}]\033[0m \u001b[1m{handler.name}:\033[0m "
                + f"{self.stop_reasons.get(handler) or 'unknown'}".replace(
                    "\n", "\n           "
                )
                for handler in self.dispatcher.handlers
            )
            if len(self.stop_reasons) > 0
            else " none"
        )


def format_args(*args, **kwargs):
    return ", ".join(
        [repr(arg) for arg in args]
        + [f"{key}={value!r}" for key, value in kwargs.items()]
    )


class DispatcherMiddleware:
    def on_call(self, context: DispatcherContext) -> None:
        pass

    def on_result(self, context: DispatcherContext) -> None:
        pass


class LogMiddleware(DispatcherMiddleware):
    def on_call(self, context: DispatcherContext):
        print(f"Calling dispatcher {context.get_dispatcher_call_string()}.")

    def on_result(self, context: DispatcherContext):
        print(
            f"Handler call {context.get_handler_call_string()} of dispatcher {context.dispatcher.name!r} returned {context.result!r}."
        )


class MeasureMiddleware(DispatcherMiddleware):
    def on_call(self, context: DispatcherContext):
        context.data["call_start"] = time.perf_counter()

    def on_result(self, context: DispatcherContext):
        t = time.perf_counter()
        if "call_start" in context.data:
            dt = t - context.data["call_start"]
            print(
                f"Dispatcher call {context.get_dispatcher_call_string()} took {dt * 1000:.3f}ms."
            )
            del context.data["call_start"]


class DispatcherContext:
    def __init__(
        self,
        dispatcher: Dispatcher,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        parent: DispatcherContext | None = None,
    ):
        self.dispatcher = dispatcher
        self.args = list(args)
        self.kwargs = kwargs
        self.parent = parent
        self.active_handler: Handler | None = None
        self.stop_handler_exceptions: dict[Handler, StopHandlerException] = {}
        self.result: Any = None
        self.active = True
        self.data = {}

    def get_context_stack(self):
        context = self
        while context is not None:
            yield context
            context = context.parent

    def get_active_handlers(self):
        for context in self.get_context_stack():
            if context.active_handler is not None:
                yield context.active_handler

    def get_dispatcher_call_string(self, handlers: bool = True):
        return " -> ".join(
            context.dispatcher.name
            + (
                f"\u001b[38;5;248m[{context.active_handler.name}]\033[0m"
                if handlers and context.active_handler is not None
                else ""
            )
            + f"({format_args(*context.args, **context.kwargs)})"
            for context in self.get_context_stack()
        )

    def get_handler_call_string(self):
        return " -> ".join(
            f"{context.active_handler.name}({format_args(*context.args, **context.kwargs)})"
            for context in self.get_context_stack()
            if context.active_handler is not None
        )


__current_dispatcher_context: DispatcherContext | None = None


def push_dispatcher_context(dispatcher: Dispatcher, args: tuple, kwargs: dict):
    global __current_dispatcher_context
    __current_dispatcher_context = DispatcherContext(
        dispatcher, args, kwargs, __current_dispatcher_context
    )
    return __current_dispatcher_context


def pop_dispatcher_context(context: DispatcherContext):
    global __current_dispatcher_context
    if context is not __current_dispatcher_context:
        raise ValueError("popped context is not the current context")

    if __current_dispatcher_context is not None:
        __current_dispatcher_context.active = False
        __current_dispatcher_context = __current_dispatcher_context.parent


def get_dispatcher_context() -> DispatcherContext:
    if __current_dispatcher_context is None:
        raise ValueError("no dispatcher context is active")
    return __current_dispatcher_context


def get_dispatcher():
    return get_dispatcher_context().dispatcher


class Dispatcher:
    def __init__(
        self,
        name: str,
        handlers: Iterable[Handler] = [],
        middleware: Iterable[DispatcherMiddleware] = [],
    ):
        self.name = name
        self.handlers = list(handlers)
        self.middleware = list(middleware)

    @overload
    def add_handler(self, handler: Handler[T], /) -> Handler[T]:
        ...

    @overload
    def add_handler(
        self,
        /,
        *,
        name: str | None = None,
        priority: float = 0.0,
        catch: Iterable[type[Exception]] = (),
    ) -> Callable[[Callable[..., T]], Handler[T]]:
        """Decorator for adding handlers to the dispatcher."""

    @overload
    def add_handler(
        self,
        function: Callable[..., T],
        /,
        *,
        name: str | None = None,
        priority: float = 0.0,
        catch: Iterable[type[Exception]] = (),
    ) -> Handler[T]:
        ...

    def add_handler(
        self,
        arg: Callable[..., T] | None = None,
        /,
        *,
        name: str | None = None,
        priority: float = 0.0,
        catch: Iterable[type[Exception]] = (),
    ) -> Handler[T] | Callable[[Callable[..., T]], Handler[T]]:
        if arg is None:
            return lambda f: self.add_handler(
                f, name=name, priority=priority, catch=catch
            )
        elif isinstance(arg, Handler):
            if name is not None:
                raise TypeError("Cannot specify name when first argument is a Handler.")
            if priority != 0.0:
                raise TypeError(
                    "Cannot specify priority when first argument is a Handler."
                )
            if catch != ():
                raise TypeError(
                    "Cannot specify catch when first argument is a Handler."
                )

            handler = arg
        else:
            handler = Handler(arg, priority=priority, name=name, catch=catch)

        self.handlers.append(handler)
        self.handlers.sort(key=lambda h: h.priority)

        return handler

    def remove_handler(self, handler: Handler | str):
        if isinstance(handler, Handler):
            self.handlers.remove(handler)
        elif isinstance(handler, str):
            self.handlers = [h for h in self.handlers if h.name != handler]
        else:
            raise TypeError(f"Expected Handler or str, got {type(handler).__name__}.")

    def add_middleware(self, middleware: DispatcherMiddleware):
        self.middleware.append(middleware)

    def __call__(self, *args, **kwargs):
        context = push_dispatcher_context(self, args, kwargs)

        for middleware in self.middleware:
            middleware.on_call(context)

        for handler in self.handlers:
            if handler in context.get_active_handlers():
                context.stop_handler_exceptions[handler] = StopHandlerException(
                    "handler recursively called itself"
                )
            else:
                context.active_handler = handler
                try:
                    context.result = handler(*args, **kwargs)

                    for middleware in self.middleware:
                        middleware.on_result(context)

                    if context.result is NotImplemented:
                        raise StopHandlerException("returned NotImplemented")

                    pop_dispatcher_context(context)

                    return context.result
                except StopHandlerException as e:
                    context.stop_handler_exceptions[handler] = e

        pop_dispatcher_context(context)

        raise NoHandlerFoundException(
            self,
            args,
            kwargs,
            {
                handler: e.reason
                for handler, e in context.stop_handler_exceptions.items()
            },
        )


add = Dispatcher(
    "add", [operator_handler("add"), commute_args_handler, associative_handler]
)
sub = Dispatcher("sub", [operator_handler("sub"), associative_handler])
mul = Dispatcher(
    "mul", [operator_handler("mul"), commute_args_handler, associative_handler]
)
div = Dispatcher("div", [operator_handler("truediv"), associative_handler])
pow = Dispatcher("pow", [operator_handler("pow")])
mod = Dispatcher("mod", [operator_handler("mod")])
neg = Dispatcher("neg", [operator_handler("neg")])
pos = Dispatcher("pos", [operator_handler("pos")])
abs = Dispatcher("abs", [operator_handler("abs")])
invert = Dispatcher("invert", [operator_handler("invert")])
lshift = Dispatcher("lshift", [operator_handler("lshift")])
rshift = Dispatcher("rshift", [operator_handler("rshift")])
and_ = Dispatcher(
    "and", [operator_handler("and"), commute_args_handler, associative_handler]
)
or_ = Dispatcher(
    "or", [operator_handler("or"), commute_args_handler, associative_handler]
)
xor = Dispatcher(
    "xor", [operator_handler("xor"), commute_args_handler, associative_handler]
)
eq = Dispatcher("eq", [operator_handler("eq"), commute_args_handler])
ne = Dispatcher("ne", [operator_handler("ne"), commute_args_handler])
lt = Dispatcher("lt", [operator_handler("lt")])
le = Dispatcher("le", [operator_handler("le")])
gt = Dispatcher("gt", [operator_handler("gt")])
ge = Dispatcher("ge", [operator_handler("ge")])


class DispatchedOperators:
    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __pow__(self, other):
        return pow(self, other)

    def __mod__(self, other):
        return mod(self, other)

    def __neg__(self):
        return neg(self)

    def __pos__(self):
        return pos(self)

    def __abs__(self):
        return abs(self)

    def __invert__(self):
        return invert(self)

    def __lshift__(self, other):
        return lshift(self, other)

    def __rshift__(self, other):
        return rshift(self, other)

    def __and__(self, other):
        return and_(self, other)

    def __or__(self, other):
        return or_(self, other)

    def __xor__(self, other):
        return xor(self, other)

    def __eq__(self, other):
        return eq(self, other)

    def __ne__(self, other):
        return ne(self, other)

    def __lt__(self, other):
        return lt(self, other)

    def __le__(self, other):
        return le(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __ge__(self, other):
        return ge(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return sub(other, self)

    def __rmul__(self, other):
        return mul(other, self)

    def __rtruediv__(self, other):
        return div(other, self)

    def __rpow__(self, other):
        return pow(other, self)

    def __rmod__(self, other):
        return mod(other, self)

    def __rlshift__(self, other):
        return lshift(other, self)

    def __rrshift__(self, other):
        return rshift(other, self)

    def __rand__(self, other):
        return and_(other, self)

    def __ror__(self, other):
        return or_(other, self)

    def __rxor__(self, other):
        return xor(other, self)

    def __req__(self, other):
        return eq(other, self)

    def __rne__(self, other):
        return ne(other, self)

    def __rlt__(self, other):
        return lt(other, self)

    def __rle__(self, other):
        return le(other, self)

    def __rgt__(self, other):
        return gt(other, self)

    def __rge__(self, other):
        return ge(other, self)


zero = Dispatcher("zero", [Handler(lambda x: x * 0, name="multiply_0")])
one = Dispatcher("one", [Handler(lambda x: x**0, name="power_0")])

is_zero = Dispatcher("is_zero", [Handler(lambda x: x == zero(x), name="equal_zero")])
is_one = Dispatcher("is_one", [Handler(lambda x: x == one(x), name="equal_one")])


@add.add_handler(priority=1)
def add_any_zero(x, y):
    if is_zero(y):
        return x
    elif is_zero(x):
        return y

    return NotImplemented


@mul.add_handler(priority=1)
def mul_any_one(x, y):
    if is_one(y):
        return x
    elif is_one(x):
        return y

    return NotImplemented


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
