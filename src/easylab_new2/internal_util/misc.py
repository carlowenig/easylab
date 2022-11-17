import keyword


def format_args(*args, **kwargs):
    return ", ".join(
        [repr(arg) for arg in args]
        + [f"{key}={value!r}" for key, value in kwargs.items()]
    )


EllipsisType = type(Ellipsis)


def sanitize_arg_name(name: str):
    s = "".join(
        c if c.isalnum() else "_" for c in name.replace(" ", "_").replace("-", "_")
    )

    if s[0].isdigit():
        s = "_" + s

    if keyword.iskeyword(s):
        s += "_"

    if not s.isidentifier():
        raise ValueError(f"Invalid argument name {name!r}.")

    return s
