from typing import TypeGuard


class Undefined:
    __instance = None

    def __new__(cls):
        if Undefined.__instance is None:
            Undefined.__instance = object.__new__(cls)
        return Undefined.__instance

    def __repr__(self):
        return "undefined"

    def __str__(self):
        return "undefined"


undefined = Undefined()


# def is_undefined(value) -> TypeGuard[Undefined]:
#     return value is undefined
