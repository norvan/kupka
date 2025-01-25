from typing import cast, Generic, TypeVar

T = TypeVar("T")


class WrappedValue(Generic[T]):
    _value: T | None
    _initialized: bool

    @classmethod
    def from_value(cls, value: T) -> "WrappedValue":
        return cls(value, True)

    @classmethod
    def init(cls) -> "WrappedValue":
        return cls(None, False)

    def __init__(self, value: T | None, initialized: bool):
        self._value = value
        self._initialized = initialized

    def set_value(self, value: T) -> None:
        # NOTE: I am not a fan of the 2-step initialization, but for now it'll do.
        if self._initialized:
            raise AttributeError(f"Value has already been set!")
        self._value = value
        self._initialized = True

    def __call__(self) -> T:
        if not self._initialized:
            raise ValueError(f"Value has not been set!")
        return cast(T, self._value)
