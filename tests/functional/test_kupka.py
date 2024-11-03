import pytest

from kupka import (
    KP,
    KPExecutor,
    KPField,
    KPInput,
    KPNode,
    MultiprocessingKPExecutor,
    SequentialProcessKPExecutor,
    kp_settings,
)


def add(x: float, y: float) -> float:
    return x + y


def mult(x: float, y: float) -> float:
    return x * y


class FirstExample(KP):
    a: KPNode[float] = KPInput()
    b: KPNode[float] = KPInput()
    c: KPNode[float] = KPField(add, x=a, y=b)
    d: KPNode[float] = KPField(add, x=c, y=b)
    e: KPNode[float] = KPField(add, x=d, y=a)
    f: KPNode[float] = KPField(mult, x=e, y=e)


class SecondExample(KP):
    a: KPNode[float] = KPInput()
    b: KPNode[float] = KPInput()
    c: KPNode[float] = KPField(mult, x=a, y=b)
    d: KPNode[float] = KPField(mult, x=c, y=b)
    e: KPNode[float] = KPField(mult, x=d, y=a)
    f: KPNode[float] = KPField(add, x=e, y=e)


@pytest.mark.parametrize(
    "executor",
    [
        KPExecutor(),
        SequentialProcessKPExecutor(),
        MultiprocessingKPExecutor(),
    ],
)
def test_no_gap(executor):
    with kp_settings.use(executor=executor):

        e1 = FirstExample(a=1, b=2)
        e2 = FirstExample(a=3, b=2)
        e3 = SecondExample(a=2, b=4)

        assert e1.c() == 3
        assert e2.c() == 5
        assert e3.c() == 8

        assert e1.d() == 5
        assert e2.d() == 7
        assert e3.d() == 32

        assert e1.e() == 6
        assert e2.e() == 10
        assert e3.e() == 64

        assert e1.f() == 36
        assert e2.f() == 100
        assert e3.f() == 128


def test_end_first():
    assert isinstance(kp_settings.executor(), KPExecutor)
    try:
        kp_settings.set_global_executor(SequentialProcessKPExecutor())
        assert isinstance(kp_settings.executor(), SequentialProcessKPExecutor)

        print("*" * 30, "INIT", "*" * 30)

        e1 = FirstExample(a=1, b=2)
        e2 = FirstExample(a=3, b=2)
        e3 = SecondExample(a=2, b=4)

        print("*" * 30, "START END: computation", "*" * 30)

        assert e1.f() == 36
        assert e2.f() == 100
        assert e3.f() == 128

        print("*" * 30, "START: precomputed", "*" * 30)

        assert e1.e() == 6
        assert e2.e() == 10
        assert e3.e() == 64

        assert e1.d() == 5
        assert e2.d() == 7
        assert e3.d() == 32

        assert e1.c() == 3
        assert e2.c() == 5
        assert e3.c() == 8
    finally:
        kp_settings.set_global_executor(KPExecutor())
    assert isinstance(kp_settings.executor(), KPExecutor)


if __name__ == "__main__":

    e = Example(a=1, b=2)
    assert e.c() == 3, f"Found {e.c()}"
    assert e.d() == 5
    assert e.e() == 6
    assert e.f() == 12
