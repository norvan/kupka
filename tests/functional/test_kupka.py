from kupka import KP, KPField, KPInput, KPNode


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


def test_it():
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


if __name__ == "__main__":

    e = Example(a=1, b=2)
    assert e.c() == 3, f"Found {e.c()}"
    assert e.d() == 5
    assert e.e() == 6
    assert e.f() == 12
