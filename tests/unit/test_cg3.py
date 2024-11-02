from kupka.cg3 import CGInput, CGField, CG


def test_it():
    def add(x: float, y: float) -> float:
        return x + y

    class Example(CG):
        a: CGInput[float] = CGInput()
        b: CGInput[float] = CGInput()
        c: CGField[float] = CGField(add, x=a, y=b)
        d: CGField[float] = CGField(add, x=c, y=b)
    
    e = Example(a=1, b=2)
    assert e.c() == 3, f"Found {e.c()}"
    assert e.d() == 5
