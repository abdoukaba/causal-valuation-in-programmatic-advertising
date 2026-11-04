"""Minimal stand-in for mpmath.iv backed by Arb (python-flint): every quantity is an Arb ball."""
from flint import arb, ctx
ctx.prec = 128
class _IV:
    dps = 35
    @staticmethod
    def mpf(x):
        if isinstance(x, (list, tuple)):
            lo, hi = arb(str(x[0])), arb(str(x[1])); return lo.union(hi)
        if isinstance(x, arb): return x
        return arb(str(x))
    @staticmethod
    def exp(a): return a.exp()
    @staticmethod
    def log(a): return a.log()
iv = _IV()
