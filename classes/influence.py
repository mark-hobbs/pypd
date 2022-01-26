
"""
Influence function base class
-----------------------------

Notes
-----
* see pysph / solver / solver.py
    - kernel : pysph.base.kernels.Kernel
"""


class InfluenceFunction():
    pass


class constant(InfluenceFunction):
    pass


class triangular(InfluenceFunction):
    pass


class quartic(InfluenceFunction):
    pass
