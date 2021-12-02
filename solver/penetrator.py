
from numba import int16, float64
from numba.experimental import jitclass

spec = [
    ('ID', int16),
    ('centre', float64[:]),
    ('radius', float64),
    ('search_radius', float64),
    ('family', int16[:])
]

@jitclass(spec)
class Penetrator():

    def __init__(self, ID, centre, radius, search_radius, family):

        self.ID = ID
        self.centre = centre
        self.radius = radius
        self.search_radius = search_radius
        self.family = family
