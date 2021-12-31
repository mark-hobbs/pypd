
import itertools
from numba import int16, float64
from numba.experimental import jitclass

spec = [
    ('ID', int16),
    ('centre', float64[:]),
    ('radius', float64),
    ('search_radius', float64),
    ('family', int16[:])
]


# TODO: should Penetrator be a base class? Create a subclass for supports

@jitclass(spec)
class Penetrator():

    # Numba doesn't support class members (class parameters). Therefore it is
    # not currently possible to assign a unique ID to every class instance by
    # incrementing a counter

    # Use a structured numpy array to avoid issues with passing an instance of
    # a class to a jit compiled function

    def __init__(self, ID, centre, radius, search_radius, family):

        self.ID = ID
        self.centre = centre
        self.radius = radius
        self.search_radius = search_radius
        self.family = family

    def build_family():
        pass

    def calculate_contact_force():
        """
        Calculate the contact force between a rigid penetrator and deformable
        peridynamic body
        """
        pass

    def update_penetrator_position():
        """
        Update the penetrator position
        """
        pass
