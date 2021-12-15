"""
Constitutive law class
----------------------

Notes
-----

"""


class ConstitutiveLaw():
    """
    Subclass this to define a new constitutive law. This class ensures that
    all constitutive models follow the correct format.
    """

    def __init__():
        pass

    def required_parameters():
        pass

    def calculate_parameter_values():
        pass

    def calculate_damage_parameter():
        """
        Calculate bond damage (softening parameter). The value of d will range
        from 0 to 1, where 0 indicates that the bond is still in the elastic
        range, and 1 represents a bond that has failed
        """
        pass


class Linear(ConstitutiveLaw):
    pass


class Bilinear(ConstitutiveLaw):
    pass


class Trilinear(ConstitutiveLaw):
    pass


class NonLinear(ConstitutiveLaw):
    pass
