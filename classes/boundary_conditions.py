"""
Boundary conditions class
-------------------------

"""

class BoundaryConditions():
    """
    The main class for defining the boundary conditions

    Attributes
    ----------
    flag : ndarray
        0 - no boundary condition
        1 - the node is subject to a boundary condition
    unit_vector : ndarray
    constraint : ndarray
    magnitude : float
        TODO: should this be a simulation parameter?

    Methods
    -------
    
    Notes
    -----
    * Should this class inherit from the ParticleSet class (i.e. child class)?
    * applied displacement / applied force / constraint

    """

    def __init__(self, flag, unit_vector):
        """
        BoundaryConditions class constructor

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        """
        self.flag = flag
        self.unit_vector = unit_vector