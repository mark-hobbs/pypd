"""
Application base class
----------------------

Notes
-----
* See PySPH solver.application for further info
* Make use of templating to write the common code once and dynamically
substitute the variable parts. PySPH makes use of the mako template library.

"""


class Application:
    """Subclass this to run any peridynamics simulation. There are several
    important methods that this class provides."""

    def __init__(self, coordinates, horizon, input_file=None, solver=None):
        self.coordinates = coordinates
        self.horizon = horizon
        self.input_file = input_file
        self.solver = solver
        # degress of freedom

        if self.input_file is not None:
            #  Read input file in a number of formats
            pass

        # Select a solver. If no solver is selected then the user must define
        # it using the correct method
        if self.solver is None:
            pass

    def configure():
        # Time integration scheme (Euler, Euler-Cromer etc)
        pass

    def build_particles():
        # Over-load this method
        # User can load a mesh file
        pass

    def build_particle_families():
        pass

    def build_boundary_conditions():
        pass

    def create_solver():
        """
        Create the solver, note that this is needed only if one has not
        used a scheme, otherwise, this will by default return the solver
        created by the scheme chosen - define single time step
        """
        pass

    def create_constitutive_model():
        """ "
        Function to determine the damage parameter (d), and equations to
        determine any parameters (linear elastic limit, critical stretch etc)
        """
        pass

    # -------------------------------------------------------------------------
    #                           Public methods
    # -------------------------------------------------------------------------

    @classmethod
    def read_from_txt_file(cls):
        """
        Create new class instance by reading from txt file

        Examples
        --------
        Application.read_from_txt_file('input_data.txt')

        """
        pass

    @classmethod
    def read_from_mat_file(cls):
        """
        Create new class instance by reading from mat file

        Examples
        --------
        Application.read_from_mat_file('input_data.mat')

        """
        pass

    def run():
        """Run the application.
        This method calls ``setup()`` and then ``solve()``.

        Parameters
        ----------
        argv: list
            Optional command line arguments.  Handy when running
            interactively.

        """
        self.setup()
        self.solve()

    # -------------------------------------------------------------------------
    #                  User methods that can be overloaded
    # -------------------------------------------------------------------------
