"""
Application base class
----------------------

Notes
-----
* See PySPH solver.application for further info
* Make use of templating to write the common code once and dynamically
substitute the variable parts. PySPH makes use of the mako template library.

"""


class Application():
    """Subclass this to run any peridynamics simulation. There are several
    important methods that this class provides."""

    def __init__(self, coordinates, horizon, solver=None):
        self.coordinates = coordinates
        self.horizon = horizon
        self.solver = solver

        # Select a solver. If no solver is selected then the user must define 
        # it using the correct method
        if self.solver is None:
            pass

    def configure():
        # Time integration (Euler, Euler-Cromer etc)
        pass

    def build_particles():
        # Over-load this method
        pass

    def build_particle_families():
        pass

    def build_boundary_conditions():
        pass

    def create_solver():
        """Create the solver, note that this is needed only if one has not
        used a scheme, otherwise, this will by default return the solver
        created by the scheme chosen - define single time step"""
        pass

    def create_constitutive_model():
        pass

    # -------------------------------------------------------------------------
    #                           Public methods
    # -------------------------------------------------------------------------

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
