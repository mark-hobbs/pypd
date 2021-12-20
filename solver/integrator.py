"""
Integrator base class
---------------------

Notes
-----
* See - pysph/sph/integrator.py
      - pysph/sph/integrator_step.py
"""


class Integrator():
    """
    Base class for time integrator
    """

    def calculate_stable_dt():
        """
        Calculate stable time step
        """
        pass

    def one_timestep():
        pass


class Euler(Integrator):
    pass


class EulerCromer(Integrator):
    
    def one_timestep():
        """
        particles.calculate_particle_forces(bonds)
        particles.update_particle_positions()
        """
        pass


class VelocityVerlet(Integrator):
    pass
