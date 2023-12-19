"""
Integrator base class
---------------------

Notes
-----
* See - pysph/sph/integrator.py
      - pysph/sph/integrator_step.py

* OpenCL solver - /pysph/sph/acceleration_eval_gpu_helper.py
                - /pysph/sph/tests/test_acceleration_eval.py
"""


from .kernels.integrator import euler_cromer


class Integrator:
    """
    Base class for time integrator
    """

    def __init__(self, dt=None):
        self.dt = dt or self._calculate_stable_dt()

    def _calculate_stable_dt():
        """
        Calculate stable time step
        """
        pass

    def _one_timestep():
        pass


class Euler(Integrator):
    pass


# class EulerCromer(Integrator):
class EulerCromer:
    # def __init__(self) -> None:
    #     super().__init__()

    def one_timestep(self, node_force, particles, simulation):
        """
        Update particle positions using an Euler-Cromer time integration scheme

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        * self.dt or simulation.dt?

        """
        return euler_cromer(
            node_force,
            particles.u,
            particles.v,
            particles.a,
            simulation.damping,
            particles.node_density,
            simulation.dt,
            particles.bc.flag,
            particles.bc.i_magnitude,
            particles.bc.unit_vector,
        )


class VelocityVerlet(Integrator):
    pass
