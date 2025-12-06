from .kernels.integrator import euler_cromer_cpu, euler_cromer_gpu


class Integrator:
    """
    Base class for time integrator

    Notes
    -----
    * See - pysph/sph/integrator.py
        - pysph/sph/integrator_step.py

    * OpenCL solver - /pysph/sph/acceleration_eval_gpu_helper.py
                    - /pysph/sph/tests/test_acceleration_eval.py
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


class EulerCromer:

    def __call__(self, simulation, particles):
        """
        Update particle positions using an Euler-Cromer time integration scheme

        Parameters
        ----------
        particles : Particles

        simulation : Simulation

        Returns
        -------

        Notes
        -----
        * particles.u and particles.v are modified in place
        """
        if simulation.backend.cuda_available:
            euler_cromer_gpu(
                particles.d_f,
                particles.d_u,
                particles.d_v,
                particles.d_a,
                particles.material.density,
                particles.d_bc_flag,
                particles.bc.i_magnitude,
                particles.d_bc_unit_vector,
                simulation.damping,
                simulation.dt,
            )
        else:
            euler_cromer_cpu(
                particles.f,
                particles.u,
                particles.v,
                particles.a,
                particles.material.density,
                particles.bc.flag,
                particles.bc.i_magnitude,
                particles.bc.unit_vector,
                simulation.damping,
                simulation.dt,
            )


class VelocityVerlet(Integrator):
    pass
