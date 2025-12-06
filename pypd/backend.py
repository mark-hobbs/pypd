from numba import cuda
from .kernels.particles import make_compute_nodal_forces, compute_nodal_forces_gpu


class Backend:
    """
    Backend class - handle device logic (GPU/CPU)
    """

    def __init__(self, model):
        self.model = model
        self.cuda_available = cuda.is_available()
        print(f"Is CUDA available: {self.cuda_available}")

        self.compute_particle_forces_cpu = make_compute_nodal_forces(
            self.model.bonds.constitutive_law.calculate_bond_damage
        )

    def host_to_device(self):
        if self.cuda_available:
            self.model.particles._host_to_device()
            self.model.bonds._host_to_device()

    def device_to_host(self):
        if self.cuda_available:
            self.model.particles._device_to_host()
            self.model.bonds._device_to_host()

    def compute_forces(self):
        """
        Compute particle forces

        Parameters
        ----------
        model : Model

        Returns
        -------
        particles.f: ndarray (float)
            Particle forces

        Notes
        -----
        * Particle forces are modified in place
        """
        if self.cuda_available:
            compute_nodal_forces_gpu(
                self.model.particles.d_f,
                self.model.particles.d_x,
                self.model.particles.d_u,
                self.model.particles.cell_volume,
                self.model.particles.d_nlist,
                self.model.bonds.d_d,
                self.model.bonds.d_c,
                self.model.bonds.d_surface_correction_factors,
            )
        else:
            self.compute_particle_forces_cpu(
                self.model.particles.f,
                self.model.particles.x,
                self.model.particles.u,
                self.model.particles.cell_volume,
                self.model.bonds.bondlist,
                self.model.bonds.d,
                self.model.bonds.c,
                self.model.bonds.f_x,
                self.model.bonds.f_y,
                self.model.bonds.surface_correction_factors,
            )
