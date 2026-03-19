from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable

from numba import cuda
from .kernels.particles import (
    make_compute_nodal_forces_cpu,
    make_compute_nodal_forces_gpu,
)

if TYPE_CHECKING:
    from .model import Model


class Backend:
    """
    Backend class - handle device logic (GPU/CPU)
    """

    def __init__(self, model: Model) -> None:
        self.model: Model = model
        self.cuda_available: bool = cuda.is_available()
        self.compute_particle_forces_cpu: Callable[..., None] | None = None
        self.compute_particle_forces_gpu: Callable[..., None] | None = None

        print(f"Is CUDA available: {self.cuda_available}")

        if self.cuda_available:
            self._build_force_function_gpu()
        else:
            self._build_force_function_cpu()

        if self.model.penetrators:
            if self.cuda_available:
                pass
            else:
                # self.model.penetrators.compile_cpu()
                pass

    def _build_force_function_cpu(self) -> None:
        self.model.bonds.constitutive_law.compile_cpu()
        self.compute_particle_forces_cpu = make_compute_nodal_forces_cpu(
            self.model.bonds.constitutive_law.calculate_bond_damage
        )

    def _build_force_function_gpu(self) -> None:
        """
        material_law = make_material_law(sc)
        compute_nodal_forces_kernel = make_compute_nodal_forces_kernel(material_law)
        """
        self.model.bonds.constitutive_law.compile_gpu()
        self.compute_particle_forces_gpu = make_compute_nodal_forces_gpu(
            self.model.bonds.constitutive_law.calculate_bond_damage
        )

    def host_to_device(self) -> None:
        if self.cuda_available:
            self.model.particles._host_to_device()
            self.model.bonds._host_to_device()

    def device_to_host(self) -> None:
        if self.cuda_available:
            self.model.particles._device_to_host()
            self.model.bonds._device_to_host()

    def compute_forces(self) -> None:
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
            assert self.compute_particle_forces_gpu is not None
            self.compute_particle_forces_gpu(
                self.model.particles.d_f,
                self.model.particles.d_x,
                self.model.particles.d_u,
                self.model.particles.cell_volume,
                self.model.particles.d_nlist,
                self.model.bonds.d_d,
                self.model.bonds.d_c,
                self.model.bonds.d_surface_correction_factors,
                self.model.bonds.d_s0,
                self.model.bonds.d_s1,
                self.model.bonds.d_sc,
            )
        else:
            assert self.compute_particle_forces_cpu is not None
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
