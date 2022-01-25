"""
Particle array class
--------------------

TODO: rename classes as base or baseclasses?
"""

import numpy as np

from input.tools import build_particle_families
from solver.calculate import calculate_nodal_forces_bondlist, update_nodal_positions


# Particles, ParticleArray, or ParticleSet?
class ParticleSet():
    """
    The main class for storing the particle (node) set.

    Attributes
    ----------
    mesh_file : str
        Name of the mesh file defining the system of particles
        (TODO: attribute or parameter for __init__?)
    n_nodes : int
        Number of particles
    n_dim : int
        Number of dimensions (2 or 3-dimensional system)
    nlist : ndarray
        TODO: write description
    n_family_members: ndarray (int)
        Number of family members
    x : ndarray (float)
        Material point coordinates in the reference configuration
    u : ndarray (float)
        Displacement array
    v : ndarray (float)
        Velocity array
    a : ndarray (float)
        Acceleration array
    material_flag : ndarray (int)
        Flag to identify the material type. The flag is set by the user.
    cell_volume: ndarray (float)
        Cell area / volume. If a regular mesh is employed, this value will be
        a constant for all nodes.
    boundary_condition_flag : ndarray (int)
        Flag to... 1 if a boundary condition is applied, 0 if no...
    boundary_condition_value : ndarray (float)
    

    Methods
    -------
    
    Notes
    -----
    * Class should accept both regular and irregular meshes

    """

    def __init__(self, x, dx, m=np.pi, nlist=None):
        """
        ParticleSet class constructor

        Parameters
        ----------
        x : ndarray (float)
            Material point coordinates in the reference configuration
        dx : float
            Mesh resolution (only valid for regular meshes)
        m : float
            Ratio between the horizon radius and grid resolution (default
            value is pi)
        material : Material class
        boundary_conditions : BoundaryConditions class

        Returns
        -------

        Notes
        -----
        """

        self.x = x
        self.n_nodes = np.shape(self.x)[0]
        self.horizon = m * dx
        self.nlist = nlist
        # self.u = np.zeros([n_nodes, n_dim])
        # self.v = np.zeros([n_nodes, n_dim])
        # self.a = np.zeros([n_nodes, n_dim])

        if self.nlist is None:
            self.nlist = self._build_particle_families()

    # TODO: see PySPH
    def add_property():
        pass

    def add_constant():
        pass

    def _build_particle_families(self):
        """
        Build particle families

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        """
        return build_particle_families(self.x, self.horizon)

    def _build_boundary_conditions():
        pass

    # TODO: this would require passing in a instance of the bonds class. Would
    # this lead to circular references?
    def calculate_particle_forces(self, bonds):
        """
        Calculate particle forces

        Parameters
        ----------
        bonds : BondSet
            TODO: write a description
        constitutive_law

        Returns
        -------
        particle_forces: ndarray (float)

        Notes
        -----
        bonds.calculate_bond_stretch(particles)
        bonds.calculate_bond_damage(particles)
        bonds.calculate_bond_force(particles)

        * TODO: should bonds.c and bonds.beta be attributes of a constitutive
        model class?
        * TODO: give users the option to use bondlist or neighbourlist
        """
        return calculate_nodal_forces_bondlist(bonds.bondlist, self.x, self.u,
                                               bonds.d, bonds.c, bonds.beta,
                                               bonds.f_x, bonds.f_y, bonds.f_z,
                                               self.node_force)

    def update_particle_positions(self, simulation):
        """
        Update particle positions using an Euler-Cromer time integration scheme
        
        Parameters
        ----------
        simulation : Simulation class
            Defines simulation parameters

        Returns
        -------

        Notes
        -----
        * Euler / Euler-Cromer / Velocity-Verlet scheme
        * TODO: should the naming be consistent?
                update_particle_positions() / update_nodal_positions()
        """
        return update_nodal_positions(self.node_force, self.u, self.ud,
                                      self.ud, self.udd, simulation.damping,
                                      self.node_density, simulation.dt)
