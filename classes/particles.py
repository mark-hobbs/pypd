"""
Particle array class
--------------------

TODO: rename classes as base or baseclasses?
"""

import numpy as np

from input.tools import build_particle_families
from solver.calculate import (calculate_nodal_forces,
                              calculate_node_damage,
                              smooth_step_data)


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
    
    damage : ndarray (float)
        The value of damage will range from 0 to 1, where 0 indicates that
        all bonds connected to the node are in the elastic range, and 1
        indicates that all bonds connected to the node have failed
    
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
    * Should dx be an attribute? Or is a Mesh class needed?
        - particles.dx
        - mesh.dx
    """

    def __init__(self, x, dx, bc, material, m=np.pi, nlist=None):
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
        
        bc : BoundaryConditions class
        
        material : Material class

        Returns
        -------

        Notes
        -----
        """

        self.x = x
        self.n_nodes = np.shape(self.x)[0]
        self.n_dim = np.shape(self.x)[1]
        self.horizon = m * dx  # TODO: is this an attribute of the particle set?
        self.bc = bc
        self.material = material
        self.cell_volume = dx**2  # TODO: 2D or 3D problem?
        self.node_density = self.material.density

        self.nlist = nlist
        if self.nlist is None:
            self.nlist, self.n_family_members = self._build_particle_families()

        # TODO: move the following to an initisalise method in Model or
        # Simulation?
        self.node_force = np.zeros((self.n_nodes, self.n_dim))
        self.u = np.zeros((self.n_nodes, self.n_dim))
        self.v = np.zeros((self.n_nodes, self.n_dim))
        self.a = np.zeros((self.n_nodes, self.n_dim))

        self.damage = np.zeros(self.n_nodes)

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
        # TODO: probably not needed?
        pass

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
        * Is it possible to pass bonds.material_model as a variable?
            - bonds.material_model.calculate_bond_damage()
        """
        return calculate_nodal_forces(bonds.bondlist, self.x, self.u,
                                      bonds.d,
                                      self.cell_volume,
                                      bonds.constitutive_law.c,
                                      bonds.constitutive_law.sc,
                                      bonds.f_x, bonds.f_y)

    def calculate_particle_damage(self, bonds):
        """
        Calculate particle damage

        Parameters
        ----------
        bonds : BondSet
            TODO: write a description

        Returns
        -------
        damage : ndarray (float)
            The value of damage will range from 0 to 1, where 0 indicates that
            all bonds connected to the node are in the elastic range, and 1
            indicates that all bonds connected to the node have failed

        Notes
        -----
        """
        self.damage = calculate_node_damage(self.x, bonds.bondlist, bonds.d,
                                            self.n_family_members)

    def update_particle_positions(self, node_force, simulation, integrator,
                                  i_time_step):
        """
        Update particle positions using an Euler-Cromer time integration scheme
        
        Parameters
        ----------
        simulation : Simulation class
            Defines simulation parameters

        integrator : Integrator class
            Euler / Euler-Cromer / Velocity-Verlet scheme

        Returns
        -------

        Notes
        -----
        * TODO: should the naming be consistent?
                update_particle_positions() / update_nodal_positions()
        * TODO: pass bc.magnitude as a function
        """

        self.bc.magnitude = smooth_step_data(i_time_step, 0,
                                             simulation.n_time_steps,
                                             0, 1e-4)

        return integrator.one_timestep(node_force, self, simulation)