"""
Particle array class
--------------------
"""

# Particles, ParticleArray, or ParticleSet?
class Particles():
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
    """

    def __init__(self, coordinates):
        self.coordinates = coordinates
        # self.u = np.zeros([n_nodes, n_dim])
        # self.v = np.zeros([n_nodes, n_dim])
        # self.a = np.zeros([n_nodes, n_dim])

    # TODO: see PySPH
    def add_property():
        pass

    def add_constant():
        pass

    # TODO: this would require passing in a instance of the bonds class. Would
    # this lead to circular references?
    def calculate_particle_forces():
        """
        Calculate particle forces

        Parameters
        ----------

        Returns
        -------
        particle_forces: ndarray (float)

        Notes
        -----
        bonds.calculate_bond_stretch(particles)
        bonds.calculate_bond_damage(particles)
        bonds.calculate_bond_force(particles)
        """
        pass

    def update_particle_positions():
        """
        Euler / Euler-Cromer / Velocity-Verlet scheme
        """
        pass
