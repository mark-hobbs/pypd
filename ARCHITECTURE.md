# `pypd` architecture

This document outlines the current thinking on the architecture and design of `pypd`. The existing API is provisional and may change in future iterations as the architecture matures.

## Minimal example

```python
x = build_particle_coordinates(dx, n_div_x, n_div_y)
flag, unit_vector = build_boundary_conditions(x, dx)

material = pypd.Material(name="homalite", E=4.55e9, Gf=38.46, density=1230, ft=2.5)
bc = pypd.BoundaryConditions(flag, unit_vector, magnitude=1e-4)
particles = pypd.Particles(x, dx, bc, material)
bonds = pypd.Bonds(particles, influence=pypd.Constant, notch=notch)
model = pypd.Model(particles, bonds)

simulation = pypd.Simulation(n_time_steps=5000, damping=0)
simulation.run(model)
```

## Open design questions

### 1. `model.run(simulation)` vs `simulation.run(model)`

**Data vs execution split:** using `simulation.run(model)` provides clear separation of concerns between the `Model` which describes the physical system and `Simulation` which manages the execution.

This also makes it natural to reuse the same simulation parameters across multiple models:

```python
for model in models:
    simulation.run(model)
```

### 2. `particles.compute_forces(bonds)` vs `model.compute_particle_forces()`

There is a strong argument for keeping methods in data classes as each class should be responsible for operations that are intimately tied to its data. However, a problem with the existing design - `particles.compute_forces(bonds)` - is that `Particles` need to know about `Bonds` which breaks encapsulation.

### 3. Let `Model` manage `Bonds` internally

```python
x = build_particle_coordinates(dx, n_div_x, n_div_y)
flag, unit_vector = build_boundary_conditions(x, dx)

material = pypd.Material(name="homalite", E=4.55e9, Gf=38.46, density=1230, ft=2.5)
bc = pypd.BoundaryConditions(flag, unit_vector, magnitude=1e-4)
particles = pypd.Particles(x, dx, bc, material)
model = pypd.Model(particles, influence=pypd.Constant, notch=notch)

simulation = pypd.Simulation(n_time_steps=5000, damping=0)
simulation.run(model)
```

### 4. Switching material laws

Material models (e.g. linear, bilinear, trilinear, non-linear) have different parameter requirements. If these parameter differences are exposed directly, every place where `material_law()` is used must be updated whenever the model changes. This leads to rigid and error-prone code.

The call interface of the material law must remain consistent for all constitutive models. A function factory solves this problem by decoupling the model configuration from its usage. The factory embeds the model parameters when the function is created, and the returned function always has the same interface: `material_law(i, stretch, d)`

```python
def make_material_law(s0, s1, sc, beta):
    
    @njit
    def material_law(i, stretch, d):
        return trilinear(i, stretch, d, s0, s1, sc, beta)

    return material_law
```

### 5. Material law integration strategy

Is it better to pass the material law as a variable to `compute_nodal_forces()` (i.e. inject it at runtime) or bake in at compile time?

```python
# baked in
def make_compute_nodal_forces(material_law):
    """
    Returns a specialised JIT-compiled compute_nodal_forces() function with the given material law baked in.
    """
    @njit(parallel=True, fastmath=True)
    def compute_nodal_forces_cpu(
        node_force,
        x,
        u,
        cell_volume,
        bondlist,
        d,
        c,
        f_x,
        f_y,
        surface_correction_factors,
    ):
        n_bonds = bondlist.shape[0]
        node_force[:] = 0.0

        for k_bond in prange(n_bonds):
            node_i = bondlist[k_bond, 0]
            node_j = bondlist[k_bond, 1]

            xi_x = x[node_j, 0] - x[node_i, 0]
            xi_y = x[node_j, 1] - x[node_i, 1]

            xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
            xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])

            xi = np.sqrt(xi_x**2 + xi_y**2)
            y = np.sqrt(xi_eta_x**2 + xi_eta_y**2)
            stretch = (y - xi) / xi

            d[k_bond] = material_law(k_bond, stretch, d[k_bond])

            f = (
                stretch
                * c[k_bond]
                * (1 - d[k_bond])
                * cell_volume
                * surface_correction_factors[k_bond]
            )
            f_x[k_bond] = f * xi_eta_x / y
            f_y[k_bond] = f * xi_eta_y / y

        # Reduce bond forces to nodal forces
        for k_bond in range(n_bonds):
            node_i = bondlist[k_bond, 0]
            node_j = bondlist[k_bond, 1]

            node_force[node_i, 0] += f_x[k_bond]
            node_force[node_j, 0] -= f_x[k_bond]
            node_force[node_i, 1] += f_y[k_bond]
            node_force[node_j, 1] -= f_y[k_bond]

        return node_force, d

    return compute_nodal_forces_cpu
```

```python
# Compile time
@njit(parallel=True, fastmath=True)
def compute_nodal_forces_cpu(
    node_force,
    x,
    u,
    cell_volume,
    bondlist,
    d,
    c,
    f_x,
    f_y,
    material_law,
    surface_correction_factors,
):
    """
    Compute particle forces - employs bondlist (cpu optimised)

    Parameters
    ----------
    node_force : np.ndarray(float, shape=(n_nodes, n_dim))
        Nodal force array

    x : np.ndarray(float, shape=(n_nodes, n_dim))
        Material point coordinates in the reference configuration

    u : np.ndarray(float, shape=(n_nodes, n_dim))
        Nodal displacement

    cell_volume : float

    bondlist : np.ndarray(int, shape=(n_bonds, 2))
        Array of pairwise interactions (bond list)

    d : np.ndarray(float, shape=(n_bonds,))
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed

    c : np.ndarray(float, shape=(n_bonds,))
        Bond stiffness

    material_law : function

    surface_correction_factors : np.ndarray(float, shape=(n_bonds,))
    
    Returns
    -------
    node_force : np.ndarray(float, shape=(n_nodes, n_dimensions))
        Nodal force array

    d : np.ndarray(float, shape=(n_bonds,))
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed

    Notes
    -----
    * node_force and d are modified in place and returned for clarity
    """
    n_bonds = np.shape(bondlist)[0]
    node_force[:] = 0.0

    for k_bond in prange(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        xi_x = x[node_j, 0] - x[node_i, 0]
        xi_y = x[node_j, 1] - x[node_i, 1]

        xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
        xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])

        xi = np.sqrt(xi_x**2 + xi_y**2)
        y = np.sqrt(xi_eta_x**2 + xi_eta_y**2)
        stretch = (y - xi) / xi

        d[k_bond] = material_law(k_bond, stretch, d[k_bond])

        f = (
            stretch
            * c[k_bond]
            * (1 - d[k_bond])
            * cell_volume
            * surface_correction_factors[k_bond]
        )
        f_x[k_bond] = f * xi_eta_x / y
        f_y[k_bond] = f * xi_eta_y / y

    # Reduce bond forces to particle forces
    for k_bond in range(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        node_force[node_i, 0] += f_x[k_bond]
        node_force[node_j, 0] -= f_x[k_bond]
        node_force[node_i, 1] += f_y[k_bond]
        node_force[node_j, 1] -= f_y[k_bond]

    return node_force, d
```

### 6. Backend logic (CPU/GPU)

## Design notes

-  `Bonds` are always derivable from `Particles`. 
- Both `Particles` and `Bonds` should primarily be data containers (with light validation), leaving numerical methods and orchestration to `Model` and `Simulation`.
- Are shallow or deep classes preferable?
- Avoid "parameter drilling" where a parameter is passed down through multiple layers of function/method calls, even when the intermediate layers do not use the parameter - they just pass it along.