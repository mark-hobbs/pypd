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

### `model.run(simulation)` vs `simulation.run(model)`

**Data vs execution split:** using `simulation.run(model)` provides clear separation of concerns between the `Model` which describes the physical system and `Simulation` which manages the execution.

This also makes it natural to reuse the same simulation parameters across multiple models:

```python
for model in models:
    simulation.run(model)
```

### `particles.compute_forces(bonds)` vs `model.compute_particle_forces()`

There is a strong argument for keeping methods in data classes as each class should be responsible for operations that are intimately tied to its data. However, a problem with the existing design - `particles.compute_forces(bonds)` - is that `Particles` need to know about `Bonds` which breaks encapsulation.

### Let `Model` manage `Bonds` internally

### Backend logic (CPU/GPU)

## Design notes

-  `Bonds` are always derivable from `Particles`. 
- Both `Particles` and `Bonds` should primarily be data containers (with light validation), leaving numerical methods and orchestration to `Model` and `Simulation`.
- Are shallow or deep classes preferable?