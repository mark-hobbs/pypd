# `pypd` architecture

This document outlines the current thinking on the architecture and design of `pypd`. The existing API is provisional and may change in future iterations as the architecture matures.

### Minimal example

```python
x = build_particle_coordinates(dx, n_div_x, n_div_y)
flag, unit_vector = build_boundary_conditions(x, dx)

material = pypd.Material(name="homalite", E=4.55e9, Gf=38.46, density=1230, ft=2.5)
bc = pypd.BoundaryConditions(flag, unit_vector, magnitude=1e-4)
particles = pypd.Particles(x, dx, bc, material)
bonds = pypd.Bonds(particles, influence=pypd.Constant, notch=notch)
model = pypd.Model(particles, bonds)

animation = pypd.Animation(
    frequency=100, sz=0.25, show_title=False, data="strain energy density"
)
simulation = pypd.Simulation(n_time_steps=5000, damping=0, animation=animation)
simulation.run(model)
model.save_state_fig(fig_title="crack-branching")
```

### Design notes

-  `Bonds` are always derivable from `Particles`. Let `Model` manage `Bonds` internally.
- Both `Particles` and `Bonds` should primarily be data containers (with light validation), leaving numerical methods and orchestration to `Model` and `Simulation`.
- Are shallow or deep classes preferable?