# Lid-Driven Cavity Flow Solver using Physics-Informed Neural Networks (PINNs)

This project implements a solver for the lid-driven cavity flow problem using **Physics-Informed Neural Networks (PINNs)**. The approach uses a neural network to represent the solution and enforces the governing PDEs through automatic differentiation.

## Problem Description

The lid-driven cavity flow is a classic benchmark problem in computational fluid dynamics. It consists of:
- A square cavity (1×1 domain)
- Three stationary walls (bottom, left, right) with **free-slip** boundary conditions
- One moving wall (top lid) moving with constant velocity
- Incompressible, viscous flow governed by Navier-Stokes equations

### Governing Equations

For steady-state flow, we solve:

**Momentum equations:**
```
(u·∇)u = -∇p/ρ + ν∇²u
(u·∇)v = -∇p/ρ + ν∇²v
```

**Continuity equation:**
```
∇·u = 0
```

where:
- `u`, `v` are the x and y velocity components
- `p` is the pressure
- `ρ` is the fluid density
- `ν` is the kinematic viscosity

The Reynolds number is defined as: `Re = UL/ν`, where `U` is the lid velocity and `L` is the cavity length.

## Methodology: Physics-Informed Neural Networks (PINNs)

Instead of using traditional discretization methods (finite differences, finite elements), this framework:

1. **Neural Network Representation**: Uses a deep neural network to represent the solution:
   ```
   [u(x,y), v(x,y), p(x,y)] = NN(x, y; θ)
   ```
   where `θ` are the neural network parameters.

2. **Automatic Differentiation**: Computes all necessary derivatives (first and second order) using automatic differentiation, eliminating the need for finite difference approximations.

3. **Physics-Informed Loss**: Enforces the PDEs by minimizing the residuals at collocation points:
   - **PDE Loss**: Sum of squared residuals of momentum and continuity equations at interior collocation points
   - **Boundary Loss**: Sum of squared residuals enforcing boundary conditions at boundary points

4. **Optimization**: Uses gradient-based optimization (Adam optimizer) to minimize the total loss:
   ```
   L_total = λ_pde * L_PDE + λ_bc * L_BC
   ```

### Key Advantages of PINNs

- **Mesh-free**: No need for structured grids or meshes
- **Continuous solution**: The neural network provides a continuous representation of the solution
- **Automatic differentiation**: Exact derivatives without numerical errors
- **Flexible**: Easy to add physics constraints, data, or other terms to the loss

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
# venv\Scripts\activate   # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** On macOS with Homebrew Python, using a virtual environment is strongly recommended to avoid conflicts with system-managed packages.

## Usage

**Important:** Make sure to activate your virtual environment first:
```bash
source venv/bin/activate  # On macOS/Linux
```

### Basic Usage

Run the main script:
```bash
python main.py
```

This will:
- Solve the lid-driven cavity flow at Re=100 using PINNs
- Display convergence progress and timing information
- Generate visualization plots
- Export centerline velocity data to CSV files

### Custom Parameters

You can modify parameters in `main.py` or use the solver directly:

```python
from solver import LidDrivenCavitySolver

# Create solver with custom parameters
solver = LidDrivenCavitySolver(
    Re=100,                    # Reynolds number
    Lx=1.0,                    # Domain length in x
    Ly=1.0,                    # Domain length in y
    U_lid=1.0,                 # Lid velocity
    hidden_layers=[50, 50, 50, 50],  # Neural network architecture
    n_collocation=2000,        # Number of collocation points
    n_boundary=100             # Number of boundary points per edge
)

# Solve
solution = solver.solve(
    max_iter=10000,            # Maximum iterations
    lr=0.001,                  # Learning rate
    tol=1e-6,                  # Convergence tolerance
    verbose=True,              # Print progress
    lambda_pde=1.0,           # Weight for PDE loss
    lambda_bc=1.0             # Weight for boundary condition loss
)

# Access results
u = solution['u']              # x-velocity field (on visualization grid)
v = solution['v']              # y-velocity field
p = solution['p']              # pressure field
loss_history = solution['loss_history']
model = solution['model']      # Trained neural network model
```

### Visualization

The package includes visualization utilities:

```python
from visualize import plot_all_results

# Generate all plots
plot_all_results(solution, nx=200, ny=200, save_dir="results")
```

Available visualization functions:
- `plot_velocity_field()` - Velocity magnitude and streamlines
- `plot_pressure_field()` - Pressure contours
- `plot_centerline_velocity()` - Velocity profiles at centerlines
- `plot_convergence_history()` - Loss convergence curves
- `export_centerline_velocity_csv()` - Export centerline velocity data to CSV files

### Data Export

The solver automatically exports centerline velocity profiles to CSV files:
- `results/u_velocity_centerline.csv` - u-velocity along vertical centerline (x = Lx/2)
- `results/v_velocity_centerline.csv` - v-velocity along horizontal centerline (y = Ly/2)

These CSV files are useful for:
- Comparing with benchmark data (e.g., Ghia et al. 1982)
- Post-processing and analysis
- Validation studies

## Project Structure

```
Lid-Driven-Cavity-PINNs/
├── pinn_model.py            # Neural network architecture
├── pinn_loss.py             # PINN loss computation using automatic differentiation
├── boundary_conditions.py    # Boundary condition implementation (loss-based)
├── solver.py                # Main solver class
├── visualize.py             # Visualization utilities
├── main.py                  # Example usage script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Key Components

### 1. `PINN` (`pinn_model.py`)
Neural network model that takes (x, y) coordinates as input and outputs (u, v, p) velocity and pressure fields.

### 2. `PINNLossNS` (`pinn_loss.py`)
Computes PDE residuals using automatic differentiation:
- X-momentum equation residual
- Y-momentum equation residual
- Continuity equation residual

### 3. `LidDrivenCavityBC` (`boundary_conditions.py`)
Handles boundary conditions through loss terms:
- Top wall: `u = U_lid`, `v = 0` (moving lid)
- Bottom wall: `v = 0` (normal), `∂u/∂y = 0` (free-slip)
- Left wall: `u = 0` (normal), `∂v/∂x = 0` (free-slip)
- Right wall: `u = 0` (normal), `∂v/∂x = 0` (free-slip)

Free-slip means:
- Normal component of velocity = 0 (no flow through wall)
- Tangential component has zero gradient (no shear stress, fluid can slide along wall)

### 4. `LidDrivenCavitySolver` (`solver.py`)
Main solver that:
- Initializes the neural network model
- Samples collocation and boundary points
- Optimizes the total loss (PDE + BC) using Adam optimizer
- Monitors convergence
- Evaluates solution on a grid for visualization

## Results

The solver generates:
- Velocity fields (magnitude and streamlines)
- Pressure field contours
- Centerline velocity profiles (for benchmark comparison)
- Convergence history plots
- CSV files with centerline velocity data
- Timing information (solving time, visualization time, total runtime)

## Notes

- The neural network uses a fully connected architecture with Tanh activation
- Collocation points are randomly sampled in the interior domain
- Boundary points are uniformly sampled along each edge
- The solution is evaluated on a 200×200 grid for visualization
- Learning rate scheduling is used to improve convergence
- Loss weights (`lambda_pde`, `lambda_bc`) can be adjusted to balance PDE and boundary condition enforcement
- For stability, the learning rate may need adjustment for different Reynolds numbers
- The neural network architecture (number of layers, layer sizes) can be customized

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of computational physics*, 48(3), 387-411.

## License

This code is provided for educational and research purposes.
