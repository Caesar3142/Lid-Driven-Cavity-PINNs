"""
Main script for solving lid-driven cavity flow using Physics-Informed Neural Networks (PINNs)
"""

import numpy as np
import time
from solver import LidDrivenCavitySolver
from visualize import plot_all_results, export_centerline_velocity_csv


def main():
    """Main function to solve and visualize lid-driven cavity flow"""
    
    # Record total start time
    total_start_time = time.time()
    
    # Problem parameters
    Re = 100  # Reynolds number
    n_collocation = 2000  # Number of collocation points for PDE loss
    n_boundary = 100  # Number of boundary points per edge
    
    print("=" * 60)
    print("Lid-Driven Cavity Flow Solver")
    print("Using Physics-Informed Neural Networks (PINNs)")
    print("=" * 60)
    print(f"Reynolds number: Re = {Re}")
    print(f"Collocation points: {n_collocation}")
    print(f"Boundary points per edge: {n_boundary}")
    print("-" * 60)
    
    # Create solver
    solver = LidDrivenCavitySolver(
        Re=Re,
        Lx=1.0,
        Ly=1.0,
        U_lid=1.0,
        hidden_layers=[50, 50, 50, 50],
        n_collocation=n_collocation,
        n_boundary=n_boundary
    )
    
    # Solve
    print("\nSolving...")
    solve_start_time = time.time()
    solution = solver.solve(
        max_iter=10000,
        lr=0.001,
        tol=1e-6,
        verbose=True,
        log_interval=500,
        lambda_pde=1.0,
        lambda_bc=1.0
    )
    solve_elapsed_time = time.time() - solve_start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("Solution Summary")
    print("=" * 60)
    print(f"Iterations: {solution['iterations']}")
    print(f"Final loss: {solution['loss_history'][-1]:.2e}")
    if solution['loss_components_history']:
        last_components = solution['loss_components_history'][-1]
        print(f"PDE Momentum X residual: {last_components['pde_momentum_x']:.2e}")
        print(f"PDE Momentum Y residual: {last_components['pde_momentum_y']:.2e}")
        print(f"PDE Continuity residual: {last_components['pde_continuity']:.2e}")
        print(f"Boundary condition loss: {last_components['bc_loss']:.2e}")
    print(f"Solving time: {solve_elapsed_time:.2f} seconds ({solve_elapsed_time/60:.2f} minutes)")
    
    # Compute some flow statistics
    u = solution['u']
    v = solution['v']
    speed = np.sqrt(u**2 + v**2)
    
    print(f"\nFlow Statistics:")
    print(f"Maximum velocity magnitude: {np.max(speed):.4f}")
    print(f"Average velocity magnitude: {np.mean(speed):.4f}")
    print(f"Maximum u-velocity: {np.max(u):.4f}")
    print(f"Maximum v-velocity: {np.max(v):.4f}")
    
    # Visualize results
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # Grid resolution for visualization (solution is already on 200x200 grid)
    nx, ny = 200, 200
    
    viz_start_time = time.time()
    plot_all_results(solution, nx, ny, Lx=1.0, Ly=1.0, save_dir="results")
    viz_elapsed_time = time.time() - viz_start_time
    
    # Calculate total elapsed time
    total_elapsed_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("Timing Summary")
    print("=" * 60)
    print(f"Solving time: {solve_elapsed_time:.2f} seconds ({solve_elapsed_time/60:.2f} minutes)")
    print(f"Visualization time: {viz_elapsed_time:.2f} seconds ({viz_elapsed_time/60:.2f} minutes)")
    print(f"Total running time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    print("=" * 60)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
