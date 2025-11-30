"""
Main Solver for Lid-Driven Cavity Flow using Physics-Informed Neural Networks (PINNs)
"""

import numpy as np
import torch
import torch.optim as optim
from pinn_model import PINN
from pinn_loss import PINNLossNS
from boundary_conditions import LidDrivenCavityBC


class LidDrivenCavitySolver:
    """
    Solver for lid-driven cavity flow using Physics-Informed Neural Networks (PINNs)
    """
    
    def __init__(self, Lx=1.0, Ly=1.0, Re=100, rho=1.0, U_lid=1.0, 
                 hidden_layers=[50, 50, 50, 50], n_collocation=2000, n_boundary=100):
        """
        Initialize solver
        
        Args:
            Lx: Domain length in x-direction
            Ly: Domain length in y-direction
            Re: Reynolds number
            rho: Fluid density
            U_lid: Lid velocity
            hidden_layers: List of hidden layer sizes
            n_collocation: Number of collocation points for PDE loss
            n_boundary: Number of boundary points per edge
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        self.U_lid = U_lid
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        
        # Initialize PINN model
        layers = [2] + hidden_layers + [3]  # [input_dim, hidden..., output_dim]
        self.model = PINN(layers=layers)
        
        # Initialize PINN loss framework
        self.pinn_loss = PINNLossNS(Re=Re, rho=rho, Lx=Lx, Ly=Ly)
        
        # Initialize boundary conditions
        self.bc = LidDrivenCavityBC(Lx=Lx, Ly=Ly, U_lid=U_lid)
        
        # Sample collocation and boundary points
        self._sample_points()
    
    def _sample_points(self):
        """Sample collocation points for PDE loss and boundary points for BC loss"""
        # Collocation points (interior domain)
        # Create as leaf nodes with requires_grad=True for derivative computation
        # but they won't accumulate gradients (they're inputs, not parameters)
        x_colloc = (torch.rand(self.n_collocation) * self.Lx).requires_grad_(True)
        y_colloc = (torch.rand(self.n_collocation) * self.Ly).requires_grad_(True)
        
        self.collocation_points = {
            'x': x_colloc,
            'y': y_colloc
        }
        
        # Boundary points
        self.boundary_points = self.bc.sample_boundary_points(n_boundary=self.n_boundary)
    
    def resample_points(self):
        """Resample collocation and boundary points (useful for adaptive sampling)"""
        self._sample_points()
    
    def solve(self, max_iter=10000, lr=0.001, tol=1e-6, verbose=True, log_interval=100,
              lambda_pde=1.0, lambda_bc=1.0):
        """
        Solve the lid-driven cavity flow problem
        
        Args:
            max_iter: Maximum number of iterations
            lr: Learning rate
            tol: Convergence tolerance
            verbose: Print progress
            log_interval: Print loss every N iterations
            lambda_pde: Weight for PDE loss
            lambda_bc: Weight for boundary condition loss
            
        Returns:
            Dictionary with solution fields and loss history
        """
        # Set up optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler (optional, helps with convergence)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000
        )
        
        # Loss history
        loss_history = []
        loss_components_history = []
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Evaluate model at collocation points
            u_colloc, v_colloc, p_colloc = self.model(
                self.collocation_points['x'],
                self.collocation_points['y']
            )
            
            # Compute PDE loss
            pde_loss, pde_components = self.pinn_loss.compute_pde_loss(
                u_colloc, v_colloc, p_colloc,
                self.collocation_points['x'],
                self.collocation_points['y']
            )
            
            # Compute boundary condition loss
            bc_loss, bc_components = self.bc.compute_boundary_loss(
                self.model, self.boundary_points
            )
            
            # Total loss
            total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Extract values for logging (after backward pass)
            with torch.no_grad():
                loss_val = total_loss.item()
                pde_loss_val = pde_loss.item()
                bc_loss_val = bc_loss.item()
            
            # Update learning rate
            scheduler.step(loss_val)
            
            # Record loss
            loss_history.append(loss_val)
            loss_components_history.append({
                'pde_momentum_x': pde_components['momentum_x'],
                'pde_momentum_y': pde_components['momentum_y'],
                'pde_continuity': pde_components['continuity'],
                'bc_loss': bc_loss_val,
                **bc_components
            })
            
            # Check convergence
            if total_loss.item() < tol:
                if verbose:
                    print(f"Converged at iteration {iteration+1} with loss = {total_loss.item():.2e}")
                break
            
            # Print progress
            if verbose and (iteration + 1) % log_interval == 0:
                print(f"Iteration {iteration+1}/{max_iter}: Total Loss = {total_loss.item():.2e}, "
                      f"PDE Loss = {pde_loss.item():.2e}, BC Loss = {bc_loss.item():.2e}")
                print(f"  PDE - Momentum X: {pde_components['momentum_x']:.2e}, "
                      f"Momentum Y: {pde_components['momentum_y']:.2e}, "
                      f"Continuity: {pde_components['continuity']:.2e}")
        
        # Evaluate solution on a grid for visualization
        nx, ny = 200, 200  # Grid resolution for visualization
        x_grid = np.linspace(0, self.Lx, nx)
        y_grid = np.linspace(0, self.Ly, ny)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Flatten grid for model evaluation
        x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32)
        y_flat = torch.tensor(Y_grid.flatten(), dtype=torch.float32)
        
        # Evaluate model
        self.model.eval()
        with torch.no_grad():
            u_flat, v_flat, p_flat = self.model(x_flat, y_flat)
            u_grid = u_flat.numpy().reshape(ny, nx)
            v_grid = v_flat.numpy().reshape(ny, nx)
            p_grid = p_flat.numpy().reshape(ny, nx)
        
        self.model.train()
        
        return {
            'u': u_grid,
            'v': v_grid,
            'p': p_grid,
            'loss_history': loss_history,
            'loss_components_history': loss_components_history,
            'iterations': iteration + 1,
            'model': self.model  # Return model for further evaluation if needed
        }


if __name__ == "__main__":
    # Example usage
    print("Solving lid-driven cavity flow problem using PINNs...")
    
    # Create solver
    solver = LidDrivenCavitySolver(Re=100, n_collocation=2000, n_boundary=100)
    
    # Solve
    solution = solver.solve(max_iter=5000, lr=0.001, verbose=True)
    
    print(f"\nSolution completed in {solution['iterations']} iterations")
    print(f"Final loss: {solution['loss_history'][-1]:.2e}")
