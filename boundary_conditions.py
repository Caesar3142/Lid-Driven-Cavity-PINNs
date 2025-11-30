"""
Boundary Conditions for Lid-Driven Cavity Flow using PINNs
Enforces boundary conditions through loss terms
"""

import torch


class LidDrivenCavityBC:
    """
    Boundary conditions for lid-driven cavity with free-slip walls:
    - Top wall (y=Ly): u = U_lid, v = 0 (moving lid)
    - Bottom wall (y=0): v = 0 (normal), ∂u/∂y = 0 (free-slip)
    - Left wall (x=0): u = 0 (normal), ∂v/∂x = 0 (free-slip)
    - Right wall (x=Lx): u = 0 (normal), ∂v/∂x = 0 (free-slip)
    
    Free-slip means:
    - Normal component of velocity = 0 (no flow through wall)
    - Tangential component has zero gradient (no shear stress)
    """
    
    def __init__(self, Lx=1.0, Ly=1.0, U_lid=1.0):
        """
        Initialize boundary conditions
        
        Args:
            Lx: Domain length in x-direction
            Ly: Domain length in y-direction
            U_lid: Lid velocity (top wall)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.U_lid = U_lid
    
    def sample_boundary_points(self, n_boundary=100):
        """
        Sample boundary points for enforcing boundary conditions
        
        Args:
            n_boundary: Number of points per boundary edge
            
        Returns:
            Dictionary with boundary point coordinates for each boundary
        """
        # Top wall (y = Ly): moving lid
        x_top = torch.linspace(0, self.Lx, n_boundary, requires_grad=True)
        y_top = torch.full((n_boundary,), self.Ly, requires_grad=True)
        
        # Bottom wall (y = 0): free-slip
        x_bottom = torch.linspace(0, self.Lx, n_boundary, requires_grad=True)
        y_bottom = torch.zeros(n_boundary, requires_grad=True)
        
        # Left wall (x = 0): free-slip
        x_left = torch.zeros(n_boundary, requires_grad=True)
        y_left = torch.linspace(0, self.Ly, n_boundary, requires_grad=True)
        
        # Right wall (x = Lx): free-slip
        x_right = torch.full((n_boundary,), self.Lx, requires_grad=True)
        y_right = torch.linspace(0, self.Ly, n_boundary, requires_grad=True)
        
        return {
            'top': {'x': x_top, 'y': y_top},
            'bottom': {'x': x_bottom, 'y': y_bottom},
            'left': {'x': x_left, 'y': y_left},
            'right': {'x': x_right, 'y': y_right}
        }
    
    def compute_boundary_loss(self, model, boundary_points):
        """
        Compute boundary condition loss terms
        
        Args:
            model: PINN model
            boundary_points: Dictionary with boundary point coordinates
            
        Returns:
            Total boundary loss and loss components
        """
        total_loss = 0.0
        loss_tensors = {}  # Store tensors, extract values later
        
        # Top wall: u = U_lid, v = 0
        u_top, v_top, _ = model(boundary_points['top']['x'], boundary_points['top']['y'])
        loss_top_u = torch.mean((u_top - self.U_lid)**2)
        loss_top_v = torch.mean(v_top**2)
        loss_tensors['top_u'] = loss_top_u
        loss_tensors['top_v'] = loss_top_v
        total_loss += loss_top_u + loss_top_v
        
        # Bottom wall: v = 0, ∂u/∂y = 0 (free-slip)
        u_bottom, v_bottom, _ = model(boundary_points['bottom']['x'], boundary_points['bottom']['y'])
        loss_bottom_v = torch.mean(v_bottom**2)
        
        # Compute ∂u/∂y at bottom wall
        u_y_bottom = torch.autograd.grad(
            u_bottom, boundary_points['bottom']['y'],
            grad_outputs=torch.ones_like(u_bottom),
            create_graph=True, retain_graph=True
        )[0]
        loss_bottom_du_dy = torch.mean(u_y_bottom**2)
        
        loss_tensors['bottom_v'] = loss_bottom_v
        loss_tensors['bottom_du_dy'] = loss_bottom_du_dy
        total_loss += loss_bottom_v + loss_bottom_du_dy
        
        # Left wall: u = 0, ∂v/∂x = 0 (free-slip)
        u_left, v_left, _ = model(boundary_points['left']['x'], boundary_points['left']['y'])
        loss_left_u = torch.mean(u_left**2)
        
        # Compute ∂v/∂x at left wall
        v_x_left = torch.autograd.grad(
            v_left, boundary_points['left']['x'],
            grad_outputs=torch.ones_like(v_left),
            create_graph=True, retain_graph=True
        )[0]
        loss_left_dv_dx = torch.mean(v_x_left**2)
        
        loss_tensors['left_u'] = loss_left_u
        loss_tensors['left_dv_dx'] = loss_left_dv_dx
        total_loss += loss_left_u + loss_left_dv_dx
        
        # Right wall: u = 0, ∂v/∂x = 0 (free-slip)
        u_right, v_right, _ = model(boundary_points['right']['x'], boundary_points['right']['y'])
        loss_right_u = torch.mean(u_right**2)
        
        # Compute ∂v/∂x at right wall
        v_x_right = torch.autograd.grad(
            v_right, boundary_points['right']['x'],
            grad_outputs=torch.ones_like(v_right),
            create_graph=True, retain_graph=True
        )[0]
        loss_right_dv_dx = torch.mean(v_x_right**2)
        
        loss_tensors['right_u'] = loss_right_u
        loss_tensors['right_dv_dx'] = loss_right_dv_dx
        total_loss += loss_right_u + loss_right_dv_dx
        
        # Extract values for logging (after graph is built)
        loss_components = {k: v.item() for k, v in loss_tensors.items()}
        
        return total_loss, loss_components
