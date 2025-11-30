"""
Physics-Informed Neural Network (PINN) Loss for Navier-Stokes Equations
Uses automatic differentiation to compute PDE residuals
"""

import torch


class PINNLossNS:
    """
    Physics-Informed Neural Network loss for incompressible Navier-Stokes equations:
    
    Continuity: ∇·u = 0
    Momentum: (u·∇)u = -∇p/ρ + ν∇²u
    
    For steady-state lid-driven cavity, we solve:
    -∇p/ρ + ν∇²u + (u·∇)u = 0
    ∇·u = 0
    """
    
    def __init__(self, Re=100, rho=1.0, Lx=1.0, Ly=1.0):
        """
        Initialize PINN loss framework
        
        Args:
            Re: Reynolds number (Re = U*L/ν, where U is lid velocity, L is cavity length)
            rho: Fluid density
            Lx: Domain length in x-direction
            Ly: Domain length in y-direction
        """
        self.Re = Re
        self.rho = rho
        self.Lx = Lx
        self.Ly = Ly
        
        # Velocity scale (lid velocity)
        self.U_lid = 1.0
        # Kinematic viscosity from Reynolds number
        self.nu = self.U_lid * max(Lx, Ly) / Re
    
    def compute_derivatives(self, u, v, p, x, y):
        """
        Compute all necessary derivatives using automatic differentiation
        
        Args:
            u, v, p: velocity and pressure fields (tensors of shape [N])
            x, y: coordinates (tensors of shape [N])
            
        Returns:
            Dictionary with all derivatives
        """
        # First derivatives (x and y should already require gradients from solver)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), 
                                  create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), 
                                  create_graph=True, retain_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), 
                                  create_graph=True, retain_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), 
                                  create_graph=True, retain_graph=True)[0]
        
        # Second derivatives for Laplacian
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                    create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), 
                                    create_graph=True, retain_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), 
                                    create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), 
                                    create_graph=True, retain_graph=True)[0]
        
        return {
            'u_x': u_x, 'u_y': u_y,
            'v_x': v_x, 'v_y': v_y,
            'p_x': p_x, 'p_y': p_y,
            'u_xx': u_xx, 'u_yy': u_yy,
            'v_xx': v_xx, 'v_yy': v_yy
        }
    
    def compute_pde_residuals(self, u, v, p, x, y):
        """
        Compute PDE residuals using automatic differentiation
        
        Args:
            u, v, p: velocity and pressure fields (tensors of shape [N])
            x, y: coordinates (tensors of shape [N])
            
        Returns:
            Dictionary with residuals
        """
        # Compute all derivatives
        derivs = self.compute_derivatives(u, v, p, x, y)
        
        # Continuity equation: ∇·u = u_x + v_y = 0
        continuity_residual = derivs['u_x'] + derivs['v_y']
        
        # X-momentum: (u·∇)u + ∂p/∂x/ρ - ν∇²u = 0
        # (u·∇)u = u*u_x + v*u_y
        convective_x = u * derivs['u_x'] + v * derivs['u_y']
        pressure_term_x = derivs['p_x'] / self.rho
        laplacian_u = derivs['u_xx'] + derivs['u_yy']
        viscous_term_x = self.nu * laplacian_u
        momentum_x_residual = convective_x + pressure_term_x - viscous_term_x
        
        # Y-momentum: (u·∇)v + ∂p/∂y/ρ - ν∇²v = 0
        # (u·∇)v = u*v_x + v*v_y
        convective_y = u * derivs['v_x'] + v * derivs['v_y']
        pressure_term_y = derivs['p_y'] / self.rho
        laplacian_v = derivs['v_xx'] + derivs['v_yy']
        viscous_term_y = self.nu * laplacian_v
        momentum_y_residual = convective_y + pressure_term_y - viscous_term_y
        
        return {
            'momentum_x': momentum_x_residual,
            'momentum_y': momentum_y_residual,
            'continuity': continuity_residual
        }
    
    def compute_pde_loss(self, u, v, p, x, y):
        """
        Compute PDE loss (sum of squared residuals)
        
        Args:
            u, v, p: velocity and pressure fields (tensors of shape [N])
            x, y: coordinates (tensors of shape [N])
            
        Returns:
            Total loss and loss components
        """
        residuals = self.compute_pde_residuals(u, v, p, x, y)
        
        # Compute mean squared residuals
        loss_momentum_x = torch.mean(residuals['momentum_x']**2)
        loss_momentum_y = torch.mean(residuals['momentum_y']**2)
        loss_continuity = torch.mean(residuals['continuity']**2)
        
        total_loss = loss_momentum_x + loss_momentum_y + loss_continuity
        
        return total_loss, {
            'momentum_x': loss_momentum_x.item(),
            'momentum_y': loss_momentum_y.item(),
            'continuity': loss_continuity.item()
        }

