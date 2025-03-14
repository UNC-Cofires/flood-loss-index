import numpy as np

# Wrapped Fortran function created with F2PY
# Compiled extension module files must be located in same directory 
import diffusion

def steady_state_diffusion_2D(C0,dirichlet_bc,dx=1,dy=1,tol=1e-6,omega=1.0,conv_mask=None):
    """
    This function computes the steady-state solution to the 2D diffusion equation on a rectangular grid: 
    
    0 = d²C/dx² + d²C/dy²
    
    param: C0: initial guess for C (ny x nx array)
    param: dirichlet_bc: dirichlet boundary conditions (ny x nx array). Populated elements denote 
                         fixed values of C. Non-fixed (i.e., unknown) elements should be NaN valued. 
    param: dx: grid spacing in x-direction (i.e., between columns)
    param: dy: grid spacing in y-direction (i.e., between rows)
    param: tol: tolerance used to assess convergence
    param: omega: value of over-relaxation parameter used to accelerate convergence (should be between 1.0 and 1.9). 
    param: conv_mask: boolean array denoting elements to use when assessing convergence. 
    
    returns: C: solution to steady state diffusion equation subject to specified boundary conditions.
    
    notes: if not already subject to a dirichlet boundary condition, edges of computational domain 
           are assumed to be subject to a no-flux boundary condition. 
    """
    
    # Get shape of computational domain
    ny,nx = C0.shape
    
    # Get coordinates of elements that act as sources or sinks 
    ii,jj = np.meshgrid(np.arange(ny),np.arange(nx),indexing='ij')
    bc_mask = ~np.isnan(dirichlet_bc)
    bc_i = ii.ravel(order='F')[bc_mask.ravel(order='F')]
    bc_j = jj.ravel(order='F')[bc_mask.ravel(order='F')]
    bc_coords = np.array([bc_i,bc_j])
    bc_values = dirichlet_bc.ravel(order='F')[bc_mask.ravel(order='F')]
    
    # Get coordinates of elements that are used to assess convergence of solution
    # If not specified by user, will use all elements. 
    if conv_mask is None:
        (np.ones(C0.shape)==1)
    conv_i = ii.ravel(order='F')[conv_mask.ravel(order='F')]
    conv_j = jj.ravel(order='F')[conv_mask.ravel(order='F')]
    conv_coords = np.array([conv_i,conv_j])
    
    # Increment coordinates since Fortran indexing starts at 1
    bc_coords += 1
    conv_coords += 1
    
    # Call wrapped Fortran function created with F2PY
    # (much faster than pure Python implementation) 
    C = diffusion.diffusion(dx,dy,C0,bc_coords,bc_values,conv_coords,tol,omega)
    
    return(C)


