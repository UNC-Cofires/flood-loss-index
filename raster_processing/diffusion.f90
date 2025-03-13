MODULE diffusion
IMPLICIT NONE
CONTAINS

    FUNCTION steady_state_diffusion(Nx,Ny,dx,dy,C0,Nbc,bc_coords,bc_values,Nconv,conv_coords,tol,omega) RESULT(C)
    
        ! This function computes the steady-state solution to the 2-D diffusion equation
        ! using the successive over-relaxation (SOR) method. Sources and sinks are specified 
        ! based on dirichlet boundary condition coordinates and values supplied by the user. 
        ! Edges of the computational domain are subject to a no-flux boundary condition.  

        INTEGER, INTENT(IN) :: Nx, Ny, Nbc, Nconv
        REAL(KIND = 8), INTENT(IN) :: dx, dy, tol, omega
        REAL(KIND = 8), DIMENSION(Ny,Nx), INTENT(IN) :: C0
        INTEGER, DIMENSION(2,Nbc), INTENT(IN) :: bc_coords
        REAL(KIND = 8), DIMENSION(Nbc), INTENT(IN) :: bc_values
        INTEGER, DIMENSION(2,Nconv), INTENT(IN) :: conv_coords

        REAL(KIND = 8), DIMENSION(Ny,Nx) :: C, C_prev
        INTEGER, DIMENSION(Ny,Nx) :: bc_cell
        INTEGER, DIMENSION(2,2*(Nx+Ny)) :: edge_bc_coords
        REAL(KIND = 8), DIMENSION(2*(Nx+Ny)) :: edge_bc_values
        REAL(KIND = 8), DIMENSION(Nconv) :: diff
        REAL(KIND = 8) :: dx2, dy2, x_mult, y_mult, max_diff
        INTEGER :: i, j, idx, iter_num, num_edge_bc_cells
        
        ! Pre-compute values that will be used repeatedly in loop
        dx2 = dx**2.0
        dy2 = dy**2.0
        x_mult = dy2/(2.0*(dx2 + dy2))
        y_mult = dx2/(2.0*(dx2 + dy2))
        
        ! Set initial conditions
        C = C0
        bc_cell = 0
        num_edge_bc_cells = 0
        diff = 0.0
        max_diff = tol + 1
        iter_num = 0
        
        ! Enforce dirichlet boundary condition at user-supplied coordinates
        DO idx = 1,Nbc
        
            i = bc_coords(1,idx)
            j = bc_coords(2,idx)
            C(i,j) = bc_values(idx)
            bc_cell(i,j) = 1
            
            ! Note whether any of these cells fall on edge of computational domain 
            IF ((i==1).or.(i==Ny).or.(j==1).or.(j==Nx)) THEN
            
                num_edge_bc_cells = num_edge_bc_cells + 1
                edge_bc_coords(1,num_edge_bc_cells) = i
                edge_bc_coords(2,num_edge_bc_cells) = j
                edge_bc_values(num_edge_bc_cells) = bc_values(idx)
                
            ENDIF
            
        END DO
        
        ! Iteratively solve for steady state using successive over-relaxation (SOR) method
        DO WHILE (max_diff >= tol)
        
            iter_num = iter_num + 1
            C_prev = C
            
            ! Gauss-Siedel update step
            DO j = 2,Nx-1
                DO i = 2, Ny-1
                
                    ! Only update cells that are not subject to dirichlet boundary condition
                    IF (bc_cell(i,j)==0) THEN
                
                        C(i,j) = x_mult*(C_prev(i,j+1) + C(i,j-1)) + y_mult*(C_prev(i+1,j) + C(i-1,j))
                    
                    END IF
                    
                END DO
            END DO
            
            ! Over-relaxation step
            C = (1-omega)*C_prev + omega*C
            
            ! Enforce no-flux boundary condition on edge of array            
            C(:,1) = C(:,2)
            C(:,Nx) = C(:,Nx-1)
            C(1,:) = C(2,:)
            C(Ny,:) = C(Ny-1,:)
            
            ! Enforce any dirichlet boundary conditions that happen to be on edge of array
            ! (The previous step enforcing the no-flux BC would likely have altered these cells) 
            DO idx = 1, num_edge_bc_cells
            
                i = edge_bc_coords(1,idx)
                j = edge_bc_coords(2,idx)
                C(i,j) = edge_bc_values(idx)
            
            END DO
            
            ! Check for convergence to steady state at user-supplied coordinates
            DO idx = 1, Nconv
            
                i = conv_coords(1,idx)
                j = conv_coords(2,idx)
                diff(idx) = C(i,j) - C_prev(i,j)
                
            END DO
            
            ! Assess convergence based on maximum change from last iteration
            max_diff = MAXVAL(ABS(diff))
            
            PRINT "(a,i10,a,f10.9)", "Iteration: ",iter_num," Residual: ", max_diff
        
        END DO


    END FUNCTION steady_state_diffusion


END MODULE