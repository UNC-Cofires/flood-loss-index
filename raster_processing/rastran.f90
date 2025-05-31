SUBROUTINE flowpath_integral(nrows,ncols,ncoords,i_coords,j_coords,&
i_step,j_step,stepsize,integrand,nodata_value,maxiter,i_nearest,j_nearest,integral)

    ! This function computes the path integral of a function while tracing the flow of water
    ! from a user-supplied point to the nearest stream or sink point.  
    !
    ! param: nrows: number of rows in raster array
    ! param: ncols: number of columns in raster array
    ! param: ncoords: number of points from which to trace flowpath
    ! param: i_coords: i-coordinates of points from which to trace flow (vector of length ncoords)
    ! param: j_coords: j-coordinates of points from which to trace flow (vector of length ncoords)
    ! param: i_step: direction of flow in i-dimension (nrows x ncols array). Can be +1, 0, -1, or nodata. 
    ! param: j_step: direction of flow in j-dimension (nrows x ncols array). Can be +1, 0, -1, or nodata. 
    ! param: stepsize: total distance covered by flow path step (nrowx x ncols array)
    ! param: integrand: value of function to be integrated over flow path (nrows x ncols array)
    ! param: nodata_value: value used to denote presence of missing data in array
    ! param: maxiter: maximum number of flow tracing steps to attempt before quitting.
    ! param (in/out): i_nearest: i-coordinates of nearest stream point or sink (nrows x ncols array). 
    ! param (in/out): j_nearest: j-coordinates of nearest stream point or sink (nrows x ncols array). 
    ! param (in/out): integral: value of integral at each point (nrows x ncols array). 
    !                 Stream points should initially be populated, while non-stream points should be NA. 

    IMPLICIT  NONE

    INTEGER, INTENT(IN) :: nrows, ncols, ncoords, nodata_value, maxiter
    INTEGER, DIMENSION(ncoords), INTENT(IN) :: i_coords, j_coords
    INTEGER, DIMENSION(nrows,ncols), INTENT(IN) :: i_step, j_step
    REAL(KIND=8), DIMENSION(nrows,ncols), INTENT(IN) :: stepsize, integrand
    
    INTEGER, DIMENSION(nrows,ncols), INTENT(INOUT) :: i_nearest, j_nearest
    REAL(KIND=8), DIMENSION(nrows,ncols), INTENT(INOUT) :: integral

    INTEGER :: idx, i, j, k, i0, j0, di, dj, i_burn, j_burn, numiter
    REAL(KIND=8) :: running_sum, smallnum
    LOGICAL :: keepgoing, backfill

    ! Small number we'll use to check whether a value is zero or near-zero
    smallnum = 1E-10

    DO idx = 1, ncoords

        ! Initialize variables at starting point

        keepgoing = .TRUE.
        backfill = .FALSE.
        running_sum = 0.0        
        i0 = i_coords(idx)
        j0 = j_coords(idx)

        numiter = 0
        i = i0
        j = j0

        ! Trace path of flow until stopping condition is reached

        DO WHILE (numiter < maxiter .AND. keepgoing)

            numiter = numiter + 1

            IF (i < 1 .OR. j < 1 .OR. i > nrows .OR. j > ncols) THEN

                ! Stop tracing flow if it goes off the edge of the map
                keepgoing = .FALSE.

            ELSE

                IF (integral(i,j) == nodata_value) THEN

                    IF (stepsize(i,j) == nodata_value) THEN

                        ! Stop tracing flow if it exits the area where flow direction raster is defined
                        keepgoing = .FALSE.

                    ELSE IF (stepsize(i,j) < smallnum) THEN

                        ! Stop tracing flow if we reach a sink
                        keepgoing = .FALSE.
                        backfill = .TRUE.
                        i_burn = i
                        j_burn = j

                    ELSE

                        ! Advance to the next position in the flow path
                        running_sum = running_sum + stepsize(i,j)*integrand(i,j)
                        di = i_step(i,j)
                        dj = j_step(i,j)                        
                        i = i + di
                        j = j + dj

                    END IF

                ELSE

                    ! Stop if you reach a point on the map where the nearest stream point is known
                    keepgoing = .FALSE.
                    backfill = .TRUE.
                    i_burn = i_nearest(i,j)
                    j_burn = j_nearest(i,j)
                    running_sum = running_sum + integral(i,j)

                END IF

            END IF

        END DO

        ! Backfill indicies of and integralance to nearest stream point or sink if known
        IF (backfill) THEN

            i = i0
            j = j0

            DO k = 1, numiter

                i_nearest(i,j) = i_burn
                j_nearest(i,j) = j_burn
                integral(i,j) = running_sum
                
                running_sum = running_sum - stepsize(i,j)*integrand(i,j)

                di = i_step(i,j)
                dj = j_step(i,j)
                i = i + di
                j = j + dj

            END DO

        END IF

    END DO

    RETURN 

END SUBROUTINE

SUBROUTINE sample_nearest(nrows,ncols,nodata_value,i_nearest,j_nearest,src_arr,dst_arr)

    ! This function propagates values from src_arr based on the coordinates encoded in 
    ! the i_nearest and j_nearest arrays. Returns a dst_arr of same shape. 
    ! For example, can use this to get the elevation of the nearest stream at each point. 

    IMPLICIT  NONE
    
    INTEGER, INTENT(IN) :: nrows, ncols, nodata_value
    INTEGER, DIMENSION(nrows,ncols), INTENT(IN) :: i_nearest, j_nearest
    REAL(KIND=8), DIMENSION(nrows,ncols), INTENT(IN) :: src_arr
    
    REAL(KIND=8), DIMENSION(nrows,ncols), INTENT(OUT) :: dst_arr
    
    INTEGER :: i, j, i_src, j_src
    
    dst_arr = nodata_value
    
    DO i = 1, nrows
    
        DO j = 1, ncols
        
            i_src = i_nearest(i,j)
            j_src = j_nearest(i,j)
        
            IF (i_src /= nodata_value .AND. j_src /= nodata_value) THEN
            
                dst_arr(i,j) = src_arr(i_src,j_src)
            
            END IF
        
        END DO
        
    END DO
    
    RETURN
    
END SUBROUTINE

    
    
    
    

