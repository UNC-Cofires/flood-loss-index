SUBROUTINE stream_distance(nrows,ncols,ncoords,i_coords,j_coords,&
i_step,j_step,stepsize,nodata_value,maxiter,i_nearest,j_nearest,dist)

    ! This function computes the distance to stream by tracing the flow path to the nearest non-missing point,
    ! and back-propagating the indicies of and distance to the nearest stream point or sink

    IMPLICIT  NONE

    INTEGER, INTENT(IN) :: nrows, ncols, ncoords, nodata_value, maxiter
    INTEGER, DIMENSION(ncoords), INTENT(IN) :: i_coords, j_coords
    INTEGER, DIMENSION(nrows,ncols), INTENT(IN) :: i_step, j_step
    REAL(KIND=8), DIMENSION(nrows,ncols), INTENT(IN) :: stepsize
    INTEGER, DIMENSION(nrows,ncols), INTENT(INOUT) :: i_nearest, j_nearest
    REAL(KIND=8), DIMENSION(nrows,ncols), INTENT(INOUT) :: dist

    INTEGER :: idx, i, j, k, i0, j0, di, dj, i_burn, j_burn, numiter
    REAL(KIND=8) :: flowpath_length, smallnum
    LOGICAL :: keepgoing, backfill

    ! Small number we'll use to check whether a value is zero or near-zero
    smallnum = 1E-10

    DO idx = 1, ncoords

        ! Initialize variables at starting point

        keepgoing = .TRUE.
        backfill = .FALSE.
        flowpath_length = 0.0        
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

                IF (dist(i,j) == nodata_value) THEN

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
                        flowpath_length = flowpath_length + stepsize(i,j)
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
                    flowpath_length = flowpath_length + dist(i,j)

                END IF

            END IF

        END DO

        ! Backfill indicies of and distance to nearest stream point or sink if known
        IF (backfill) THEN

            i = i0
            j = j0

            DO k = 1, numiter

                i_nearest(i,j) = i_burn
                j_nearest(i,j) = j_burn
                dist(i,j) = flowpath_length

                flowpath_length = flowpath_length - stepsize(i,j)
                di = i_step(i,j)
                dj = j_step(i,j)
                i = i + di
                j = j + dj

            END DO

        END IF

    END DO

    RETURN 

END SUBROUTINE