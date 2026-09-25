module io

implicit none

contains

    subroutine gaussian(x, sigma, result)
        double precision, intent(in) :: x
        double precision, intent(in) :: sigma

        double precision, intent(out) :: result

        double precision, parameter :: pi  = 4.0d0*atan(1.0d0)

        !Normalized to one.
        result = exp(-0.5d0*(x**2)/(sigma)**2)!/(sigma*sqrt(2.0d0*pi))

    end subroutine gaussian

    subroutine get_map_from_bands(gridx, gridy, bands, weight, sigma, mmap, Nx, Ny, Nk, Nbands)
        double precision, dimension(Nx), intent(in) :: gridx
        double precision, dimension(Ny), intent(in) :: gridy
        double precision, dimension(Nk, Nbands), intent(in) :: bands, weight
        double precision, intent(in) :: sigma

        double precision, dimension(Nx,Ny), intent(out) :: mmap

        double precision :: tmp

        integer ::  Nx, Ny, Nk, Nbands
        integer :: xi, yi, Ni

        mmap = 0.0d0
        do xi = 1, Nx
            do yi = 1, Ny
                do Ni = 1, Nbands
                    call gaussian(bands(xi,Ni)-gridy(yi), sigma, tmp)
                    tmp = tmp*weight(xi,Ni)
                    mmap(xi,yi) = mmap(xi,yi)+tmp
                end do
            end do
        end do
    end subroutine
end module io

