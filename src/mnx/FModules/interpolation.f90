module interpolation

implicit none

public :: ift_fcq
public :: ft_fcr
public :: ft_phir
public :: dipole_dipole
public :: nonanalytic

contains

    subroutine ift_fcq(phiqs,qs,prim_cell,super_atom_coords,gauge,phiR,Nqpoint,prim_natoms,super_natoms)
        complex*16, dimension(Nqpoint, prim_natoms, prim_natoms, 3, 3), intent(in) :: phiqs
        double precision, dimension(Nqpoint, 3), intent(in) :: qs
        double precision, dimension(3,3) :: prim_cell
        double precision, dimension(super_natoms,3), intent(in) :: super_atom_coords

        character(len=*), intent(in) :: gauge

        double precision, dimension(super_natoms,super_natoms,3,3), intent(out) :: phiR
        
        integer :: Nqpoint, prim_natoms, super_natoms
        integer :: a, b, qi
        complex*16 :: phase
        double precision, dimension(3) :: R
        double precision, parameter :: pi  = 4.0d0*atan(1.0d0)
        complex*16, parameter :: j = (0.0d0, 1.0d0)

        phiR = 0.0d0
        do a = 1, super_natoms
            do b = 1, super_natoms
                if (gauge == "lattice") then
                    call cart2cryst( &
super_atom_coords((b-1)/prim_natoms*prim_natoms+1,:)-super_atom_coords((a-1)/prim_natoms*prim_natoms+1,:), prim_cell, R)
                else if (gauge == "atomic") then
                    call cart2cryst( &  
super_atom_coords(b,:)-super_atom_coords(a,:), prim_cell, R)
                end if
                do qi = 1, Nqpoint
                    phase = exp(-2*j*pi*dot_product(qs(qi,:),R))
                    phiR(a,b,:,:) = phiR(a,b,:,:) + real(phiqs(qi,mod(a-1,prim_natoms)+1,mod(b-1,prim_natoms)+1,:,:)*phase)
                end do
            end do
        end do
        phiR = phiR / Nqpoint
    end subroutine
    
    subroutine ft_fcr(phiR,q,masses,modulation,prim_cell,super_atom_coords,gauge, &
phiq, frequencies, polvecs, prim_natoms, super_natoms)
        ! Fourier transform the real-space force constants to q (ft_phir) and diagonalize.

        implicit none

        double precision, dimension(super_natoms, super_natoms, 3, 3), intent(in) :: phiR
        double precision, dimension(3), intent(in) :: q
        double precision, dimension(prim_natoms), intent(in) :: masses
        double precision, dimension(3,3), intent(in) :: prim_cell
        double precision, dimension(super_natoms,3), intent(in) :: super_atom_coords

        integer, dimension(3), intent(in) :: modulation
        character(len=*), intent(in) :: gauge

        complex*16, dimension(prim_natoms,prim_natoms,3,3), intent(out) :: phiq
        double precision, dimension(prim_natoms*3), intent(out) :: frequencies
        complex*16, dimension(prim_natoms*3,prim_natoms*3), intent(out) :: polvecs

        integer :: prim_natoms, super_natoms

        call ft_phir(phiR,q,modulation,prim_cell,super_atom_coords,gauge,phiq,prim_natoms,super_natoms)
        call diag_dynq(phiq,masses,prim_natoms,frequencies,polvecs)
    end subroutine

    subroutine ft_phir(phiR,q,modulation,prim_cell,super_atom_coords,gauge,phiq,prim_natoms,super_natoms)
        ! Fourier transform the real-space force constants to q with Wigner-Seitz folding.

        implicit none

        double precision, dimension(super_natoms, super_natoms, 3, 3), intent(in) :: phiR
        double precision, dimension(3), intent(in) :: q
        double precision, dimension(3,3), intent(in) :: prim_cell
        double precision, dimension(super_natoms,3), intent(in) :: super_atom_coords

        integer, dimension(3), intent(in) :: modulation
        character(len=*), intent(in) :: gauge

        complex*16, dimension(prim_natoms,prim_natoms,3,3), intent(out) :: phiq

        integer, dimension(27,3) :: shifts
        integer :: prim_natoms, super_natoms
        integer :: a, b, b_super, si, nx, ny, nz, n_cells, t, n_equiv
        complex*16 :: phase_weight
        complex*16, dimension(8) :: equiv_phases
        double precision, dimension(3) :: R_base, R_test, R_cart
        double precision :: dist, min_dist
        double precision, parameter :: pi  = 4.0d0*atan(1.0d0)
        complex*16, parameter :: j = (0.0d0, 1.0d0)
        phiq = 0.0d0
        n_cells = modulation(1)*modulation(2)*modulation(3)
        si = 0
        do nx = -1,1
            do ny = -1,1
                do nz = -1,1
                    si = si+1
                    shifts(si,1) = nx*modulation(1)
                    shifts(si,2) = ny*modulation(2)
                    shifts(si,3) = nz*modulation(3)
                end do
            end do
        end do
        do t = 1, n_cells
            !omp parallel do schedule(dynamic,1) private(b,b_super,R_base,min_dist,n_equiv,si,R_test,R_cart,dist,equiv_phases,phase_weight)
            do a = 1, prim_natoms
                do b = 1, prim_natoms
                    b_super = b+(t-1)*prim_natoms
                    if (gauge == "lattice") then
                        call cart2cryst(super_atom_coords((t-1)*prim_natoms+1,:)-super_atom_coords(1,:), prim_cell, R_base)
                    else if (gauge == "atomic") then
                        call cart2cryst(super_atom_coords(b_super,:)-super_atom_coords(a,:), prim_cell, R_base)
                    end if
                    min_dist = 1.0d10
                    n_equiv = 0
                    ! 1. Search the 3x3x3 neighboring cells (shifts)
                    do si=1,27
                        ! R_test = R_base + shift
                        R_test = R_base + shifts(si,:)
                        ! Convert R_test to Cartesian to check actual distance
                        ! R_cart = R_test . prim_cell
                        R_cart = matmul(R_test, prim_cell)
                        dist = sqrt(dot_product(R_cart, R_cart))

                        ! 2. Check for shortest image
                        if (dist < (min_dist - 1.0d-5)) then
                            ! Found a new strictly shorter distance
                            min_dist = dist
                            n_equiv = 1
                            equiv_phases(1) = exp(2*j*pi*dot_product(q, R_test))
                        
                        else if (abs(dist - min_dist) < 1.0d-5) then
                            ! Found another equidistant point (boundary of Wigner-Seitz)
                            if (n_equiv < 8) then
                                n_equiv = n_equiv + 1
                                equiv_phases(n_equiv) = exp(2*j*pi*dot_product(q, R_test))
                            end if
                        end if
                    end do
                    phase_weight = sum(equiv_phases(1:n_equiv)) / dble(n_equiv)
                    phiq(a,b,:,:) = phiq(a,b,:,:)+phiR(a,b_super,:,:)*phase_weight
                end do
            end do
            !omp end parallel do
        end do
    end subroutine

    subroutine dipole_dipole(q,tau,rcell,epsil,zeu,omega,alpha,gmax,gauge,phi_lr,nat)
        ! Long-range dipole-dipole force constants of a polar crystal at q (Ry/bohr^2).
        !
        ! Ewald sum in reciprocal space of X. Gonze and C. Lee, PRB 55, 10355 (1997),
        ! ported from QuantumESPRESSO's rgd_blk (3D case). The G = 0 term is left out at
        ! q = 0, so the result at Gamma is analytic; see nonanalytic for the q -> 0 limit.
        !
        ! q      : cartesian q-point in bohr^-1, including 2*pi
        ! tau    : atomic positions in bohr, tau(atom, cartesian)
        ! rcell  : reciprocal lattice vectors in bohr^-1 including 2*pi, one per row
        ! epsil  : high-frequency dielectric tensor
        ! zeu    : Born effective charges, zeu(atom, field, displacement)
        ! omega  : cell volume in bohr^3
        ! alpha  : Ewald parameter in bohr^-2; G-vectors with |G|^2/(4 alpha) > gmax are left out
        ! gauge  : "lattice" (QuantumESPRESSO phases) or "atomic" (phases with the atomic positions)
        implicit none

        integer :: nat
        double precision, dimension(3), intent(in) :: q
        double precision, dimension(nat,3), intent(in) :: tau
        double precision, dimension(3,3), intent(in) :: rcell, epsil
        double precision, dimension(nat,3,3), intent(in) :: zeu
        double precision, intent(in) :: omega, alpha, gmax
        character(len=*), intent(in) :: gauge
        complex*16, dimension(nat,nat,3,3), intent(out) :: phi_lr

        integer :: a, b, i, k, m1, m2, m3
        integer, dimension(3) :: nrx
        double precision :: fac, geg, facgd, arg
        double precision, dimension(3) :: G, zag, zbg, fnat
        double precision, parameter :: pi = 4.0d0*atan(1.0d0)
        double precision, parameter :: e2 = 2.0d0
        complex*16, parameter :: j = (0.0d0, 1.0d0)

        phi_lr = (0.0d0, 0.0d0)
        fac = e2*4.0d0*pi/omega

        ! Range of G-vectors that contains every |G|^2 < 4*alpha*gmax.
        do i = 1, 3
            nrx(i) = int(sqrt(4.0d0*alpha*gmax)/sqrt(dot_product(rcell(i,:), rcell(i,:)))) + 1
        end do

        do m1 = -nrx(1), nrx(1)
        do m2 = -nrx(2), nrx(2)
        do m3 = -nrx(3), nrx(3)
            G = m1*rcell(1,:) + m2*rcell(2,:) + m3*rcell(3,:)

            ! q-independent G /= 0 term on the diagonal, it makes the sum vanish for uniform translations.
            geg = dot_product(G, matmul(epsil, G))
            if (geg > 0.0d0 .and. geg/alpha/4.0d0 < gmax) then
                facgd = fac*exp(-geg/alpha/4.0d0)/geg
                do a = 1, nat
                    zag = matmul(G, zeu(a,:,:))
                    fnat = 0.0d0
                    do b = 1, nat
                        arg = dot_product(G, tau(a,:)-tau(b,:))
                        fnat = fnat + matmul(G, zeu(b,:,:))*cos(arg)
                    end do
                    do i = 1, 3
                        do k = 1, 3
                            phi_lr(a,a,i,k) = phi_lr(a,a,i,k) - facgd*zag(i)*fnat(k)
                        end do
                    end do
                end do
            end if

            ! q + G term.
            G = G + q
            geg = dot_product(G, matmul(epsil, G))
            if (geg > 0.0d0 .and. geg/alpha/4.0d0 < gmax) then
                facgd = fac*exp(-geg/alpha/4.0d0)/geg
                do a = 1, nat
                    zag = matmul(G, zeu(a,:,:))
                    do b = 1, nat
                        zbg = matmul(G, zeu(b,:,:))
                        arg = dot_product(G, tau(a,:)-tau(b,:))
                        do i = 1, 3
                            do k = 1, 3
                                phi_lr(a,b,i,k) = phi_lr(a,b,i,k) + facgd*exp(j*arg)*zag(i)*zbg(k)
                            end do
                        end do
                    end do
                end do
            end if
        end do
        end do
        end do

        ! The sum above has the phases of the lattice gauge; the atomic gauge adds exp(i q.(tau_b - tau_a)).
        if (gauge == "atomic") then
            do a = 1, nat
                do b = 1, nat
                    phi_lr(a,b,:,:) = phi_lr(a,b,:,:)*exp(j*dot_product(q, tau(b,:)-tau(a,:)))
                end do
            end do
        end if
    end subroutine

    subroutine nonanalytic(qhat,epsil,zeu,omega,phi_na,nat)
        ! Non-analytic term of the force constants at q -> 0 along qhat (Ry/bohr^2), the LO-TO splitting:
        !
        !   phi_na(a,b,i,k) = 4 pi e^2 / omega * (qhat.Z_a)_i (qhat.Z_b)_k / (qhat.epsil.qhat)
        !
        ! as in QuantumESPRESSO's nonanal, PRB 55, 10355 (1997) Eq. (60). Only the direction of qhat
        ! matters; a vanishing qhat gives a vanishing term.
        implicit none

        integer :: nat
        double precision, dimension(3), intent(in) :: qhat
        double precision, dimension(3,3), intent(in) :: epsil
        double precision, dimension(nat,3,3), intent(in) :: zeu
        double precision, intent(in) :: omega
        double precision, dimension(nat,nat,3,3), intent(out) :: phi_na

        integer :: a, b, i, k
        double precision :: qeq
        double precision, dimension(3) :: zag, zbg
        double precision, parameter :: pi = 4.0d0*atan(1.0d0)
        double precision, parameter :: e2 = 2.0d0

        phi_na = 0.0d0
        qeq = dot_product(qhat, matmul(epsil, qhat))
        if (qeq < 1.0d-8*dot_product(qhat, qhat)) return
        do a = 1, nat
            zag = matmul(qhat, zeu(a,:,:))
            do b = 1, nat
                zbg = matmul(qhat, zeu(b,:,:))
                do i = 1, 3
                    do k = 1, 3
                        phi_na(a,b,i,k) = 4.0d0*pi*e2*zag(i)*zbg(k)/qeq/omega
                    end do
                end do
            end do
        end do
    end subroutine

    subroutine cart2cryst(xyz, cell, coords_crystal)
        implicit none
        real(8), intent(in)  :: xyz(3)
        real(8), intent(in)  :: cell(3,3)
        real(8), intent(out) :: coords_crystal(3)

        real(8) :: A(3,3), Ainv(3,3)
        real(8) :: det

        ! Transpose of cell
        A = transpose(cell)

        ! Determinant
        det = A(1,1)*(A(2,2)*A(3,3) - A(2,3)*A(3,2)) &
            - A(1,2)*(A(2,1)*A(3,3) - A(2,3)*A(3,1)) &
            + A(1,3)*(A(2,1)*A(3,2) - A(2,2)*A(3,1))

        if (abs(det) < 1.0d-12) then
            print *, "Matrix is singular!"
            stop
        end if

        ! Inverse of A (transpose(cell))
        Ainv(1,1) =  (A(2,2)*A(3,3) - A(2,3)*A(3,2)) / det
        Ainv(1,2) = -(A(1,2)*A(3,3) - A(1,3)*A(3,2)) / det
        Ainv(1,3) =  (A(1,2)*A(2,3) - A(1,3)*A(2,2)) / det

        Ainv(2,1) = -(A(2,1)*A(3,3) - A(2,3)*A(3,1)) / det
        Ainv(2,2) =  (A(1,1)*A(3,3) - A(1,3)*A(3,1)) / det
        Ainv(2,3) = -(A(1,1)*A(2,3) - A(1,3)*A(2,1)) / det

        Ainv(3,1) =  (A(2,1)*A(3,2) - A(2,2)*A(3,1)) / det
        Ainv(3,2) = -(A(1,1)*A(3,2) - A(1,2)*A(3,1)) / det
        Ainv(3,3) =  (A(1,1)*A(2,2) - A(1,2)*A(2,1)) / det

        ! coords_crystal = Ainv * xyz
        coords_crystal = matmul(Ainv, xyz)

    end subroutine cart2cryst

subroutine diag_dynq(phiq, masses, prim_natoms, frequencies, polvecs)

    implicit none

    integer, intent(in) :: prim_natoms

    ! FIX: force double precision complex
    complex*16, intent(in) :: phiq(prim_natoms, prim_natoms, 3, 3)
    double precision, intent(in) :: masses(prim_natoms)

    ! Outputs
    double precision, intent(out) :: frequencies(3*prim_natoms)
    complex*16, intent(out) :: polvecs(3*prim_natoms, 3*prim_natoms)

    ! LAPACK variables
    integer :: n, info, lwork
    complex*16, allocatable :: work(:)
    double precision, allocatable :: rwork(:)
    complex*16 :: work_query(1)

    integer :: a, b

    ! >>> ADDED for gauge fix
    integer :: i, imax
    double precision :: maxval, norm
    complex*16 :: phase

    n = 3 * prim_natoms

    ! 1. Build dynamical matrix
    do a = 1, prim_natoms
        do b = 1, prim_natoms
            polvecs((a-1)*3+1:a*3, (b-1)*3+1:b*3) = &
                phiq(a,b,:,:) / sqrt(masses(a)*masses(b))
        end do
    end do

    ! 2. Workspace query
    lwork = -1
    allocate(rwork(max(1, 3*n-2)))

    call zheev('V', 'U', n, polvecs, n, frequencies, work_query, lwork, rwork, info)

    if (info /= 0) then
        print *, "ZHEEV workspace query failed:", info
        stop
    end if

    lwork = int(real(work_query(1)))

    if (lwork < 1) then
        print *, "Invalid LWORK:", lwork
        stop
    end if

    allocate(work(lwork))

    ! 3. Diagonalization
    call zheev('V', 'U', n, polvecs, n, frequencies, work, lwork, rwork, info)

    if (info /= 0) then
        print *, "Error in ZHEEV:", info
        stop
    end if

    deallocate(work, rwork)

    ! =====================================================
    ! 4. ADDED: QE-style gauge fixing (phase removal ONLY)
    ! =====================================================
    do a = 1, n

        ! find largest component
        imax = 1
        maxval = 0.0d0

        do i = 1, n
            if (abs(polvecs(i,a)) > maxval) then
                maxval = abs(polvecs(i,a))
                imax = i
            end if
        end do

        ! remove arbitrary complex phase
        if (abs(polvecs(imax,a)) > 1d-14) then
            phase =  polvecs(imax,a) / abs(polvecs(imax,a))
            polvecs(:,a) = - polvecs(:,a) / phase
        end if

        ! optional: ensure proper Hermitian normalization (safe)
        norm = sqrt(sum(conjg(polvecs(:,a)) * polvecs(:,a)))
        if (norm > 1d-14) then
            polvecs(:,a) = polvecs(:,a) / norm
        end if

    end do

end subroutine
end module interpolation