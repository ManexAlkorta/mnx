module bands
    implicit none

contains

subroutine get_nesting(Ndown, modulation, downsample, hugek, diffuse_fermis, &
 Nsk, downk, ispin, Nk)

    integer, intent(in) :: Ndown

    integer, dimension(3), intent(in) :: modulation, downsample
    integer, dimension(Nk, 3), intent(in) :: hugek
    double precision, dimension(ispin, Nk), intent(in) :: diffuse_fermis

    double precision, dimension(ispin, Ndown), intent(out) :: Nsk
    double precision, dimension(Ndown, 3), intent(out) :: downk

    integer, dimension(3) :: downmod
    integer, dimension(3) :: kp
    double precision, allocatable :: diffuse_lookup(:,:,:,:)
    integer :: ispin, Nk, ki_idx, si, q1i, q2i, q3i, qi, sqi, k1i, k2i, k3i, ki

    downmod(1) = modulation(1) / downsample(1)
    downmod(2) = modulation(2) / downsample(2)
    downmod(3) = modulation(3) / downsample(3)

    allocate(diffuse_lookup(ispin, modulation(1), modulation(2), modulation(3)))

    do ki_idx = 1, Nk
        do si = 1, ispin
            diffuse_lookup(si, &
                hugek(ki_idx, 1) + 1, &
                hugek(ki_idx, 2) + 1, &
                hugek(ki_idx, 3) + 1) = diffuse_fermis(si, ki_idx)
        end do
    end do

    Nsk = 0.0d0

    do si = 1, ispin
        do q1i = 0, downmod(1) - 1
            do q2i = 0, downmod(2) - 1
                do q3i = 0, downmod(3) - 1
                    qi = q1i * downsample(1) * modulation(2) * modulation(3) + &
                         q2i * downsample(2) * modulation(3) + &
                         q3i * downsample(3) + 1

                    sqi = q1i * downmod(2) * downmod(3) + q2i * downmod(3) + q3i + 1

                    downk(sqi, 1) = 1.0d0 * hugek(qi, 1) / dble(downsample(1))
                    downk(sqi, 2) = 1.0d0 * hugek(qi, 2) / dble(downsample(2))
                    downk(sqi, 3) = 1.0d0 * hugek(qi, 3) / dble(downsample(3))

                    do k1i = 0, downmod(1) - 1
                        do k2i = 0, downmod(2) - 1
                            do k3i = 0, downmod(3) - 1

                                ki = k1i * downsample(1) * modulation(2) * modulation(3) + &
                                     k2i * downsample(2) * modulation(3) + &
                                     k3i * downsample(3) + 1

                                kp(:) = hugek(ki, :) + hugek(qi, :)

                                if (kp(1) >= modulation(1)) kp(1) = kp(1) - modulation(1)
                                if (kp(2) >= modulation(2)) kp(2) = kp(2) - modulation(2)
                                if (kp(3) >= modulation(3)) kp(3) = kp(3) - modulation(3)

                                Nsk(si, sqi) = Nsk(si, sqi) + &
                                    diffuse_lookup(si, hugek(ki, 1) + 1, hugek(ki, 2) + 1, hugek(ki, 3) + 1) * &
                                    diffuse_lookup(si, kp(1) + 1, kp(2) + 1, kp(3) + 1)

                            end do
                        end do
                    end do

                end do
            end do
        end do
    end do

    deallocate(diffuse_lookup)
end subroutine get_nesting
end module bands