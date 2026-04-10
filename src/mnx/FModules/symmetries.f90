module symmetries

contains

subroutine classify_singlets()
end subroutine
subroutine classify_doublets(nat, nat_sc, tot2, nref2, nsym, mappings, map_uc, nontrivial, M, &
        orbit3t, orbit3o, norbit, indep_fc, n_indep_fc, kernel, mapping_doublet)

        integer, intent(in) :: nat, nat_sc, nsym, tot2, nref2
        integer, dimension(:,:,:), intent(in) :: nontrivial
        double precision, dimension(:,:,:,:), intent(in) :: M
        integer, dimension(:,:), intent(in) :: mappings, map_uc

        double precision, dimension(nref2,9,9), intent(out) :: kernel
        integer, dimension(nref2,27), intent(out) :: indep_fc 
        integer, dimension(nref2), intent(out) :: norbit, n_indep_fc
        integer, dimension(nref2,6*nsym,3), intent(out) :: orbit3t
        integer, dimension(nref2,6*nsym,2), intent(out) :: orbit3o
        integer, dimension(nat,nat_sc,3), intent(out) :: mapping_doublet

        double precision, allocatable, dimension(:,:) :: constrain_reduced

        integer, dimension(tot2,3) :: all2
        integer, dimension(6*nsym, 3) :: equilist
        integer, dimension(9) :: indep
        double precision, dimension(9,9) :: kern
        double precision, dimension(6*nsym*9,9) :: constrain

        integer, dimension(2,2) :: permutations
        integer, dimension(2) :: doublet, doublet_perm, doublet_sym
        integer :: ii, jj, kk, nall2, equiv, nconstrain, iperm, isym, iaux, &
jaux, indexprime, ll, nindep, iw, ref2
        logical :: its_in_list

        character(len=100) :: filename


        kernel = 0
        orbit3t = 0
        orbit3o = 0
        mapping_doublet = 0
        permutations = reshape([1,2,2,1],[2,2])

        ref3 = 0
        nall3 = 0
        doii : do ii = 0, nat-1
                print *, "New ii: ", ii
                dojj : do jj = 0, nat_sc-1
                        doubet = [ii, jj]
                        call doublet_in_list(doublet, all2, nall2, its_in_list)
                        if (its_in_list) cycle dojj
                        ref2 = ref2 + 1
                        equiv = 0
                        nconstrain = 0
                        constrain = 0
                        do iperm = 1, 2
                                doublet_perm(1) = doublet(permutations(iperm,1))
                                doublet_perm(2) = doublet(permutations(iperm,2))
                                doublet_perm(3) = doublet(permutations(iperm,3))
                                do isym = 1, nsym
                                        doublet_sym = &
[mappings(doublet_perm(1)+1,isym), mappings(doublet_perm(2)+1,isym)]
                                        if (doublet_sym(1) >= nat) then
                                                doublet_sym(2) = map_uc(doublet_sym(1)+1, doublet_sym(2)+1)
                                                doublet_sym(1) = map_uc(doublet_sym(1)+1, doublet_sym(1)+1)
                                        end if
                                        call doublet_in_list(doublet_sym, equilist, equiv, its_in_list)
                                        if (((iperm==1) .and. (isym==1)) .or. (.not. its_in_list)) then
                                                equilist(equiv+1,:) = doublet_sym
                                                orbit2t(ref2,equiv+1,:) = doublet_sym
                                                all2(nall2+1,:) = doublet_sym
                                                orbit2o(ref3,equiv+1,:) = [iperm-1, isym-1]
                                                mapping_doublet(doublet_sym(1)+1,doublet_sym(2)+1,:) = &
[ref2-1,iperm-1,isym-1]
                                                equiv = equiv + 1
                                                nall2 = nall2 + 1
                                        end if
                                        if (all(doublet == doublet_sym)) then
                                                do indexprime = 1, 9
                                                        if (nontrivial(iperm,isym,indexprime)==1) then
                                                                do ll = 1, 9
                                                                        constrain(nconstrain+1, ll) = & 
                                                                M(iperm,isym,indexprime,ll)
                                                                end do
                                                                nconstrain = nconstrain + 1
                                                        end if
                                                end do
                                        end if
                                end do
                        end do
                        norbit(ref2) = equiv
                        allocate(constrain_reduced(max(nconstrain,9),9))
                        constrain_reduced(:,:) = 0
                        do iaux = 1, nconstrain
                                do jaux =1, 9
                                        constrain_reduced(iaux,jaux) = constrain(iaux,jaux)
                                end do
                        end do
                        call gauss_jordan_gemini(constrain_reduced, max(nconstrain,9), 9, nconstrain, kern, indep, nindep)

                        deallocate(constrain_reduced)
                        do iaux = 1, 9
                                do jaux = 1, 9
                                        kernel(ref2,iaux,jaux) = kern(iaux,jaux)
                                end do
                        end do

                        n_indep_fc(ref2) = nindep
                        do iaux = 1, nindep
                                indep_fc(ref2,iaux) = indep(iaux)-1
                        end do
                        if (nall2 == tot2) exit doii+
                end do dojj
        end do doii
end subroutine


end module symmetries