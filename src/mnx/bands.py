from .structure import Structure
import mnx.utils.cell as _cell

import mnx.FModules

import matplotlib.pyplot as plt
import numpy as np
import os
import spglib
import ase.io

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Voronoi, Delaunay

from scipy.special import erfc


from skimage.measure import marching_cubes, find_contours
import pyvista as pv

class Bands():
    def __init__(self, mod, format, folder="./", upsample=[1,1,1], magmoms=0):

        self.format, self.mod, self.upsample, self.magmoms = format, mod, upsample, magmoms

        self.Nk, self.Nbands, self.Efermi, self.Enk, self.Gnk, self.DDk, self.kpoint, self.ispin, self.structure = _read_nscf(folder, format=self.format)

        self.colors = [[
    "#979797", "#dfb843", "#770000", "#FFFC48", "#90BE6D", "#43AA8B",
    "#4D908E", "#577590", "#495867", "#9D4EDD", "#F72585", "#6A040F",
    "#8D6E63", "#6C757D", "#FFD166", "#06D6A0", "#118AB2", "#073B4C",
    "#FF6B6B", "#FF8E72", "#FFE66D", "#3D348B", "#7678ED", "#F15BB5",
    "#9B5DE5", "#00BBF9", "#00F5D4", "#FEE440", "#FF5D8F", "#6A0572",
    "#3A0CA3", "#4361EE", "#4CC9F0", "#720026", "#D81159", "#8F2D56",
    "#218380", "#73D2DE", "#F46036", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#2B2D42", "#8D99AE", "#EDF2F4", "#EF233C"],
    [
    "#56B3FF","#FF5555", "#003B94", "#AA0000", "#90BE6D", "#43AA8B",
    "#4D908E", "#577590", "#495867", "#9D4EDD", "#F72585", "#6A040F",
    "#8D6E63", "#6C757D", "#FFD166", "#06D6A0", "#118AB2", "#073B4C",
    "#FF6B6B", "#FF8E72", "#FFE66D", "#3D348B", "#7678ED", "#F15BB5",
    "#9B5DE5", "#00BBF9", "#00F5D4", "#FEE440", "#FF5D8F", "#6A0572",
    "#3A0CA3", "#4361EE", "#4CC9F0", "#720026", "#D81159", "#8F2D56",
    "#218380", "#73D2DE", "#F46036", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#2B2D42", "#8D99AE", "#EDF2F4", "#EF233C"]
]
        

    def build_setup(self, filter = True, Ecut=0):
        if np.all(self.magmoms == 0):
            self.magmoms = np.zeros(self.structure.Natoms)
        # Using spglib, find the mapping to the calculated k-grid.
        self.bigEnk, self.bigk = complete_grid(self.structure, self.Enk, self.kpoint, self.mod, self.magmoms)
        if filter:
            self.fbigEnk = filter_bands(self.bigEnk, self.bigk, Ecut, self.mod, self.ispin, delta=0)
        else:
            self.fbigEnk = self.bigEnk
        # Now, using fourier interpolation, we generate a Gamma centered
        # periodic sampling of the brillouin zone.
        self.fhugeEnk, self.hugek = denser_grid(self.bigk, self.fbigEnk, Ecut, self.mod, self.upsample, self.ispin)
        
    def visualize_surface(self, plot=True):
        meshes = np.empty([self.ispin, self.fbigEnk.shape[2]], dtype=object)
        do_plot = np.empty([self.ispin, self.fbigEnk.shape[2]], dtype=bool)
        for s in range(self.ispin):
            for n in range(self.fhugeEnk.shape[1]):
                try:
                    verts, faces, _, _ = marching_cubes(self.fhugeEnk[s,n], level=0.0, spacing = (1/self.upsample[0],1/self.upsample[1],1/self.upsample[2]))
                    faces_pv = np.empty(faces.shape[0] * 4, dtype=np.int64)
                    faces_pv[0::4] = 3
                    faces_pv[1::4] = faces[:,0]
                    faces_pv[2::4] = faces[:,1]
                    faces_pv[3::4] = faces[:,2]
                    verts /= self.mod
                    mesh = pv.PolyData(_cell.cryst2cart(verts, self.structure.rcell), faces_pv)
                    mesh.points = mesh.points - self.structure.rcell[0,:] - self.structure.rcell[1,:]- self.structure.rcell[2,:]
                    meshes[s,n] = plot_fs_in_bz_voronoi(mesh, self.structure.rcell)
                    do_plot[s,n] = True
                except ValueError:
                    do_plot[s,n] = False
        bz_wire = build_bz_wireframe(self.structure.rcell)
        if plot:
            pv.global_theme.jupyter_backend = "trame"
            p = pv.Plotter()
            for s in range(self.ispin):
                for mi, mesh in enumerate(meshes[s]):
                    if do_plot[s,mi]:
                        p.add_mesh(mesh, color=self.colors[s][mi], specular=0.5, specular_power=20, diffuse=0.8, ambient=0.3)
            p.add_mesh(bz_wire, color="black", style="wireframe", line_width=2)
            p.show()
        return meshes, bz_wire

    def visualize_fp(self, ax, axis="z", k=0, plot=True):
        for s in range(self.ispin):
            for n in range(self.fhugeEnk.shape[1]):
                if axis == "a":
                    contours = find_contours(self.fhugeEnk[s,n,int(np.round(self.mod[0]*self.upsample[0]+self.mod[0]*self.upsample[0]*k, decimals=0)),:,:], level=0)
                    new_contours = []
                    for ci,c in enumerate(contours):
                        contours3 = np.empty([len(c),3])
                        for yzi, yz in enumerate(c):
                            contours3[yzi,:] = _cell.cryst2cart(np.array([k,yz[0]/(self.mod[1]*self.upsample[1])-1,yz[1]/(self.mod[2]*self.upsample[2])-1]), self.structure.rcell)
                        segments = split_contour_by_bz(contours3, rcell = self.structure.rcell)
                        new_contours.extend(segments)
                elif axis == "b":
                    contours = find_contours(self.fhugeEnk[s,n,:,int(np.round(self.mod[1]*self.upsample[1]+self.mod[1]*self.upsample[1]*k, decimals=0)),:], level=0)
                    new_contours = []
                    for ci,c in enumerate(contours):
                        contours3 = np.empty([len(c),3])
                        for xzi, xz in enumerate(c):
                            contours3[xzi,:] = _cell.cryst2cart(np.array([xz[0]/(self.mod[0]*self.upsample[0])-1,k,xz[1]/(self.mod[2]*self.upsample[2])-1]), self.structure.rcell)
                        segments = split_contour_by_bz(contours3, rcell = self.structure.rcell)
                        new_contours.extend(segments)
                elif axis == "c":
                    contours = find_contours(self.fhugeEnk[s,n,:,:,int(np.round(self.mod[2]*self.upsample[2]+self.mod[2]*self.upsample[2]*k, decimals=0))], level=0)
                    new_contours = []
                    for ci,c in enumerate(contours):
                        contours3 = np.empty([len(c),3])
                        for xyi, xy in enumerate(c):
                            contours3[xyi,:] = _cell.cryst2cart(np.array([xy[0]/(self.mod[0]*self.upsample[0])-1,xy[1]/(self.mod[1]*self.upsample[1])-1,k]), self.structure.rcell)
                        segments = split_contour_by_bz(contours3, rcell = self.structure.rcell)
                        new_contours.extend(segments)
                elif axis == "ab":
                    contours = find_contours(np.array([self.fhugeEnk[s,n,i, i, :] for i in range(self.mod[0]*self.upsample[0]*2)]),level=0)
                    new_contours = []
                    for ci,c in enumerate(contours):
                        contours3 = np.empty([len(c),3])
                        for xzi, xz in enumerate(c):
                            contours3[xzi,:] = _cell.cryst2cart(np.array([xz[0]/(self.mod[0]*self.upsample[0])-1,xz[0]/(self.mod[0]*self.upsample[0])-1,xz[1]/(self.mod[2]*self.upsample[2])-1]), self.structure.rcell)
                        segments = split_contour_by_bz(contours3, rcell = self.structure.rcell)
                        new_contours.extend(segments)
                elif axis == "a-b":
                    contours = find_contours(np.array([self.fhugeEnk[s,n,i, self.mod[0]*self.upsample[0]*2-i-1  , :] for i in range(self.mod[0]*self.upsample[0]*2)]),level=0)
                    new_contours = []
                    for ci,c in enumerate(contours):
                        contours3 = np.empty([len(c),3])
                        for xzi, xz in enumerate(c):
                            contours3[xzi,:] = _cell.cryst2cart(np.array([xz[0]/(self.mod[0]*self.upsample[0])-1,-xz[0]/(self.mod[0]*self.upsample[0])+1,xz[1]/(self.mod[2]*self.upsample[2])-1]), self.structure.rcell)
                        segments = split_contour_by_bz(contours3, rcell = self.structure.rcell)
                        new_contours.extend(segments)
                elif axis == "2ab":
                    contours = find_contours(np.array([self.fhugeEnk[s,n,2*i,int(self.mod[0]*self.upsample[0]*3/4)-i-1,:] for i in range(self.mod[0]*self.upsample[0])]),level=0)
                    new_contours = []
                    for ci,c in enumerate(contours):
                        contours3 = np.empty([len(c),3])
                        for xzi, xz in enumerate(c):
                            contours3[xzi,:] = _cell.cryst2cart(np.array([xz[0]/(self.mod[0]*self.upsample[0])*2-1,-xz[0]/(self.mod[0]*self.upsample[0])/2+1,xz[1]/(self.mod[2]*self.upsample[2])-1]), self.structure.rcell)
                        segments = split_contour_by_bz(contours3, rcell = self.structure.rcell)
                        new_contours.extend(segments)
                contours = new_contours
                if n == 0:
                    contours_list = [contours]
                else:
                    contours_list.append(contours)
            if plot:
                for n,contour in enumerate(contours_list):
                    for c in contour:
                        ax.plot(c[:,0], c[:,1], color=self.colors[s][n], alpha=0.5)
        
        if plot:
            bz_mesh = build_bz_mesh(self.structure.rcell)  # reuse your BZ generator
            bz_points = bz_mesh.points
            bz_faces = bz_mesh.faces.reshape(-1, 4)[:, 1:]

            z0 = 0.0
            for f in bz_faces:
                tri = bz_points[f]

                # Separate vertices by z position
                z = tri[:, 2]
                above = z > z0
                below = z < z0

                if np.any(above) and np.any(below):
                    # Plane cuts this triangle → find intersection points
                    pts = []
                    for i in range(3):
                        j = (i + 1) % 3
                        z1, z2 = z[i], z[j]
                        if (z1 - z0) * (z2 - z0) <= 0:  # Edge crosses plane
                            t = (z0 - z1) / (z2 - z1 + 1e-16)
                            pt = tri[i] + t * (tri[j] - tri[i])
                            pts.append(pt)

                    if len(pts) == 2:
                        xy = np.array(pts)[:, :2]
                        ax.plot(xy[:, 0], xy[:, 1], color='black', linewidth=1.0)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(r'$k_x (2\pi/{\AA})$', fontsize=12)
            ax.set_ylabel(r'$k_y (2\pi/{\AA})$', fontsize=12)
            ax.tick_params(labelsize=12)
        return contours_list
        
        
    def visualize_diffuse_fp(self, f1, kmax=0.3, sigma=5e-3):
        diffuse_fermi = np.average(smear_bands(self.fhugeEnk, sigma=sigma), axis=0)
        shape = (self.mod[0]*2*self.upsample[0], self.mod[1]*2*self.upsample[1], self.mod[2]*2*self.upsample[2])
        f_fft = fourier_interpolator(self.hugek, diffuse_fermi, shape=shape)
        kxs, kys = np.arange(-kmax,kmax,2e-3), np.arange(-kmax,kmax,2e-3)
        X = np.empty([len(kxs), len(kys)])
        for kxi,kx in enumerate(kxs):
            for kyi,ky in enumerate(kys):
                X[kxi,kyi] = f_fft(_cell.cart2cryst(np.array([kx,ky,0])*self.mod, self.structure.rcell))
        f1.imshow(X,vmin=0)
    
    def get_gaussian_bands(self, Ecut, sigma=5e-3):
        bigEnk, bigk = denser_grid(self.bigk, self.bigEnk, Ecut, self.mod, self.upsample, self.ispin, fix_rcell=True)
        diffuse_fermis = smear_bands(bigEnk, sigma=sigma)
        return diffuse_fermis, bigk
    
    
    def get_Nsk(self, sigma=1e-2, downsample=[1,1,1], Ecut = 0):
        Nk = self.bigk.shape[0]
        Ndown = int(Nk/np.prod(downsample))

        self.Nsk = np.zeros([self.ispin,Ndown])
        self.downk = np.empty([Ndown,3])
        diffuse_fermis, hugek = self.get_gaussian_bands(Ecut=Ecut, sigma=sigma)
        hugek = np.round(hugek)

        self.Nsk, self.downk = mnx.FModules.bands.get_nesting(Ndown, self.mod, downsample, hugek, diffuse_fermis)

    def get_nesting_qpath(self, k_inpath, N, downsample):
        k_inpath = np.asarray(k_inpath)
        qpath=np.empty([len(k_inpath)*(N+1),3])
        nesting = np.zeros([self.ispin, len(qpath)])
        rq = np.zeros(len(k_inpath)*(N+1))
        xticks = [0]
        for ki,ks in enumerate(k_inpath):
            dk = (ks[1]-ks[0])
            rq[ki*(N+1):(ki+1)*(N+1)] = rq[ki*(N+1)-1]+np.linalg.norm(_cell.cryst2cart(dk, self.structure.rcell))*np.arange(0,10+10/N,10/N)
            xticks.append(rq[(ki+1)*(N+1)-1])
            dk = dk * self.mod
            qpath[ki*(N+1):(ki+1)*(N+1)] = dk[None,:]*np.arange(0,1+1/N,1/N)[:,None]+ks[0]*self.mod
        for si in range(self.ispin):
            f_fft = fourier_interpolator(self.downk, self.Nsk[si], shape=(int(self.mod[0]/downsample[0]), int(self.mod[1]/downsample[1]), int(self.mod[2]/downsample[2])), upsample=self.upsample)
            nesting[si] = f_fft(qpath/downsample)
        return nesting, rq, xticks


    def get_dos(self, N, Emin, Emax, sigma):
        Es = np.arange(Emin,Emax+(Emax-Emin)/N,(Emax-Emin)/N)
        DOS = np.zeros([self.ispin, N+1])
        for Ei,E in enumerate(Es):
            diffuse_fermis, hugek = self.get_gaussian_bands(Ecut=E, sigma=sigma)
            hugek = np.round(hugek)
            my_dict = {tuple(np.array([s,hugek[i,0],hugek[i,1],hugek[i,2]], dtype=int)): diffuse_fermis[s,i] for s in range(diffuse_fermis.shape[0]) for i in range(diffuse_fermis.shape[1])}
            for s in range(self.ispin):
                for kxi in range(int(self.mod[0])):
                    for kyi in range(int(self.mod[1])):
                        for kzi in range(int(self.mod[2])):
                            try:
                                DOS[s,Ei] += my_dict[tuple([s,kxi,kyi,kzi])]
                            except:
                                breakpoint()
        return DOS, Es
    
    def get_bands(self, k_inpath, N):
        k_inpath = np.asarray(k_inpath)
        qpath=np.empty([len(k_inpath)*(N+1),3])
        bands = np.empty([self.ispin,self.bigEnk.shape[2],len(k_inpath)*(N+1)])
        rq = np.zeros(len(k_inpath)*(N+1))
        xticks = [0]
        for ki,ks in enumerate(k_inpath):
            dk = (ks[1]-ks[0])
            rq[ki*(N+1):(ki+1)*(N+1)] = rq[ki*(N+1)-1]+np.linalg.norm(_cell.cryst2cart(dk, self.structure.rcell))*np.arange(0,10+10/N,10/N)
            xticks.append(rq[(ki+1)*(N+1)-1])
            dk = dk * self.mod
            qpath[ki*(N+1):(ki+1)*(N+1)] = dk[None,:]*np.arange(0,1+1/N,1/N)[:,None]+ks[0]*self.mod
        for s in range(self.ispin):
            for n in range(self.bigEnk.shape[2]):
                f_fft = fourier_interpolator(self.bigk, self.bigEnk[s,:,n], shape=(self.mod[0], self.mod[1], self.mod[2]), upsample=self.upsample)
                bands[s,n] = f_fft(qpath)
        return bands, rq, xticks
    
    def get_linewidth(self, k_inpath, N):
        k_inpath = np.asarray(k_inpath)
        qpath=np.empty([len(k_inpath)*(N+1),3])
        linewidth = np.empty([1,self.bigGnk.shape[2],len(k_inpath)*(N+1)])
        rq = np.zeros(len(k_inpath)*(N+1))
        xticks = [0]
        for ki,ks in enumerate(k_inpath):
            dk = (ks[1]-ks[0])
            rq[ki*(N+1):(ki+1)*(N+1)] = rq[ki*(N+1)-1]+np.linalg.norm(_cell.cryst2cart(dk, self.structure.rcell))*np.arange(0,10+10/N,10/N)
            xticks.append(rq[(ki+1)*(N+1)-1])
            dk = dk * self.mod
            qpath[ki*(N+1):(ki+1)*(N+1)] = dk[None,:]*np.arange(0,1+1/N,1/N)[:,None]+ks[0]*self.mod
        for n in range(self.bigGnk.shape[2]):
            f_fft = fourier_interpolator(self.bigk, self.bigGnk[0,:,n], shape=(self.mod[0], self.mod[1], self.mod[2]), upsample=self.upsample)
            linewidth[0,n] = f_fft(qpath)
            for qi, gamma in enumerate(linewidth[0,n]):
                if gamma < 0:
                    linewidth[0,n,qi] = np.abs(linewidth[0,n,:]).min()*np.exp(gamma)
        return linewidth, rq, xticks
    
    def get_doubledeltas(self, k_inpath, N):
        k_inpath = np.asarray(k_inpath)
        qpath=np.empty([len(k_inpath)*(N+1),3])
        dd = np.empty([1,len(qpath)])
        rq = np.zeros(len(k_inpath)*(N+1))
        xticks = [0]
        for ki,ks in enumerate(k_inpath):
            dk = (ks[1]-ks[0])
            rq[ki*(N+1):(ki+1)*(N+1)] = rq[ki*(N+1)-1]+np.linalg.norm(_cell.cryst2cart(dk, self.structure.rcell))*np.arange(0,10+10/N,10/N)
            xticks.append(rq[(ki+1)*(N+1)-1])
            dk = dk * self.mod
            qpath[ki*(N+1):(ki+1)*(N+1)] = dk[None,:]*np.arange(0,1+1/N,1/N)[:,None]+ks[0]*self.mod
        f_fft = fourier_interpolator(self.bigk, self.bigDDk[0,:], shape=(self.mod[0], self.mod[1], self.mod[2]), upsample=self.upsample)
        dd[0] = f_fft(qpath)
        for qi, d in enumerate(dd[0]):
                if d < 0:
                    dd[0,qi] = np.exp(d)
        return dd, rq, xticks


###############################################################################################
# Some day we will have to implement proper fourier interpolations of the dynamical matrixes. #
###############################################################################################
# def get_PhiR(qpoints, list_of_dyns):
#     Nq = len(qpoints)
#     Phi_R = np.empty([len(Rs), Natoms, Natoms, 3, 3])
#     for R in Rs:
#         for qi, q in qpoints:
#             dyn = list_of_dyns[qi]
#             for a in range(Natoms):
#                 for b in range(Natoms):
#                     for alpha in range(3):
#                         for beta in range(3):
#                             Phi_R[R,a,b,alpha,beta] = 1/Nq*np.sum(np.sqrt(dyn.masses[a]*dyn.masses[b])*dyn.dyn_matrixes[0,a,b,alpha,beta])*np.exp(-1j*np.matmul(q*Rs[ri]))
#     return Phi_R

# def get_DQ(qpoint, PhiR, masses):
#     DQ = np.empty([Natoms, Natoms, 3, 3])
#     for R in Rs:
#         for a in range(Natoms):
#             for b in range(Natoms):
#                 for alpha in range(3):
#                     for beta in range(3):
#                         DQ[a,b,alpha,beta] += 1/np.sqrt(masses[a]*masses[b])*PhiR[R,a,b,alpha,beta]*np.exp(1j*np.matmul(qpoint,R))
#     return DQ

    

def fourier_interpolator(kpoints, energies, shape, upsample=(1,1,1)):
    """
    Fourier-based interpolation for band energies on a uniform 3D k-grid.

    Parameters
    ----------
    kpoints : (Nk,3) array
        k-points (assumed uniform grid, ordered consistently with `shape`)
    energies : (Nk,) array
        Energies at each k-point
    shape : tuple of ints (Nx, Ny, Nz)
        Grid dimensions of the uniform mesh
    upsample : tuple (ux, uy, uz)
        Factor to upsample in each direction

    Returns
    -------
    f : callable
        Interpolator f(K) with K (...,3) returning interpolated energies
    """

    
    Nx, Ny, Nz = shape
    # Reshape energies into 3D grid
    E = energies.reshape(Nx, Ny, Nz)

    # Extract unique grid axes
    kx = np.unique(kpoints[:,0])
    ky = np.unique(kpoints[:,1])
    kz = np.unique(kpoints[:,2])

    # --- Fourier upsampling (pad in reciprocal space, not real space) ---
    F = np.fft.fftn(E)   # to Fourier domain
    F_pad = np.zeros((Nx*upsample[0],
                      Ny*upsample[1],
                      Nz*upsample[2]), dtype=complex)

    # Centered placement of Fourier coefficients
    F_pad[:Nx//2, :Ny//2, :Nz//2] = F[:Nx//2, :Ny//2, :Nz//2]
    F_pad[-(Nx-Nx//2):, :Ny//2, :Nz//2] = F[-(Nx-Nx//2):, :Ny//2, :Nz//2]
    F_pad[:Nx//2, -(Ny-Ny//2):, :Nz//2] = F[:Nx//2, -(Ny-Ny//2):, :Nz//2]
    F_pad[:Nx//2, :Ny//2, -(Nz-Nz//2):] = F[:Nx//2, :Ny//2, -(Nz-Nz//2):]
    F_pad[-(Nx-Nx//2):, -(Ny-Ny//2):, :Nz//2] = F[-(Nx-Nx//2):, -(Ny-Ny//2):, :Nz//2]
    F_pad[-(Nx-Nx//2):, :Ny//2, -(Nz-Nz//2):] = F[-(Nx-Nx//2):, :Ny//2, -(Nz-Nz//2):]
    F_pad[:Nx//2, -(Ny-Ny//2):, -(Nz-Nz//2):] = F[:Nx//2, -(Ny-Ny//2):, -(Nz-Nz//2):]
    F_pad[-(Nx-Nx//2):, -(Ny-Ny//2):, -(Nz-Nz//2):] = F[-(Nx-Nx//2):, -(Ny-Ny//2):, -(Nz-Nz//2):]

    # Back to k-space on finer mesh
    E_hi = np.fft.ifftn(F_pad).real * np.prod(upsample)

    # Build refined grids
    kx_hi = np.linspace(kx[0], kx[-1]+1, Nx*upsample[0], endpoint=False)
    ky_hi = np.linspace(ky[0], ky[-1]+1, Ny*upsample[1], endpoint=False)
    kz_hi = np.linspace(kz[0], kz[-1]+1, Nz*upsample[2], endpoint=False)

    interp_hi = RegularGridInterpolator((kx_hi, ky_hi, kz_hi), E_hi,
                                        method='linear',
                                        bounds_error=False, fill_value=None)

    Lx, Ly, Lz = kx[-1]-kx[0]+(kx[1]-kx[0]), ky[-1]-ky[0]+(ky[1]-ky[0]), kz[-1]-kz[0]+(kz[1]-kz[0])
    def f(points):
        pts = np.atleast_2d(points).astype(float).copy()
        # wrap into first BZ
        pts[:,0] = (pts[:,0]-kx[0]) % Lx + kx[0]
        pts[:,1] = (pts[:,1]-ky[0]) % Ly + ky[0]
        pts[:,2] = (pts[:,2]-kz[0]) % Lz + kz[0]
        return interp_hi(pts)

    return f


def denser_grid(big_kpoint, big_Enk, Ecut, mod, upsample, ispin, fix_rcell=False):
    tmp_mod = np.empty(len(mod), dtype=int)
    if fix_rcell:
        tmp_mod[0], tmp_mod[1], tmp_mod[2] = mod[0]*upsample[0], mod[1]*upsample[1], mod[2]*upsample[2]
        dx = (big_kpoint[:,0].max()+1)/(tmp_mod[0])
        dy = (big_kpoint[:,1].max()+1)/(tmp_mod[1])
        dz = (big_kpoint[:,2].max()+1)/(tmp_mod[2])
        kxs = np.arange(0, big_kpoint[:,0].max()+1, dx)
        kys = np.arange(0, big_kpoint[:,1].max()+1, dy)
        kzs = np.arange(0, big_kpoint[:,2].max()+1, dz)
    else:
        tmp_mod[0], tmp_mod[1], tmp_mod[2] = mod[0]*2*upsample[0], mod[1]*2*upsample[1], mod[2]*2*upsample[2]
        dx = (2*(big_kpoint[:,0].max()+1))/(tmp_mod[0])
        dy = (2*(big_kpoint[:,1].max()+1))/(tmp_mod[1])
        dz = (2*(big_kpoint[:,2].max()+1))/(tmp_mod[2])
        kxs = np.arange((-1)*big_kpoint[:,0].max()-1, big_kpoint[:,0].max()+1, dx)
        kys = np.arange((-1)*big_kpoint[:,1].max()-1, big_kpoint[:,1].max()+1, dy)
        kzs = np.arange((-1)*big_kpoint[:,2].max()-1, big_kpoint[:,2].max()+1, dz)
    if ispin == 1:
        huge_Enk = np.empty([1, big_Enk.shape[2], tmp_mod[0], tmp_mod[1], tmp_mod[2]])
    else:
        huge_Enk = np.empty([2, big_Enk.shape[2], tmp_mod[0], tmp_mod[1], tmp_mod[2]])

    KX, KY, KZ = np.meshgrid(kxs, kys, kzs, indexing='ij')
    points = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=-1)

    for s in range(ispin):
        for n in range(big_Enk.shape[2]):
            f_fft = fourier_interpolator(big_kpoint, big_Enk[s,:,n]-Ecut, shape=(mod[0], mod[1], mod[2]), upsample=upsample)
            huge_Enk[s,n] = f_fft(points).reshape(len(kxs), len(kys), len(kzs))
    return huge_Enk, points


def complete_grid(structure, Enk, kpoint, mod, magmoms=0, symprec=1e-5):
    cell = (structure.cell, _cell.cart2cryst(structure.atom_coords, structure.cell), structure.atom_species[:,1])
    mapping, address, trs = get_ir_reciprocal_mesh_magnetic(mod, cell, magmoms, kpoint, symprec)
    kgrid = np.array([[i, j, k]
                    for i in range(mod[0])
                    for j in range(mod[1])
                    for k in range(mod[2])], dtype=int)
    big_Enk = np.empty([Enk.shape[0], len(kgrid), Enk.shape[2]])
    for ki, k in enumerate(kgrid):
        if trs[ki] == 0:
            big_Enk[0,ki,:] = Enk[1, mapping[ki], :]
            big_Enk[1,ki,:] = Enk[0, mapping[ki], :]
        else:
            big_Enk[:,ki,:] = Enk[:, mapping[ki], :]
    return big_Enk, kgrid


def filter_bands(big_Enk, kpoints, Efermi, mod, ispin, delta = 0):
    """In case of a colinear calcuation, the spins must be aligned with the
    c axis."""
    fermi_bands = []
    for s in range(ispin):
        for i in range(big_Enk.shape[2]):
            if np.any(big_Enk[s,:,i]<=Efermi+delta) and np.any(big_Enk[s,:,i]>=Efermi-delta):
                fermi_bands.append((s,i))
    if ispin == 1:
        fermi_Enk = np.empty([1, big_Enk.shape[1],len(fermi_bands)])
        for ii,si in enumerate(fermi_bands):
            fermi_Enk[0,:,ii] = big_Enk[si[0],:,si[1]]
    else:
        fermi_Enk = np.empty([2, big_Enk.shape[1], len(np.unique(np.array(fermi_bands)[:,1]))])
        for si in np.unique(np.array(fermi_bands)[:,0]):
            for ii, ni in enumerate(np.unique(np.array(fermi_bands)[:,1])):
                fermi_Enk[si,:,ii] = big_Enk[si,:,ni]
    return(fermi_Enk)

def gaussian(x, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*(x**2/sigma**2))

def smear_bands(Enk, sigma=5e-3):
    diffuse_fermi = np.zeros([Enk.shape[0],Enk.shape[2], Enk.shape[3], Enk.shape[4]])
    for s in  range(Enk.shape[0]): 
        for n in range(Enk.shape[1]):
            diffuse_fermi[s] += gaussian(Enk[s,n,:,:,:], sigma=sigma)
    if Enk.shape[0] == 1:
        return np.array([diffuse_fermi[0].ravel("C")])
    else:
        return np.array([diffuse_fermi[0].ravel("C"), diffuse_fermi[1].ravel("C")])


def compute_bz_vertices(rcell, nmax=2):
    """
    Compute vertices of the first Brillouin zone using Voronoi/Wigner-Seitz around origin.
    """
    rcell = np.array(rcell)
    hs = np.arange(-nmax, nmax+1)
    grid = np.array(np.meshgrid(hs, hs, hs)).T.reshape(-1,3)
    points = grid @ rcell
    origin_idx = np.where((grid==[0,0,0]).all(axis=1))[0][0]

    vor = Voronoi(points)
    bz_vertices = []
    for (p,q), rverts in zip(vor.ridge_points, vor.ridge_vertices):
        if origin_idx not in (p,q):
            continue
        if any(v < 0 for v in rverts):
            continue
        bz_vertices.extend(vor.vertices[rverts])
    bz_vertices = np.unique(np.array(bz_vertices), axis=0)
    return bz_vertices

# ---------------------------
# Filter FS mesh using BZ convex hull
# ---------------------------
def filter_fs_inside_bz_voronoi(fs, rcell, nmax=2, tol=1e-8, cut=True):
    """
    Remove triangles of FS mesh if any vertex is outside the first BZ.
    """
    # Compute BZ vertices and convex hull
    bz_vertices = compute_bz_vertices(rcell, nmax=nmax)
    hull = Delaunay(bz_vertices)

    # Check which points are inside BZ
    points = fs.points
    inside_mask = hull.find_simplex(points) >= 0
    positive_mask = np.all(points>=0, axis=1)

    outside_indices = np.where(~inside_mask)[0]
    positive_indices = np.where(~positive_mask)[0]

    # Filter faces: remove any triangle with at least one vertex outside
    faces = fs.faces.reshape(-1, 4)
    keep_faces = []
    for f in faces:
        vidxs = f[1:]
        if not np.any(np.isin(vidxs, outside_indices)):
            if cut:
                if np.any(np.isin(vidxs, positive_indices)):
                    keep_faces.append(f)
            else:
                keep_faces.append(f)
    if len(keep_faces) == 0:
        print("Warning: no faces remain inside first BZ!")
        return None

    new_faces = np.array(keep_faces).ravel()
    filtered_fs = pv.PolyData(points, new_faces)
    filtered_fs = filtered_fs.triangulate().clean(tolerance=tol)
    if filtered_fs.n_faces_strict > 0:
        filtered_fs = filtered_fs.compute_normals(auto_orient_normals=True, feature_angle=150)
    return filtered_fs

# ---------------------------
# Build BZ wireframe mesh
# ---------------------------
def build_bz_mesh(rcell, nmax=2):
    """
    Build wireframe mesh for first BZ (Voronoi around origin)
    """
    bz_vertices = compute_bz_vertices(rcell, nmax=nmax)
    hull = Delaunay(bz_vertices)
    # Extract convex hull simplices for wireframe
    faces = []
    for simplex in hull.convex_hull:
        faces.append([3, *simplex])
    flat_faces = np.array(faces).ravel()
    bz_mesh = pv.PolyData(bz_vertices, flat_faces).triangulate().extract_surface().clean()
    return bz_mesh

# ---------------------------
# Plot FS inside BZ
# ---------------------------
def plot_fs_in_bz_voronoi(fs_mesh, rcell, nmax=2, fs_color="orange"):
    filtered_fs = filter_fs_inside_bz_voronoi(fs_mesh, rcell, nmax=nmax)
    return filtered_fs

def build_bz_wireframe(rcell, nmax=2):
    """
    Build a clean wireframe-only Brillouin Zone mesh for visualization.
    
    Parameters
    ----------
    rcell : (3,3) array
        Reciprocal lattice vectors (rows = b1, b2, b3)
    nmax : int
        Range of reciprocal lattice points to generate
    
    Returns
    -------
    bz_edges : pv.PolyData
        Wireframe representation of the BZ
    """
    rcell = np.array(rcell)
    hs = np.arange(-nmax, nmax+1)
    grid = np.array(np.meshgrid(hs, hs, hs)).T.reshape(-1,3)
    points = grid @ rcell
    origin_idx = np.where((grid == [0,0,0]).all(axis=1))[0][0]

    # Build Voronoi around reciprocal lattice points
    vor = Voronoi(points)
    edge_lines = []

    # Collect edges (ridge lines) connected to the origin
    for (p, q), verts in zip(vor.ridge_points, vor.ridge_vertices):
        if origin_idx not in (p, q):
            continue
        if any(v < 0 for v in verts):  # skip infinite ridges
            continue
        verts = np.array(verts)
        for i in range(len(verts)):
            v0, v1 = verts[i], verts[(i+1) % len(verts)]
            edge_lines.append([2, v0, v1])  # polyline with 2 points

    vertices = vor.vertices
    faces = np.array(edge_lines).ravel()
    bz_edges = pv.PolyData(vertices, faces)
    return bz_edges

def is_point_inside_bz(k, rcell, tol=1e-8):
    """
    Check if point k is inside the first Brillouin Zone using Wigner–Seitz construction.
    """
    k = np.array(k, dtype=float)
    rcell = np.array(rcell, dtype=float)

    # Generate nearby reciprocal lattice points
    nmax = 1  # ±1 is enough for 1st BZ
    G_points = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for l in range(-nmax, nmax + 1):
                if i == 0 and j == 0 and l == 0:
                    continue
                G_points.append(i * rcell[0] + j * rcell[1] + l * rcell[2])
    G_points = np.array(G_points)

    # Distance of k to origin
    k_norm = np.linalg.norm(k)

    # Compare distance to all other G points
    for G in G_points:
        if np.linalg.norm(k - G) < k_norm - tol:
            return False  # closer to another G → outside

    return True


def split_contour_by_bz(contour, rcell):
    """
    Split a contour into smaller ones, keeping only the points inside the BZ.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (N, 3) representing one contour polyline.
    rcell : np.ndarray
        3x3 reciprocal lattice vectors.

    Returns
    -------
    inside_segments : list of np.ndarray
        List of new contour segments fully inside the BZ.
    """
    inside_segments = []
    current_segment = []

    for point in contour:
        if is_point_inside_bz([point[0], point[1], point[2]], rcell):
            current_segment.append(point)
        else:
            # If leaving BZ, save current segment if it has enough points
            if len(current_segment) > 1:
                inside_segments.append(np.array(current_segment))
            current_segment = []  # reset segment

    # Save any leftover segment
    if len(current_segment) > 1:
        inside_segments.append(np.array(current_segment))

    return inside_segments

def split_by_mask(mesh, mask):
    """
    Split a contour into smaller ones, keeping only the points inside the BZ.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (N, 3) representing one contour polyline.
    rcell : np.ndarray
        3x3 reciprocal lattice vectors.

    Returns
    -------
    inside_segments : list of np.ndarray
        List of new contour segments fully inside the BZ.
    """
    inside_segments = []
    current_segment = []

    for pi, point in enumerate(mesh.points):
        if mask[pi]:
            current_segment.append(point)
        else:
            # If leaving BZ, save current segment if it has enough points
            if len(current_segment) > 1:
                inside_segments.append(np.array(current_segment))
            current_segment = []  # reset segment

    # Save any leftover segment
    if len(current_segment) > 1:
        inside_segments.append(np.array(current_segment))
    
    inside_meshes = []
    for segment in inside_segments:
        mesh = pv.PolyData()
        mesh.points = np.array(segment)
        n = len(segment)
        mesh.lines = np.hstack([[n], np.arange(n)])
        inside_meshes.append(mesh)

    return inside_meshes

def inrcell(k, mod):
    for i in range(3):
        if k[i]<0 and k[i]%mod[i]!=0:
            k[i]+=(np.abs(k[i])//mod[i]+1)*mod[i]
        elif k[i]<0 and k[i]%mod[i]==0:
            k[i]+=(np.abs(k[i])//mod[i])*mod[i]
        elif k[i]>=mod[i]:
            k[i]+=-np.abs((k[i]//mod[i])*mod[i])
    return np.round(k)


def get_ir_reciprocal_mesh_magnetic(mod, cell, magmoms, kpoint, symprec=1e-5):
    """
    """
    tmp_kpoint = np.copy(kpoint)
    tmp_kpoint[:,0]=kpoint[:,0]*mod[0]
    tmp_kpoint[:,1]=kpoint[:,1]*mod[1]
    tmp_kpoint[:,2]=kpoint[:,2]*mod[2]
    tmp_kpoint = np.round(tmp_kpoint)
    # Build magnetic cell tuple
    cell0 = (cell[0], cell[1], cell[2], magmoms)
    mag_dataset = spglib.get_magnetic_symmetry_dataset(cell0, symprec=symprec)

    rotations = mag_dataset['rotations']
    if np.all(magmoms==0):
        print("Non magnetic calculation.")
        time_reversals = np.ones(len(rotations))
    else:
        time_reversals = mag_dataset['time_reversals']  # 0 = TRS, 1 = normal

    kgrid = np.array([[i, j, k]
                            for i in range(mod[0])
                            for j in range(mod[1])
                            for k in range(mod[2])], dtype=int)

    mapping = -np.ones(len(kgrid), dtype=int)
    trs_map = np.ones(len(kgrid), dtype=int)
    address_dict = {tuple(inrcell(tmp_kpoint[i], mod)):i for i in range(len(tmp_kpoint))}

    rkg_list = []
    for kgi, kg in enumerate(kgrid):
        tmp_kg = kg
        address_dict[kgi] = kg
        is_found = False
        for ri, R in enumerate(rotations):
            try:
                if time_reversals[ri] == 0:
                    rkg = tuple(inrcell(np.matmul(-np.transpose(R), tmp_kg), mod))
                else:
                    rkg = tuple(inrcell(np.matmul(np.transpose(R), tmp_kg), mod))
                mapping[kgi] = address_dict[rkg]
                trs_map[kgi] = time_reversals[ri]
                is_found = True
                break
            except KeyError:
                None
        if not is_found:
            rkg_list.append(kg)
            print(f"{kgi}:{kg}-->{tmp_kg}-->{rkg}-->X")
    return(mapping, address_dict, trs_map)

def mp(x):
    return 0.5 * erfc(x) + (x/np.sqrt(np.pi)) * np.exp(-x*x)

def nesting_k(qpath,Enk,hugek,n,mmin,mmax,smearing,degauss,mod,downsample):
    delta = 1e-12
    ispin = Enk.shape[0]
    value = np.zeros([ispin,int(mod[0]/downsample[0]),int(mod[1]/downsample[1]),int(mod[2]/downsample[2])])
    downk = np.empty([int(hugek.shape[0]/(downsample[0]*downsample[1]*downsample[2])),3])
    nesting = np.empty([ispin,len(qpath)])
    if smearing == "mp":
        for s in range(ispin):
            for m in range(mmin,mmax+1):
                for qxi in range(int(mod[0]/downsample[0])):
                    for qyi in range(int(mod[1]/downsample[1])):
                        for qzi in range(int(mod[2]/downsample[2])):
                            for kxi in range(int(mod[0]/downsample[0])):
                                for kyi in range(int(mod[1]/downsample[1])):
                                    for kzi in range(int(mod[2]/downsample[2])):
                                        ki = kxi*downsample[0]*mod[1]*mod[2]+kyi*downsample[1]*mod[2]+kzi*downsample[2]
                                        kxii, kyii, kzii = kxi*downsample[0],kyi*downsample[1],kzi*downsample[2]
                                        qxii, qyii, qzii = qxi*downsample[0],qyi*downsample[1],qzi*downsample[2]
                                        value[s,kxi,kyi,kzi] += (mp(Enk[s,n,kxii,kyii,kzii]/degauss) - mp(Enk[s,m,qxii,qyii,qzii]/degauss))/(Enk[s,n,kxii,kyii,kzii]-Enk[s,m,qxii,qyii,qzii]+1j*delta)
                                        ski = int(kxi*mod[1]*mod[2]/(downsample[2]*downsample[1])+kyi*mod[2]/downsample[2]+kzi)
                                        downk[ski] = hugek[ki]/downsample
        f_fft = fourier_interpolator(downk, value[s], shape=(int(mod[0]/downsample[0]), int(mod[1]/downsample[1]), int(mod[2]/downsample[2])), upsample=[2,2,2])
        nesting[s] = f_fft(qpath/np.array(downsample))
    return nesting

def get_qpath(k_inpath, N, structure, mod):
    k_inpath = np.asarray(k_inpath)
    qpath=np.empty([len(k_inpath)*(N+1),3])
    rq = np.zeros(len(k_inpath)*(N+1))
    xticks = [0]
    for ki,ks in enumerate(k_inpath):
        dk = (ks[1]-ks[0])
        rq[ki*(N+1):(ki+1)*(N+1)] = rq[ki*(N+1)-1]+np.linalg.norm(_cell.cryst2cart(dk, structure.rcell))*np.arange(0,10+10/N,10/N)
        xticks.append(rq[(ki+1)*(N+1)-1])
        dk = dk * mod
        qpath[ki*(N+1):(ki+1)*(N+1)] = dk[None,:]*np.arange(0,1+1/N,1/N)[:,None]+ks[0]*mod
    return qpath


def _read_nscf(folder, format):
    if format=="qe":
        info = os.popen(f"grep \"number of k points=\" {folder}/RUN.nscf.out").read()
        Nk = int(info.split()[4])
        kpoint=np.empty([Nk, 3])
        info = os.popen(f"grep \"the Fermi energy is\" {folder}/RUN.scf.out").read()
        try:
            Efermi = float(info.split()[4])
        except:
            info = os.popen(f"grep Fermi {folder}/RUN.nscf.out").read()
            print("The Fermi energy was read from the nscf calculation")
            Efermi = float(info.split()[4])
        info = os.popen(f"grep \"SPIN UP\" {folder}/RUN.nscf.out").read()
        try:
            ispin = 1
            if len(info.split())!=0:
                print("ispin = 2")
                ispin = 2
        except:
            ispin = 1
        #SOC not implemented
        with open(f"{folder}/RUN.nscf.out", "r") as f:
            fileread = f.readlines()
            i, j  = 0, 2
            for li, line in enumerate(fileread):
                if line[:36] == "                       cryst. coord.":
                    i+=1
                    for ki in range(Nk):
                        info = fileread[li+i+ki].translate(str.maketrans("","","(),"))
                        kpoint[ki,:] = np.array(info.split()[-6:-3])
                    if np.any(kpoint==-0.5):
                        indices = np.array(np.where(kpoint==-0.5), dtype=int)
                        for ind in np.transpose(indices[:]):
                            kpoint[ind[0], ind[1]]=0.5
                i=0
                if line[:38] == "     End of band structure calculation":
                    i+=2
                    if ispin == 1:
                        Nbands = 0
                        while len(fileread[li+i+j].split())!=0:
                            Nbands += int(len(fileread[li+i+j].split()))
                            j+=1
                        Enk = np.empty([1, Nk, Nbands])
                        for ki in range(0, Nk):
                            tmp_Enk = []
                            i+=2
                            while len(fileread[li+i].split())!=0:
                                for element in fileread[li+i].split():
                                    tmp_Enk.append(element)
                                i+=1
                            if len(tmp_Enk)!=Nbands:
                                print("The number of bands in not consistent.")
                                raise ValueError
                            Enk[0,ki,:] = tmp_Enk
                            i+=1
                            i+=j

                    elif ispin == 2:
                        i+=3
                        Nbands = 0
                        while len(fileread[li+i+j].split())!=0:
                            Nbands += int(len(fileread[li+i+j].split()))
                            j+=1
                        Enk = np.empty([2, Nk, Nbands])
                        for ki in range(0, Nk):
                            tmp_Enk = []
                            i+=2
                            while len(fileread[li+i].split())!=0:
                                for element in fileread[li+i].split():
                                    tmp_Enk.append(element)
                                i+=1
                            if len(tmp_Enk)!=Nbands:
                                print("The number of bands in not consistent.")
                                raise ValueError
                            Enk[0,ki,:] = tmp_Enk
                            i+=j+1
                        # Now spin down.
                        i+=3
                        for ki in range(0, Nk):
                            tmp_Enk = []
                            i+=2
                            while len(fileread[li+i+j].split())!=0:
                                for element in fileread[li+i].split():
                                    tmp_Enk.append(element)
                                i+=1
                            if len(tmp_Enk)!=Nbands:
                                print("The number of bands in not consistent.")
                                raise ValueError
                            Enk[1,ki,:] = tmp_Enk
                            i+=j+1
                    break
            atom = ase.io.read(f"{folder}/RUN.nscf.in", format="espresso-in")
            structure = Structure.from_ASE(atom)

    elif format=="vasp":
        # NOT TESTED !!!!!!!
        info = os.popen(f"grep NKPTS {folder}/OUTCAR").read()
        Nk, Nbands = int(info.split()[3]), int(info.split()[-1])
        info = os.popen(f"grep E-fermi {folder}/OUTCAR.scf").read()
        Efermi = float(info.split()[2])
        ispin = int(os.popen(f"grep ISPIN {folder}/OUTCAR").read().split()[2])

        with open(f"{folder}/OUTCAR", "r") as f:
            fileread = f.readlines()
            for li, line in enumerate(fileread):
                if line[1:8] == "E-fermi":
                    kpoint = np.zeros([Nk,3])
                    if ispin == 1:
                        Enk = np.zeros([ispin, Nk, Nbands])
                        k = li+2
                        for ik in range(Nk):
                            k+=2
                            kpoint[ik] = [float(fileread[k].split()[-3]), float(fileread[k].split()[-2]), float(fileread[k].split()[-1])]
                            k+=1
                            for ib in range(Nbands):
                                k+=1
                                Enk[0,ik,ib] = float(fileread[k].split()[1])
                    elif ispin == 2:
                        Enk = np.zeros([ispin, Nk, Nbands])
                        k = li+2
                        for ispin in range(2):
                            for ik in range(Nk):
                                k+=2
                                kpoint[ik] = [float(fileread[k].split()[-3]), float(fileread[k].split()[-2]), float(fileread[k].split()[-1])]
                                k+=1
                                for ib in range(Nbands):
                                    k+=1
                                    Enk[ispin,ik,ib] = float(fileread[k].split()[1])
                    break
        structure = Structure.from_file(f"{folder}/POSCAR", format="vasp")
    else:
        print("This format is not implemented.")
        raise NotImplementedError
    return Nk, Nbands, Efermi, Enk-Efermi, np.zeros(Enk.shape), np.zeros([Enk.shape[0], Enk.shape[1]]), kpoint, ispin, structure