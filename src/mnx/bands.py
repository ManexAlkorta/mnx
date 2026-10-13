from .structure import Structure
import mnx.utils.cell as _cell

import mnx.FModules

import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import os
import spglib
import ase.io

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Voronoi, Delaunay, ConvexHull

from scipy.special import erfc


from skimage.measure import marching_cubes, find_contours
import pyvista as pv

class Bands():
    """Object that contains the data from a regular grid band calculation. Supports
    non-magnetic and collinear calculations."""

    @classmethod
    def from_nscf(cls, folder, mod, format, upsample=[1,1,1], magmoms=0):
        """
        Classmethor of Bands. Reads the bands from a nscf calculation.
        
        Parameters
        ----------
            folder : str
                Path to the folder containing the OUTCAR.scf/RUN.scf.out, OUTCAR/RUN.nscf.out
                and POSCAR/STRUCT.pwi.
            mod : list or np.ndarray
                The regular k-grid employed in the calculation.
            format : str
                The code employed for the nscf calculation.
                Supported options:
                    qe: Quantum-Espresso
                    vasp: VASP
            upsample : list or np.ndarray
                The integers to build a denser k-grid from Fourier interpolation. For example,
                to make a 2 times denser grid in all reciprocal lattice-vector directions we 
                should specify upsample = [2,2,2].
                Defaults to [1,1,1].
            magmoms : list or np.ndarray
                In the case of a collinear calculation, specify the local spin magnetic moment
                of each atom. It is used to detect symmetries.
        Returns
        -------
            Bands : object
                An initialized instance of the Bands object.
        """
        instance = cls()
        instance.format, instance.mod, instance.upsample, instance.magmoms = format, np.asarray(mod), np.asarray(upsample), np.asarray(magmoms)

        instance.Nk, instance.Nbands, instance.Efermi, instance.Enk, instance.kpoint, instance.ispin, instance.structure = _read_nscf(folder, format=instance.format)

        instance.colors = [[
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
        return instance
        

    def build_setup(self, filter = True, Ecut=0, delta=0):
        """
        Function which builds the setup of the Bands object, filtering to bands crossing
        a specified energy if desired.

        Parameters
        ----------
            filer : boolean
                If True, only bands crossing Ecut are considered.
                Defaults to True.
            Ecut : float
                If filter == True, defines the energy employed for the 
                filering. It is defined respect to the Fermi energy in eV.
            delta : float
                If filter == True, defines how close has to be a band to consider
                it crosses Ecut.
        Returns
        -------
            None
        """
        if np.all(self.magmoms == 0):
            self.magmoms = np.zeros(self.structure.Natoms)
        # Using spglib, find the mapping to the calculated k-grid.
        self.bigEnk, self.bigk = _complete_grid(self.structure, self.Enk, self.kpoint, self.mod, self.magmoms)
        if filter:
            self.fbigEnk = _filter_bands(self.bigEnk, Ecut, self.ispin, delta=delta)
        else:
            self.fbigEnk = self.bigEnk
        # Now, using fourier interpolation, we generate a Gamma centered
        # periodic sampling of the brillouin zone.
        self.fhugeEnk, self.hugek = _denser_grid(self.bigk, self.fbigEnk, Ecut, self.mod, self.upsample, self.ispin)


    def visualize_fsurface(self, plot=True):
        """
        Function that visualizes the Fermi surface.

        Parameters
        ----------
            plot : bool
                If True, plots the fermi surface apart from just returning the data.
                Defaults to True.
        Returns
        -------
            meshes : pyvista.PolyData
                Pyvista meshes of the bands crossing the Fermi energy
            bz_mesh : pyvista.PolyData
                Pyvista mesh of the first Brillouin zone
        """
        nbands = self.fhugeEnk.shape[1]
        meshes = np.empty([self.ispin, self.fbigEnk.shape[2]], dtype=object)
        do_plot = np.zeros([self.ispin, self.fbigEnk.shape[2]], dtype=bool)

        # Pre-extract structure and coordinate constants
        rcell = self.structure.rcell
        shift = rcell[0, :] + rcell[1, :] + rcell[2, :]
        spacing = (
            1 / self.upsample[0],
            1 / self.upsample[1],
            1 / self.upsample[2],
        )
        mod = self.mod

        def process_band(s, n):
            """Worker function for parallel Marching Cubes + PyVista conversion."""
            try:
                data = self.fhugeEnk[s, n]
                # Fast check: skip marching_cubes if level=0.0 is out of bounds
                if data.min() > 0.0 or data.max() < 0.0:
                    return s, n, None, False

                verts, faces, _, _ = marching_cubes(
                    data, level=0.0, spacing=spacing
                )

                # Vectorized PyVista face padding ([3, v0, v1, v2, 3, v0, ...])
                faces_pv = np.pad(faces, ((0, 0), (1, 0)), constant_values=3).ravel()

                # Transform coordinates
                verts /= mod
                cart_verts = _cell.cryst2cart(verts, rcell) - shift

                # Create PyVista PolyData
                mesh = pv.PolyData(cart_verts, faces_pv)
                bz_mesh = _plot_fs_in_bz_voronoi(mesh, rcell)

                return s, n, bz_mesh, True
            except ValueError:
                return s, n, None, False

        # Parallelize Marching Cubes extraction across bands and spins
        tasks = [
            (s, n) for s in range(self.ispin) for n in range(nbands)
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda args: process_band(*args), tasks
            )

        for s, n, mesh, success in results:
            if success:
                meshes[s, n] = mesh
                do_plot[s, n] = True

        bz_wire = _build_bz_wireframe(rcell)

        if plot:
            pv.global_theme.jupyter_backend = "trame"
            p = pv.Plotter()
            for s in range(self.ispin):
                for mi, mesh in enumerate(meshes[s]):
                    if do_plot[s, mi]:
                        p.add_mesh(
                            mesh,
                            color=self.colors[s][mi],
                            specular=0.5,
                            specular_power=20,
                            diffuse=0.8,
                            ambient=0.3,
                        )
            p.add_mesh(bz_wire, color="black", style="wireframe", line_width=2)
            p.show()

        return meshes, bz_wire


    def visualize_fp(self, ax, axis="ab", k=0, plot=True):
        """
        Function for visualizing Fermi surface cuts.

        Parameters
        ----------
        ax : matplotlib axis
            Matplotlib axis in which the plane will be plotted.

        axis : str
            The two crystallographic axes defining the Fermi plane.
            Options: "ab", "ac", or "bc".

        k : float
            Point defining the position of the plane, given in
            fractional coordinates.

        plot : bool
            If True, plot the Fermi contours and Brillouin zone cut.

        Returns
        -------
        contours_list : list
            List containing the bands to plot.
        """

        rcell = np.asarray(self.structure.rcell, dtype=float)

        # Reciprocal lattice vectors
        b1 = rcell[0]
        b2 = rcell[1]
        b3 = rcell[2]

        Mx = self.mod[0] * self.upsample[0]
        My = self.mod[1] * self.upsample[1]
        Mz = self.mod[2] * self.upsample[2]

        # ---------------------------------------------------------------
        # Define the cutting plane
        # ---------------------------------------------------------------

        if axis == "bc":
            # Fixed ka = k
            origin = k * b1
            normal = np.cross(b2, b3)
            cut_type = "bc"

        elif axis == "ac":
            # Fixed kb = k
            origin = k * b2
            normal = np.cross(b3, b1)
            cut_type = "ac"

        elif axis == "ab":
            # Fixed kc = k
            origin = k * b3
            normal = np.cross(b1, b2)
            cut_type = "ab"

        else:
            raise ValueError("axis must be 'bc', 'ac', or 'ab'")

        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-14:
            raise ValueError(
                f"Could not construct plane normal for axis='{axis}'."
            )

        normal = normal / normal_norm

        # ---------------------------------------------------------------
        # Construct an orthonormal basis in the cutting plane.
        # The plotted coordinates are only used for visualization;
        # no crystallographic axis labels are assigned to them.
        # ---------------------------------------------------------------

        if cut_type == "ab":

            u = b1 / np.linalg.norm(b1)

            v = b2 - np.dot(b2, u) * u
            v_norm = np.linalg.norm(v)

            if v_norm < 1e-14:
                raise ValueError(
                    "Could not construct in-plane basis for the ab cut."
                )

            v = v / v_norm

        else:

            b3_norm = np.linalg.norm(b3)

            if b3_norm < 1e-14:
                raise ValueError("Could not construct the c direction.")

            v = b3 / b3_norm

            if cut_type == "ac":
                x_ref = b1
            else:
                x_ref = b2

            u = x_ref - np.dot(x_ref, v) * v
            u_norm = np.linalg.norm(u)

            if u_norm < 1e-14:
                raise ValueError(
                    f"Could not construct in-plane basis for "
                    f"the {cut_type} cut."
                )

            u = u / u_norm

        # ---------------------------------------------------------------
        # Contour extraction
        # ---------------------------------------------------------------

        contours_list = [[] for _ in range(self.ispin)]

        for s in range(self.ispin):

            spin_contours = []

            for n in range(self.fhugeEnk.shape[1]):

                # =======================================================
                # bc: fixed ka, vary kb and kc
                # =======================================================

                if cut_type == "bc":

                    index = int(np.round(Mx + Mx * k))

                    contours = find_contours(
                        self.fhugeEnk[s, n, index, :, :],
                        level=0
                    )

                    new_contours = []

                    for c in contours:

                        contours3 = np.empty([len(c), 3])

                        for yzi, yz in enumerate(c):

                            kb = yz[0] / My - 1.0
                            kc = yz[1] / Mz - 1.0
                            ka = k

                            contours3[yzi, :] = _cell.cryst2cart(
                                np.array([ka, kb, kc]),
                                rcell
                            )

                        segments = _split_contour_by_bz(
                            contours3,
                            rcell=rcell
                        )

                        new_contours.extend(segments)

                # =======================================================
                # ac: fixed kb, vary ka and kc
                # =======================================================

                elif cut_type == "ac":

                    index = int(np.round(My + My * k))

                    contours = find_contours(
                        self.fhugeEnk[s, n, :, index, :],
                        level=0
                    )

                    new_contours = []

                    for c in contours:

                        contours3 = np.empty([len(c), 3])

                        for xzi, xz in enumerate(c):

                            ka = xz[0] / Mx - 1.0
                            kc = xz[1] / Mz - 1.0
                            kb = k

                            contours3[xzi, :] = _cell.cryst2cart(
                                np.array([ka, kb, kc]),
                                rcell
                            )

                        segments = _split_contour_by_bz(
                            contours3,
                            rcell=rcell
                        )

                        new_contours.extend(segments)

                # =======================================================
                # ab: fixed kc, vary ka and kb
                # =======================================================

                elif cut_type == "ab":

                    index = int(np.round(Mz + Mz * k))

                    contours = find_contours(
                        self.fhugeEnk[s, n, :, :, index],
                        level=0
                    )

                    new_contours = []

                    for c in contours:

                        contours3 = np.empty([len(c), 3])

                        for xyi, xy in enumerate(c):

                            ka = xy[0] / Mx - 1.0
                            kb = xy[1] / My - 1.0
                            kc = k

                            contours3[xyi, :] = _cell.cryst2cart(
                                np.array([ka, kb, kc]),
                                rcell
                            )

                        segments = _split_contour_by_bz(
                            contours3,
                            rcell=rcell
                        )

                        new_contours.extend(segments)

                # -------------------------------------------------------
                # Store contours
                # -------------------------------------------------------

                spin_contours.append(new_contours)

                # -------------------------------------------------------
                # Plot Fermi contours
                # -------------------------------------------------------

                if plot:

                    for c in new_contours:

                        c_rel = c - origin

                        # Project onto the in-plane basis
                        c_u = c_rel @ u
                        c_v = c_rel @ v

                        ax.plot(
                            c_u,
                            c_v,
                            color=self.colors[s][n],
                            alpha=0.5
                        )

            contours_list[s] = spin_contours

        # ===============================================================
        # Brillouin zone cut
        # ===============================================================

        if plot:

            bz_mesh = _build_bz_mesh(rcell)

            bz_points = np.asarray(bz_mesh.points)
            bz_faces = bz_mesh.faces.reshape(-1, 4)[:, 1:]

            for f in bz_faces:

                tri = bz_points[f]

                # Signed distance from the cutting plane
                dists = (tri - origin) @ normal

                # Triangle entirely on one side of the plane
                if np.all(dists > 0):
                    continue

                if np.all(dists < 0):
                    continue

                pts = []

                # -------------------------------------------------------
                # Intersect triangle edges with the plane
                # -------------------------------------------------------

                for i in range(3):

                    j = (i + 1) % 3

                    d1 = dists[i]
                    d2 = dists[j]

                    # Vertex lies on the plane
                    if abs(d1) < 1e-12:
                        pts.append(tri[i])

                    # Edge crosses the plane
                    if d1 * d2 < 0:

                        t = d1 / (d1 - d2)

                        pt = tri[i] + t * (tri[j] - tri[i])

                        pts.append(pt)

                # -------------------------------------------------------
                # Remove duplicate intersection points
                # -------------------------------------------------------

                unique_pts = []

                for p in pts:

                    if not any(
                        np.linalg.norm(p - q) < 1e-10
                        for q in unique_pts
                    ):
                        unique_pts.append(p)

                if len(unique_pts) == 2:

                    pts_arr = np.asarray(unique_pts) - origin

                    # Project BZ boundary into the same plane coordinates
                    bz_u = pts_arr @ u
                    bz_v = pts_arr @ v

                    ax.plot(
                        bz_u,
                        bz_v,
                        color="black",
                        linewidth=1.0
                    )

            # -----------------------------------------------------------
            # Final plot formatting: no axes, ticks, labels, or frame
            # -----------------------------------------------------------

            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()

        return contours_list
    
    def get_bands(self, k_inpath, N):
        """
        Function to obtain visualization ready bands. The interpolation is unphysical, and completly fails in 
        band crossings and parallel bands.
        
        Parameters
        ----------
            k_inpath : list or np.ndarray
                k-path in fractional coordinates.
            N : int
                Number of points to include between the k-points defined by k-path

        Returns
        -------
            bands : np.ndarray
                Visualization ready bands in eV.
            rq : np.ndarray
                x-axis for the visualization of the bands.
            xticks : np.ndarray
                Values of rq that correspond to the points defined by k_inpath.
        """

        k_inpath = np.asarray(k_inpath)
        qpath=np.empty([len(k_inpath)*(N+1),3])
        bands = np.empty([self.ispin,len(k_inpath)*(N+1),self.bigEnk.shape[2],])
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
                f_fft = _fourier_interpolator(self.bigk, self.bigEnk[s,:,n], shape=(self.mod[0], self.mod[1], self.mod[2]), upsample=self.upsample)
                bands[s,:,n] = f_fft(qpath)
        return bands, rq, xticks
    
    def get_Nsq(self, sigma=1e-2, downsample=[1,1,1], Ecut = 0):
        """
        Function that computes the q-resolved nesting function. Interface of the 
        fortran subroutine FModules.bands.get_nesting().
        
        Parameters
        ----------
            sigma : float
                Smearing parameter in eV.
                Defaults to 1e-2.
            downsample : list or np.ndarray
                Downsampling grid to makes the calculation affordable.
                Defaults to [1,1,1].
            Ecut : float
                Energy in which the gaussian smearing is employed.
                Defaults to 0 (i.e. the Fermi energy).

        Returns
        -------
            None
        """

        Nk = self.bigk.shape[0]
        Ndown = int(Nk/np.prod(downsample))

        self.Nsk = np.zeros([self.ispin,Ndown])
        self.Nsk_downk = np.empty([Ndown,3])
        diffuse_fermis, hugek = self._get_gaussian_bands(Ecut=Ecut, sigma=sigma)
  
        hugek = np.round(hugek)
        self.Nsk, self.Nsk_downk = mnx.FModules.bands.get_nesting(Ndown, self.mod, downsample, hugek, diffuse_fermis)

    def get_Xnk(self,nmin,nmax,sigma=1e-2,downsample=[1,1,1]):
        """
        Function that computes the band resolved Lindhard susceptibility nesting function.
        Interface of the fortran subroutine FModules.bands.get_xnk(). Fermi dirac distribution
        function is employed, where sigma == kbT in eV.
        
        Parameters
        ----------
            nmin : integer
                Index of the first band conidered in the calculation of Xnk.
            nmax : integer
                Index of the last band considered in the calculation of Xnk.
            sigma : float
                Smearing parameter in eV.
                Defaults to 1e-2.
            downsample : list or np.ndarray
                Downsampling grid to makes the calculation affordable.
                Defaults to [1,1,1].

        Returns
        -------
            None
        """
        
        N = nmax-nmin+1
        Nk = self.bigk.shape[0]
        Ndown = int(Nk/np.prod(downsample))

        self.Xnk = np.zeros([self.ispin,N,Ndown])
        self.Xnk_downk = np.empty([Ndown,3])

        # this function is not needed, but its the way hugek is implemented at the moment
        diffuse_fermis, hugek = self._get_gaussian_bands(Ecut=0, sigma=sigma)

        self.Xnk, self.Xnk_downk = mnx.FModules.bands.get_xnk(Ndown,nmin,nmax,self.bigEnk,sigma,hugek,self.mod,downsample)

    def get_nesting_qpath(self, k_inpath, N=50, downsample=[1,1,1]):
        """
        Function to obtain a visualization ready nesting in a q-path.

        Parameters
        ----------
            k_inpath : list or np.ndarray
                k-path in fractional coordinates.
            N : int
                Number of points to include between the k-points defined by k-path.
                Defaults to 50.
            downsample: list or np.ndarray
                Downsample employed at get_Nsk.
                Defaults to [1,1,1].
        
        Returns
        -------
            nesting : np.ndarray
                k-resolved nesting function in units of (states/spin/eV^2).
            rq : np.ndarray
                x-axis for the visualization of the nesting function.
            xticks : np.ndarray
                Values of rq that correspond to the points defined by k_inpath.
        """
        
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
            f_fft = _fourier_interpolator(self.Nsk_downk, self.Nsk[si], shape=(int(self.mod[0]/downsample[0]), int(self.mod[1]/downsample[1]), int(self.mod[2]/downsample[2])), upsample=self.upsample)
            nesting[si] = f_fft(qpath/downsample)
        return nesting, rq, xticks

    def get_Xnk_qpath(self, k_inpath, N=50, downsample=[1,1,1]):
        """
        Function to obtain a visualization ready Xnk in a q-path.

        Parameters
        ----------
            k_inpath : list or np.ndarray
                k-path in fractional coordinates.
            N : int
                Number of points to include between the k-points defined by k-path.
                Defaults to 50.
            downsample: list or np.ndarray
                Downsample employed at get_Nsk.
                Defaults to [1,1,1].
        
        Returns
        -------
            Xnk : np.ndarray
                k-resolved Xnk function in units of (states/spin/eV^2).
            rq : np.ndarray
                x-axis for the visualization of the nesting function.
            xticks : np.ndarray
                Values of rq that correspond to the points defined by k_inpath.
        """
        
        k_inpath = np.asarray(k_inpath)
        qpath=np.empty([len(k_inpath)*(N+1),3])
        xnk = np.zeros([self.ispin, len(qpath), self.Xnk.shape[1]])
        rq = np.zeros(len(k_inpath)*(N+1))
        xticks = [0]
        for ki,ks in enumerate(k_inpath):
            dk = (ks[1]-ks[0])
            rq[ki*(N+1):(ki+1)*(N+1)] = rq[ki*(N+1)-1]+np.linalg.norm(_cell.cryst2cart(dk, self.structure.rcell))*np.arange(0,10+10/N,10/N)
            xticks.append(rq[(ki+1)*(N+1)-1])
            dk = dk * self.mod
            qpath[ki*(N+1):(ki+1)*(N+1)] = dk[None,:]*np.arange(0,1+1/N,1/N)[:,None]+ks[0]*self.mod
        for si in range(self.ispin):
            for ni in range(self.Xnk.shape[1]):
                f_fft = _fourier_interpolator(self.Xnk_downk, self.Xnk[si,ni], shape=(int(self.mod[0]/downsample[0]), int(self.mod[1]/downsample[1]), int(self.mod[2]/downsample[2])), upsample=self.upsample)
                xnk[si,:,ni] = f_fft(qpath/downsample)
        return xnk, rq, xticks
    
    def _get_gaussian_bands(self, Ecut, sigma=5e-3):
        bigEnk, bigk = _denser_grid(self.bigk, self.bigEnk, Ecut, self.mod, self.upsample, self.ispin, fix_rcell=True)
        diffuse_fermis = _smear_bands(bigEnk, sigma=sigma)
        return diffuse_fermis, bigk

def _fourier_interpolator(kpoints, energies, shape, upsample=(1,1,1)):
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


def _denser_grid(big_kpoint, big_Enk, Ecut, mod, upsample, ispin, fix_rcell=False):
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
            f_fft = _fourier_interpolator(big_kpoint, big_Enk[s,:,n]-Ecut, shape=(mod[0], mod[1], mod[2]), upsample=upsample)
            huge_Enk[s,n] = f_fft(points).reshape(len(kxs), len(kys), len(kzs))
    return huge_Enk, points


def _complete_grid(structure, Enk, kpoint, mod, magmoms=0, symprec=1e-5):
    cell = (structure.cell, _cell.cart2cryst(structure.atom_coords, structure.cell), structure.atom_species[:,1])
    mapping, address, trs = _get_ir_reciprocal_mesh_magnetic(mod, cell, magmoms, kpoint, symprec)
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


def _filter_bands(big_Enk, Efermi, ispin, delta = 0):
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

def _gaussian(x, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*(x**2/sigma**2))

def _smear_bands(Enk, sigma=5e-3):
    diffuse_fermi = np.zeros([Enk.shape[0],Enk.shape[2], Enk.shape[3], Enk.shape[4]])
    for s in  range(Enk.shape[0]): 
        for n in range(Enk.shape[1]):
            diffuse_fermi[s] += _gaussian(Enk[s,n,:,:,:], sigma=sigma)
    if Enk.shape[0] == 1:
        return np.array([diffuse_fermi[0].ravel("C")])
    else:
        return np.array([diffuse_fermi[0].ravel("C"), diffuse_fermi[1].ravel("C")])


def _compute_bz_vertices(rcell, nmax=2):
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


def _filter_fs_inside_bz_voronoi(
    fs, rcell, nmax=2, tol=1e-8, cut=True, is_masked=False
):
    """Remove triangles of FS mesh if any vertex is outside the first BZ.

    Fast vectorized implementation that preserves original cell orientation.
    """
    points = fs.points

    # 1. Get exact BZ vertices in real coordinate space and extract plane equations
    bz_vertices = _compute_bz_vertices(rcell, nmax=nmax)
    hull = ConvexHull(bz_vertices)

    # hull.equations gives plane equations: A*x + B*y + C*z + D <= 0
    planes = hull.equations  # shape: (N_planes, 4)
    normals = planes[:, :-1]  # A, B, C
    offsets = planes[:, -1]  # D

    # 2. Fast analytical inside-BZ check: point is inside if ALL A*x + B*y + C*z + D <= 1e-8
    inside_mask = np.all((points @ normals.T + offsets) <= 1e-8, axis=1)

    # 3. Extract vertex indices for all faces simultaneously: shape (N_faces, 3)
    faces = fs.faces.reshape(-1, 4)
    vidxs = faces[:, 1:]

    # 4. Keep face ONLY if ALL 3 vertices are inside the BZ
    keep_face_mask = np.all(inside_mask[vidxs], axis=1)

    # 5. Vectorized octant cutting (if enabled)
    if cut and is_masked:
        positive_points = np.all(points >= 0, axis=1)
        # Keep face if at least one vertex is negative (~positive)
        has_negative_vertex = np.any(~positive_points[vidxs], axis=1)
        keep_face_mask &= has_negative_vertex

    keep_faces = faces[keep_face_mask]

    if len(keep_faces) == 0:
        print("Warning: no faces remain inside first BZ!")
        return None

    # Reconstruct PyVista PolyData mesh
    filtered_fs = pv.PolyData(points, keep_faces.ravel())
    filtered_fs = filtered_fs.triangulate().clean(tolerance=tol)
    if filtered_fs.n_faces_strict > 0:
        filtered_fs = filtered_fs.compute_normals(
            auto_orient_normals=True, feature_angle=150
        )

    return filtered_fs

# ---------------------------
# Build BZ wireframe mesh
# ---------------------------
def _build_bz_mesh(rcell, nmax=2):
    """
    Build wireframe mesh for first BZ (Voronoi around origin)
    """
    bz_vertices = _compute_bz_vertices(rcell, nmax=nmax)
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
def _plot_fs_in_bz_voronoi(fs_mesh, rcell, nmax=2):
    filtered_fs = _filter_fs_inside_bz_voronoi(fs_mesh, rcell, nmax=nmax)
    return filtered_fs

def _build_bz_wireframe(rcell, nmax=2):
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

def _is_point_inside_bz(k, rcell, tol=1e-8):
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


def _split_contour_by_bz(contour, rcell):
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
        if _is_point_inside_bz([point[0], point[1], point[2]], rcell):
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



def _inrcell(k, mod):
    for i in range(3):
        if k[i]<0 and k[i]%mod[i]!=0:
            k[i]+=(np.abs(k[i])//mod[i]+1)*mod[i]
        elif k[i]<0 and k[i]%mod[i]==0:
            k[i]+=(np.abs(k[i])//mod[i])*mod[i]
        elif k[i]>=mod[i]:
            k[i]+=-np.abs((k[i]//mod[i])*mod[i])
    return np.round(k)


def _get_ir_reciprocal_mesh_magnetic(mod, cell, magmoms, kpoint, symprec=1e-5):
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
    address_dict = {tuple(_inrcell(tmp_kpoint[i], mod)):i for i in range(len(tmp_kpoint))}

    rkg_list = []
    for kgi, kg in enumerate(kgrid):
        tmp_kg = kg
        address_dict[kgi] = kg
        is_found = False
        for ri, R in enumerate(rotations):
            try:
                if time_reversals[ri] == 0:
                    rkg = tuple(_inrcell(np.matmul(-np.transpose(R), tmp_kg), mod))
                else:
                    rkg = tuple(_inrcell(np.matmul(np.transpose(R), tmp_kg), mod))
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
            structure = Structure.from_file(f"{folder}/STRUCT.pwi", format="qe")

    elif format=="vasp":
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
                        try:
                            k = li+2
                            for ik in range(Nk):
                                k+=2
                                kpoint[ik] = [float(fileread[k].split()[-3]), float(fileread[k].split()[-2]), float(fileread[k].split()[-1])]
                                k+=1
                                for ib in range(Nbands):
                                    k+=1
                                    Enk[0,ik,ib] = float(fileread[k].split()[1])
                        except:
                            k = li+5
                            for ik in range(Nk):
                                k+=2
                                kpoint[ik] = [float(fileread[k].split()[-3]), float(fileread[k].split()[-2]), float(fileread[k].split()[-1])]
                                k+=1
                                for ib in range(Nbands):
                                    k+=1
                                    Enk[0,ik,ib] = float(fileread[k].split()[1])
                    elif ispin == 2:
                        # Not tested.
                        Enk = np.zeros([ispin, Nk, Nbands])
                        try:
                            k = li+2
                            for ispin in range(2):
                                for ik in range(Nk):
                                    k+=2
                                    kpoint[ik] = [float(fileread[k].split()[-3]), float(fileread[k].split()[-2]), float(fileread[k].split()[-1])]
                                    k+=1
                                    for ib in range(Nbands):
                                        k+=1
                                        Enk[ispin,ik,ib] = float(fileread[k].split()[1])
                        except:
                            k = li+5
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
    return Nk, Nbands, Efermi, Enk-Efermi, kpoint, ispin, structure