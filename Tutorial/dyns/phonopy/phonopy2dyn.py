import mnx
import mnx.utils.cell as _cell

dyn = mnx.DynMatrix.from_phonopy(folder = "./", qgrid=[2,2,2])
dyn.write("dyn")
