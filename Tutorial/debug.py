import numpy as np

phiR_cc, phiR_mnx = np.load("phiR1_cc.npy"), np.load("phiR1_mnx.npy")

print(phiR_mnx.max())
if not np.isclose(phiR_cc,phiR_mnx,atol=1e-6).all():
    print("PhiR wrong:",(phiR_cc-phiR_mnx).max())
breakpoint()
