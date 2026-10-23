"""
extend_mesh.py -- lengthen the backward-facing-step node set downstream and
add the missing node at the re-entrant corner.

The node layout in mesh_claude_single_backstep.mat repeats with period 1 in x
downstream of the step (1400 V-nodes and 480 P-nodes per unit length, slabs
match to < 0.005 = 0.2 h).  So the domain is lengthened exactly by
  * keeping every node with x < 8,
  * repeating the slab 7 <= x < 8  K times,
  * shifting the original outlet section (x >= 8, incl. the outlet column)
    by K.
Usage:
    python extend_mesh.py mesh_claude_single_backstep.mat 12 mesh_long.mat
then in the notebook load "mesh_long.mat" and set  L = AR = nodes[:,0].max().
"""
import sys
import numpy as np
from scipy.io import loadmat, savemat

X_STEP = 1.9802
S_STEP = 0.0049 / 0.0101


def extend(P, K, x0=7.0, x1=8.0):
    keep = P[P[:, 0] < x1]
    slab = P[(P[:, 0] >= x0) & (P[:, 0] < x1)]
    tail = P[P[:, 0] >= x1]
    parts = [keep] + [slab + [j * (x1 - x0), 0.0] for j in range(1, K + 1)]
    parts.append(tail + [K * (x1 - x0), 0.0])
    return np.vstack(parts)


def add_corner(P):
    c = np.array([[X_STEP, S_STEP]])
    if np.min(np.hypot(*(P - c).T)) > 1e-9:
        P = np.vstack([P, c])
    return P


if __name__ == "__main__":
    src, K, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    m = loadmat(src)
    V = add_corner(extend(m["pressure"], K))   # velocity nodes (named 'pressure' in the .mat)
    Pn = extend(m["p"], K)                      # pressure nodes
    L = V[:, 0].max()
    savemat(dst, {"pressure": V, "p": Pn, "t": m["t"], "AR": L})
    print(f"V nodes {len(m['pressure'])} -> {len(V)},  P nodes {len(m['p'])} -> {len(Pn)},  L = {L:.4f}")
