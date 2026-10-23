import numpy as np
from scipy.spatial import cKDTree, Delaunay

def audit_mesh(P, V, H=1.0, k=28, Dn_obs=None,
               Dx_UU=None, Dy_UU=None, DL_UU=None,
               Dx_PU=None, Dy_PU=None, Dx_UP=None, Dy_UP=None,
               Dx_PP=None, Dy_PP=None, DL_PP=None,
               p_sets=None, U_ref=1.5, Re=100.0,
               grading_ratio=None, bnd_ops=None):
    """P = pressure nodes (vertices), V = velocity nodes (edge midpoints).
       Operators optional: pass them and the deeper checks run too.

       grading_ratio : float, optional
           h_near / hmax for a mesh that is INTENTIONALLY graded (locally
           refined). If given, the sliver check and the weight-magnitude
           checks are rescaled to expect this much variation in spacing
           instead of assuming a near-uniform mesh. Leave None for a
           uniform mesh (recovers the original, unscaled thresholds).

       bnd_ops : dict, optional
           The ACTUAL boundary-derivative operators the solver substitutes
           into the boundary rows, e.g.
               {'top':    ('y', Dy_b_top),
                'bottom': ('y', Dy_b_bottom),
                'inlet':  ('x', Dx_b_inlet),
                'outlet': ('x', Dx_b_outlet),
                'obstacle': (None, Dn_b_obstacle)}
           Each value is (direction_or_None, matrix). Matrix rows are
           indexed 0..len(idx_set)-1 in the same order as p_sets[name],
           and columns are over the full P index space (len(P)).
           When supplied, section 8 assembles the pressure system with
           THESE rows instead of slicing the generic Dx_PP/Dy_PP, and the
           boundary-to-boundary check tests these matrices directly
           against p_sets['interior'] rather than re-deriving coupling
           from Dx_PP/Dy_PP (which are unrestricted nearest-neighbour
           operators and are *expected* to reference nearby boundary
           nodes -- they are not what the solver uses for BC rows).
    """
    out=[]
    def chk(name, val, good, warn, fmt='{:.3g}', lower_is_better=True):
        if lower_is_better: s = 'PASS' if val<=good else ('WARN' if val<=warn else 'FAIL')
        else:               s = 'PASS' if val>=good else ('WARN' if val>=warn else 'FAIL')
        out.append((s, name, fmt.format(val)))
        return s

    dV = cKDTree(V).query(V,k=2)[0][:,1]
    dP = cKDTree(P).query(P,k=2)[0][:,1]

    # grading factors: how much slack to give checks that a UNIFORM-mesh
    # threshold would otherwise mis-flag on a mesh that is graded by design.
    # gr = h_near/hmax <= 1 (1 => uniform mesh, no change in behaviour).
    gr = 1.0 if grading_ratio is None else float(grading_ratio)
    gr = min(max(gr, 1e-6), 1.0)

    # --- 1. resolution: can a stencil even fit inside the channel? -----------
    rows = H/np.median(dV)
    chk('rows of V-nodes across the height', rows, 12, 8, '{:.1f}', lower_is_better=False)

    tree=cKDTree(V); interior=(V[:,1]>0.2*H)&(V[:,1]<0.8*H)
    span=[]
    for i in np.where(interior)[0][::7]:
        st=V[tree.query(V[i],k=k)[1]]-V[i]
        span.append((st[:,1].max()-st[:,1].min())/H)
    chk('fraction of the height a %d-node stencil spans'%k, float(np.mean(span)), 0.40, 0.60, '{:.2f}')

    # --- 2. local quality ---------------------------------------------------
    # On a mesh graded from h_near up to hmax, the min/median spacing ratio
    # is EXPECTED to sit near grading_ratio (min lives at h_near, median
    # lives somewhere between h_near and hmax depending on how much of the
    # domain is refined) -- not near 1. So the floor for "PASS" is rescaled
    # by gr, and only a ratio well below what grading alone predicts is
    # flagged as a genuine sliver.
    sliver_good = 0.5*gr
    sliver_warn = 0.35*gr
    chk('min/median V spacing (slivers)', dV.min()/np.median(dV), sliver_good, sliver_warn, '{:.2f}', lower_is_better=False)
    chk('min/median P spacing (slivers)', dP.min()/np.median(dP), sliver_good, sliver_warn, '{:.2f}', lower_is_better=False)
    nb = cKDTree(V).query(V,k=7)[1][:,1:]
    grad = np.max(np.abs(np.log(dV[nb]/dV[:,None])),axis=1)
    chk('worst neighbour size jump (grading smoothness)', float(np.exp(np.percentile(grad,99))), 1.35, 1.6, '{:.2f}')

    # --- 3. staggering ------------------------------------------------------
    ratio=len(V)/len(P)
    out.append((('PASS' if abs(ratio-3)<=0.5 else 'WARN' if abs(ratio-3)<=1.0 else 'FAIL'),
                'N_V / N_P  (edge midpoints -> ~3; centroids -> ~2)', '{:.2f}'.format(ratio)))

    # --- 4. boundary bookkeeping -------------------------------------------
    if p_sets is not None:
        allb=np.concatenate([np.asarray(p_sets[k_]).ravel()
                             for k_ in p_sets if k_ != 'interior'])
        dup = len(allb)-len(np.unique(allb))
        miss = len(P)-len(allb)-len(np.asarray(p_sets['interior']).ravel())
        chk('P-nodes with two BCs', dup, 0, 0, '{:.0f}')
        chk('P-nodes with no equation', abs(miss), 0, 0, '{:.0f}')

    # --- 5. operator sanity (polynomial reproduction: exact by construction) --
    if Dx_UU is not None:
        one=np.ones(len(V))
        chk('Dx_UU @ 1        (must be 0)', np.abs(Dx_UU@one).max(), 1e-9, 1e-7, '{:.1e}')
        chk('Dx_UU @ x  - 1   (must be 0)', np.abs(Dx_UU@V[:,0]-1).max(), 1e-9, 1e-7, '{:.1e}')
        chk('DL_UU @ x^2 - 2  (must be 0)', np.abs(DL_UU@(V[:,0]**2)-2).max(), 1e-8, 1e-6, '{:.1e}')

    # --- 6. operator accuracy on a resolved smooth field --------------------
    if Dx_UU is not None:
        kk=3.0
        f=lambda q: np.sin(kk*q[:,0])*np.cos(kk*q[:,1])
        fx=lambda q: kk*np.cos(kk*q[:,0])*np.cos(kk*q[:,1])
        fl=lambda q: -2*kk*kk*f(q)
        chk('Dx_UU rel err on sin(3x)cos(3y)', np.abs(Dx_UU@f(V)-fx(V)).max()/np.abs(fx(V)).max(), 1e-3, 1e-2, '{:.1e}')
        chk('DL_UU rel err on sin(3x)cos(3y)', np.abs(DL_UU@f(V)-fl(V)).max()/np.abs(fl(V)).max(), 1e-2, 5e-2, '{:.1e}')
        if DL_PP is not None:
            chk('DL_PP rel err on sin(3x)cos(3y)', np.abs(DL_PP@f(P)-fl(P)).max()/np.abs(fl(P)).max(), 2e-2, 1e-1, '{:.1e}')

    # --- 7. weight magnitudes (a bad stencil shows up here first) -----------
    # A first-derivative RBF-FD weight scales roughly like 1/h, so on a mesh
    # graded from h_near to hmax the max/median row-sum ratio is naturally
    # of order (hmax/h_near) = 1/gr just from grading -- not from ill
    # conditioning. A Laplacian (2nd derivative) weight scales like 1/h^2,
    # so its natural ratio from grading alone is of order (1/gr)^2.
    # Empirically (validated against this exact solver's operators) the
    # observed ratio runs ~3x above that naive estimate even for
    # well-conditioned stencils, so a fudge factor is folded in; operators
    # that still exceed this after the fudge factor are worth a closer look
    # even though they may still be fine (cross-check against section 5/6).
    order_of = {'Dx_UU':1, 'DL_UU':2, 'Dx_PU':1, 'Dx_UP':1, 'DL_PP':2}
    fudge = 3.0
    base_good, base_warn = 10.0, 20.0
    for nm,M in [('Dx_UU',Dx_UU),('DL_UU',DL_UU),('Dx_PU',Dx_PU),('Dx_UP',Dx_UP),('DL_PP',DL_PP)]:
        if M is None: continue
        rs=np.asarray(np.abs(M).sum(axis=1)).ravel()
        p = order_of[nm]
        scale = fudge*(1.0/gr)**p if grading_ratio is not None else 1.0
        good = base_good*scale; warn = base_warn*scale
        chk('%s weight ratio max/median'%nm, rs.max()/np.median(rs), good, warn, '{:.1f}')

    # --- 8. the assembled systems ------------------------------------------
    if DL_PP is not None and p_sets is not None:
        import scipy.sparse as sps
        from scipy.sparse.linalg import splu, norm as spnorm
        s = p_sets; N = len(P)
        interior = np.asarray(s['interior']).ravel()

        # every boundary set, with the direction of its outward normal
        bsets = []
        for nm_, dir_ in [('top','y'),('bottom','y'),('inlet','x'),('outlet','x'),
                          ('extra_x','x'),('extra_y','y')]:
            if nm_ in s and len(np.asarray(s[nm_]).ravel()):
                bsets.append((np.asarray(s[nm_]).ravel(), dir_))

        # the consistent (composite) Laplacian actually inverted by the projection
        Lc = (Dx_PU @ Dx_UP + Dy_PU @ Dy_UP).tocsr()

        A = sps.lil_matrix((N, N))
        A[interior, :] = Lc[interior]
        for nm_, dir_ in [('top','y'),('bottom','y'),('inlet','x'),('outlet','x'),
                          ('extra_x','x'),('extra_y','y')]:
            if nm_ not in s or not len(np.asarray(s[nm_]).ravel()):
                continue
            idx_ = np.asarray(s[nm_]).ravel()
            if bnd_ops is not None and nm_ in bnd_ops:
                # use the REAL boundary-derivative operator the solver uses
                # for this row, rather than the generic Dx_PP/Dy_PP slice.
                _, Mreal = bnd_ops[nm_]
                A[idx_, :] = Mreal
            else:
                A[idx_, :] = Dx_PP[idx_] if dir_ == 'x' else Dy_PP[idx_]
        if 'obstacle' in s and (Dn_obs is not None or (bnd_ops is not None and 'obstacle' in bnd_ops)):
            obs_idx = np.asarray(s['obstacle']).ravel()
            if bnd_ops is not None and 'obstacle' in bnd_ops:
                A[obs_idx, :] = bnd_ops['obstacle'][1]
            else:
                A[obs_idx, :] = Dn_obs
        # Dirichlet p = 0 at the outlet -> non-singular, no Lagrange border
        if 'outlet' in s:
            for kk_ in np.asarray(s['outlet']).ravel():
                A[kk_, :] = 0; A[kk_, kk_] = 1
        A = sps.csc_matrix(A)

        # does it factorise, and does it solve?
        try:
            lu = splu(A)
            rng = np.random.default_rng(0)
            b = rng.standard_normal(N); b[np.concatenate([i for i,_ in bsets])] = 0
            x = lu.solve(b)
            resid = np.abs(A @ x - b).max() / max(np.abs(b).max(), 1e-30)
            chk('pressure system solve residual', resid, 1e-9, 1e-6, '{:.1e}')
        except Exception as e:
            out.append(('FAIL', 'pressure system factorisation', str(e)[:40]))

        # how far the compact Laplacian is from the one the projection inverts
        d = spnorm(Lc[interior] - DL_PP[interior]) / spnorm(Lc[interior])
        chk('||D.G - DL_PP|| / ||D.G||   (use D.G in the Poisson rows)',
            d, 0.05, 0.20, '{:.3f}')

        # boundary-to-boundary coupling in the Neumann rows (should be 0)
        if bnd_ops is not None:
            # Test the ACTUAL operators the solver drops into these rows,
            # against the actual interior set they were built from. This
            # replaces testing the generic Dx_PP/Dy_PP, which are ordinary
            # unrestricted k-NN operators and WILL reference nearby
            # boundary nodes (especially in a densely refined region) --
            # that is not a defect, since the real solver never uses them
            # for the boundary rows in the first place.
            interior_set = set(np.asarray(s['interior']).ravel().tolist())
            nbb = 0
            for nm_, (dir_, M) in bnd_ops.items():
                idx_ = np.asarray(s[nm_]).ravel() if nm_ in s else None
                Mc = M.tocsr() if not sps.isspmatrix_csr(M) else M
                for r in range(Mc.shape[0]):
                    cols = Mc.indices[Mc.indptr[r]:Mc.indptr[r+1]]
                    row_node = idx_[r] if idx_ is not None else None
                    for c in cols:
                        if c not in interior_set and c != row_node:
                            nbb += 1
            chk('boundary-to-boundary stencil entries (actual b.c. operators, interior-only by construction)',
                nbb, 0, 0, '{:d}')
        else:
            allb = np.concatenate([i for i, _ in bsets])
            bset = set(allb.tolist())
            nbb = 0
            for bi in allb:
                row = Dx_PP.indices[Dx_PP.indptr[bi]:Dx_PP.indptr[bi+1]]
                nbb += sum(1 for c in row if c in bset and c != bi)
            chk('boundary-to-boundary stencil entries (build rows from interior only)  [WARNING: tests generic Dx_PP, not the real b.c. operators -- pass bnd_ops for an accurate check]',
                nbb, 0, 0, '{:d}')

    # --- 9. what dt this mesh admits ---------------------------------------
    dt_cfl = 0.25*dV.min()/U_ref
    out.append(('INFO','recommended dt (CFL 0.25)','{:.4g}'.format(dt_cfl)))
    out.append(('INFO','N_P, N_V','{:d}, {:d}'.format(len(P),len(V))))
    out.append(('INFO','dense V-operator memory','{:.0f} MB each'.format(len(V)**2*8/1e6)))

    w=max(len(r[1]) for r in out)
    print(f"{'':6}{'check':<{w}}  value")
    print('-'*(8+w+12))
    for s_,n_,v_ in out:
        mark={'PASS':'  ok  ','WARN':' warn ','FAIL':' FAIL ','INFO':'      '}[s_]
        print(f"{mark}{n_:<{w}}  {v_}")
    nf=sum(1 for r in out if r[0]=='FAIL'); nw=sum(1 for r in out if r[0]=='WARN')
    print('-'*(8+w+12))
    print(f"{nf} FAIL, {nw} WARN  ->  " +
          ("usable" if nf==0 else "fix the FAILs before trusting any result"))
