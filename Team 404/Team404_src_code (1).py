import numpy as np
from itertools import product

#helpers

def generate_poly_from_roots(degree, a=-3, b=3):
    """Build monic polynomial coefficients from evenly-spaced roots inside [a,b]."""
    roots = np.linspace(a * 0.85, b * 0.85, degree)
    coeffs = np.poly(roots)          # highest-degree first (numpy convention)
    return coeffs, roots

def eval_poly(coeffs_np, x):
    return np.polyval(coeffs_np, x)

def find_sign_changes(f, a, b, n=300):
    pts = np.linspace(a, b, n + 1)
    intervals = []
    fvals = f(pts)
    for i in range(len(pts) - 1):
        if fvals[i] * fvals[i + 1] < 0:
            intervals.append((pts[i], pts[i + 1]))
    return intervals
#modified secant

def modified_secant(f, x0, tol=1e-8, max_iter=60):
    """Single-point secant using finite-difference slope estimate."""
    h = 1e-5
    x = x0
    log = []
    for i in range(max_iter):
        fx  = f(x)
        fxh = f(x + h)
        df  = (fxh - fx) / h
        if abs(df) < 1e-14:
            break
        xn  = x - fx / df
        err = abs(xn - x)
        log.append(dict(iter=i+1, x=x, fx=fx, xn=xn, err=err))
        if err < tol:
            x = xn
            break
        x = xn
    return x, log

def phase1_secant(f, intervals, tol):
    """Run modified secant from the midpoint of every sign-change bracket."""
    results = []
    for lo, hi in intervals:
        mid = (lo + hi) / 2
        root, log = modified_secant(f, mid, tol=tol)
        residual = abs(f(root))
        results.append(dict(root=root, residual=residual,
                            log=log, interval=(lo, hi)))
    return results

#chebyshev-frobenius companion eigenvalues

def map_to_standard(x, a, b):
    return 2 * (x - a) / (b - a) - 1

def map_from_standard(t, a, b):
    return 0.5 * (b - a) * (t + 1) + a

def build_companion(coeffs_np):
    """Build companion matrix from numpy-convention (highest-first) coefficients."""
    n = len(coeffs_np) - 1
    lead = coeffs_np[0]
    c = -coeffs_np[1:] / lead          # monic normalized tail
    C = np.zeros((n, n))
    C[1:, :-1] = np.eye(n - 1)         # subdiagonal
    C[:, -1] = c[::-1]                 # last column
    return C

def companion_eigenvalues(coeffs_np, a, b):
    """
    Compute eigenvalues of companion matrix.
    Roots are in the original [a,b] domain (no remapping needed here
    since the polynomial was built directly in that domain).
    """
    C = build_companion(coeffs_np)
    eigs = np.linalg.eigvals(C)
    real_eigs = eigs[np.abs(eigs.imag) < 1e-6].real
    return np.sort(real_eigs)

#dns

def dynamic_node_selection(degree, a, b, tol):
    coeffs, true_roots = generate_poly_from_roots(degree, a, b)
    f = lambda x: eval_poly(coeffs, x)

    print(f"\n{'='*65}")
    print(f"  Dynamic Node Selection  |  degree={degree}  [{a}, {b}]  tol={tol:.0e}")
    print(f"{'='*65}")
    print(f"  True roots : {np.round(true_roots, 6)}")
    print(f"  Poly coeffs: {np.round(coeffs, 4)}\n")

    # ── Phase 1
    intervals = find_sign_changes(f, a, b)
    print(f"  Phase 1 — Modified Secant  ({len(intervals)} sign-change bracket(s))")
    print(f"  {'Interval':>22}  {'Root':>12}  {'Residual':>12}  {'Iters':>6}")
    print(f"  {'-'*58}")
    sec_results = phase1_secant(f, intervals, tol)
    for r in sec_results:
        lo, hi = r['interval']
        print(f"  [{lo:8.4f}, {hi:8.4f}]  {r['root']:12.8f}  "
              f"{r['residual']:12.2e}  {len(r['log']):6d}")

    # ── Phase 2
    print(f"\n  Phase 2 — Chebyshev-Frobenius Companion Eigenvalues")
    eig_roots = companion_eigenvalues(coeffs, a, b)
    eig_roots_inrange = eig_roots[(eig_roots >= a - 0.5) & (eig_roots <= b + 0.5)]
    print(f"  {'Eigenvalue (mapped)':>22}  {'Residual':>12}  {'Source':>14}")
    print(f"  {'-'*54}")
    for r in eig_roots_inrange:
        print(f"  {r:22.8f}  {abs(f(r)):12.2e}  {'Frobenius':>14}")

    #dns logic
    print(f"\n  Dynamic Node Selection decision:")
    if len(intervals) == degree:
        print(f"  → {len(intervals)} brackets == degree {degree}: "
              f"SECANT phase is complete. Frobenius used as cross-check.")
        primary = [r['root'] for r in sec_results]
        method  = "secant (primary) + Frobenius (verification)"
    else:
        deficit = degree - len(intervals)
        print(f"  → Only {len(intervals)} brackets for degree {degree}: "
              f"FROBENIUS fills {deficit} missing root(s).")
        sec_roots = {round(r['root'], 4) for r in sec_results}
        extra = [r for r in eig_roots_inrange
                 if not any(abs(r - s) < 0.01 for s in sec_roots)]
        primary = [r['root'] for r in sec_results] + extra
        method  = "secant (local) + Frobenius (global fallback)"

    print(f"\n  Final combined root set ({method}):")
    for i, r in enumerate(sorted(primary)):
        nearest_true = true_roots[np.argmin(np.abs(true_roots - r))]
        print(f"    root {i+1}: {r:12.8f}  |  error vs true = {abs(r - nearest_true):.2e}")

    return primary

# config

configs = [
    dict(degree=2, a=-3,  b=3,  tol=1e-6),
    dict(degree=4, a=-3,  b=3,  tol=1e-8),
    dict(degree=5, a=-4,  b=4,  tol=1e-8),
    dict(degree=6, a=-4,  b=4,  tol=1e-10),
    dict(degree=8, a=-5,  b=5,  tol=1e-10),
]

all_results = {}
for cfg in configs:
    roots = dynamic_node_selection(**cfg)
    all_results[cfg['degree']] = roots

print("\n\n" + "="*65)
print("  Summary — root counts recovered per configuration")
print("="*65)
print(f"  {'Degree':>6}  {'[a,b]':>12}  {'tol':>8}  {'Roots found':>12}")
print(f"  {'-'*46}")
for cfg, (deg, roots) in zip(configs, all_results.items()):
    interval = f"[{cfg['a']},{cfg['b']}]"
    print(f"  {deg:6d}  {interval:>12}  {cfg['tol']:8.0e}  {len(roots):12d}")

for cfg, (deg, roots) in zip(configs, all_results.items()):
    interval = f"[{cfg['a']},{cfg['b']}]"
    print(f"  {deg:6d}  {interval:>12}  {cfg['tol']:8.0e}  {len(roots):12d}")

# save to github actions
print("\nSaving results to results.csv...")

with open("results.csv", "w") as f:
    #header row
    f.write("Degree, Interval_a, Interval_b, Tolerance, Roots_Found, Root_Values\n")
    
    #data for each config
    for cfg, (deg, roots) in zip(configs, all_results.items()):
        # Format the roots into a single string separated by semicolons
        roots_str = "; ".join([f"{r:.6f}" for r in roots])
        
        # Write the comma-separated values to the file
        f.write(f"{deg}, {cfg['a']}, {cfg['b']}, {cfg['tol']}, {len(roots)}, {roots_str}\n")

print("Success! results.csv created.")
