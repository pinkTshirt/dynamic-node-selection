import numpy as np
from itertools import product
import sys

# ==========================================
# CORE MATHEMATICAL HELPERS (Preserved)
# ==========================================

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
    C = build_companion(coeffs_np)
    eigs = np.linalg.eigvals(C)
    real_eigs = eigs[np.abs(eigs.imag) < 1e-6].real
    return np.sort(real_eigs)

# ==========================================
# CORE SOLVER ENGINE
# ==========================================

def dynamic_node_selection(degree, a, b, tol, silent=False):
    coeffs, true_roots = generate_poly_from_roots(degree, a, b)
    f = lambda x: eval_poly(coeffs, x)

    if not silent:
        print(f"\n{'='*65}")
        print(f"  Dynamic Node Selection  |  degree={degree}  [{a}, {b}]  tol={tol:.0e}")
        print(f"{'='*65}")
        print(f"  True roots : {np.round(true_roots, 6)}")
        print(f"  Poly coeffs: {np.round(coeffs, 4)}\n")

    # ── Phase 1
    intervals = find_sign_changes(f, a, b)
    if not silent:
        print(f"  Phase 1 — Modified Secant  ({len(intervals)} sign-change bracket(s))")
        print(f"  {'Interval':>22}  {'Root':>12}  {'Residual':>12}  {'Iters':>6}")
        print(f"  {'-'*58}")
    
    sec_results = phase1_secant(f, intervals, tol)
    
    if not silent:
        for r in sec_results:
            lo, hi = r['interval']
            print(f"  [{lo:8.4f}, {hi:8.4f}]  {r['root']:12.8f}  "
                  f"{r['residual']:12.2e}  {len(r['log']):6d}")

    # ── Phase 2
    if not silent:
        print(f"\n  Phase 2 — Chebyshev-Frobenius Companion Eigenvalues")
    eig_roots = companion_eigenvalues(coeffs, a, b)
    eig_roots_inrange = eig_roots[(eig_roots >= a - 0.5) & (eig_roots <= b + 0.5)]
    
    if not silent:
        print(f"  {'Eigenvalue (mapped)':>22}  {'Residual':>12}  {'Source':>14}")
        print(f"  {'-'*54}")
        for r in eig_roots_inrange:
            print(f"  {r:22.8f}  {abs(f(r)):12.2e}  {'Frobenius':>14}")

    # DNS Decision Logic
    if len(intervals) == degree:
        decision = f"SECANT phase complete. Frobenius used as cross-check."
        primary = [r['root'] for r in sec_results]
        method  = "secant (primary) + Frobenius (verification)"
    else:
        deficit = degree - len(intervals)
        decision = f"Only {len(intervals)} brackets found. FROBENIUS fills {deficit} missing root(s)."
        sec_roots = {round(r['root'], 4) for r in sec_results}
        extra = [r for r in eig_roots_inrange
                 if not any(abs(r - s) < 0.01 for s in sec_roots)]
        primary = [r['root'] for r in sec_results] + extra
        method  = "secant (local) + Frobenius (global fallback)"

    if not silent:
        print(f"\n  Dynamic Node Selection decision:")
        print(f"  → {decision}")
        print(f"\n  Final combined root set ({method}):")
        for i, r in enumerate(sorted(primary)):
            nearest_true = true_roots[np.argmin(np.abs(true_roots - r))]
            print(f"    root {i+1}: {r:12.8f}  |  error vs true = {abs(r - nearest_true):.2e}")

    return primary

# ==========================================
# NEW INTERACTIVE TERMINAL CLI ENVIRONMENT
# ==========================================

def get_float_input(prompt, default):
    try:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return float(user_input) if user_input else default
    except ValueError:
        print("Invalid number. Sticking to default.")
        return default

def get_int_input(prompt, default):
    try:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return int(user_input) if user_input else default
    except ValueError:
        print("Invalid integer. Sticking to default.")
        return default

def main_interactive_menu():
    # Setup default state variables
    current_degree = 4
    current_a = -3.0
    current_b = 3.0
    current_tol = 1e-8
    
    historical_configs = [
        dict(degree=2, a=-3,  b=3,  tol=1e-6),
        dict(degree=4, a=-3,  b=3,  tol=1e-8),
        dict(degree=5, a=-4,  b=4,  tol=1e-8),
        dict(degree=6, a=-4,  b=4,  tol=1e-10),
        dict(degree=8, a=-5,  b=5,  tol=1e-10),
    ]

    while True:
        print("\n" + "="*50)
        print("  DYNAMIC NODE SELECTION — INTERACTIVE PLAYGROUND")
        print("="*50)
        print(f"  [1] Run Solver with Current Variables")
        print(f"  [2] Modify Parameters / Toggle Variables")
        print(f"  [3] Run Full Standard Benchmark Suite ({len(historical_configs)} configs)")
        print(f"  [4] Save Last Custom Run to Local CSV")
        print(f"  [5] Exit")
        print("-"*50)
        print(f"  CURRENT STATE: Degree={current_degree} | Domain=[{current_a}, {current_b}] | Tol={current_tol:.0e}")
        print("="*50)
        
        choice = input("Select an action (1-5): ").strip()

        if choice == '1':
            # Run the single configuration using current state
            _ = dynamic_node_selection(current_degree, current_a, current_b, current_tol)
            input("\nPress Enter to return to main menu...")

        elif choice == '2':
            # Toggle / Change parameters
            print("\n--- Change Variables (Press Enter to keep current values) ---")
            current_degree = get_int_input("Enter Polynomial Degree", current_degree)
            current_a = get_float_input("Enter Interval Left Bound (a)", current_a)
            current_b = get_float_input("Enter Interval Right Bound (b)", current_b)
            current_tol = get_float_input("Enter Solver Tolerance (e.g., 1e-8)", current_tol)
            print("\nParameters updated successfully!")

        elif choice == '3':
            # Run through all predefined batch conditions
            print("\nRunning Batch Pipeline...")
            batch_results = {}
            for cfg in historical_configs:
                roots = dynamic_node_selection(silent=True, **cfg)
                batch_results[cfg['degree']] = roots

            print("\n" + "="*65)
            print("  Summary — Batch Root Extraction Report")
            print("="*65)
            print(f"  {'Degree':>6}  {'[a,b]':>12}  {'tol':>8}  {'Roots found':>12}")
            print(f"  {'-'*46}")
            for cfg in historical_configs:
                deg = cfg['degree']
                roots_found = batch_results[deg]
                interval = f"[{cfg['a']},{cfg['b']}]"
                print(f"  {deg:6d}  {interval:>12}  {cfg['tol']:8.0e}  {len(roots_found):12d}")
            input("\nPress Enter to return to main menu...")

        elif choice == '4':
            # Grab real-time data dynamically and dump it locally
            print(f"\nProcessing active variable layout to 'local_results.csv'...")
            roots = dynamic_node_selection(current_degree, current_a, current_b, current_tol, silent=True)
            roots_str = "; ".join([f"{r:.6f}" for r in roots])
            
            try:
                with open("local_results.csv", "w") as f:
                    f.write("Degree, Interval_a, Interval_b, Tolerance, Roots_Found, Root_Values\n")
                    f.write(f"{current_degree}, {current_a}, {current_b}, {current_tol}, {len(roots)}, {roots_str}\n")
                print("Success! File saved as 'local_results.csv' in your working directory.")
            except Exception as e:
                print(f"Error saving file: {e}")
            input("\nPress Enter to return to main menu...")

        elif choice == '5':
            print("\nShutting down playground. Happy computing!")
            sys.exit(0)
        else:
            print("\n[!] Invalid option. Please key in a number from 1 to 5.")

if __name__ == "__main__":
    main_interactive_menu()
