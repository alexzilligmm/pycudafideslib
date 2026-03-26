#!/usr/bin/env python3
"""
Remez rational approximation CLI.

Computes (n, m) minimax rational coefficients P(x)/Q(x) ≈ f(x) on [a, b].
Q is monic: q[0] = 1.0.

Supported target functions (--func):
  inv_sqrt   1/sqrt(x)   (default)
  sqrt       sqrt(x)
  exp        exp(x)
  log        log(x)
  tanh       tanh(x)
  sigmoid    1/(1+exp(-x))

Usage examples:
  python remez_cli.py 1 600 3 1
  python remez_cli.py 1e-4 1 3 1
  python remez_cli.py 0 1 4 2 --func tanh
  python remez_cli.py 1 600 3 1 --max-iter 100 --tol 1e-14 --grid 10000
  python remez_cli.py 1 600 3 1 --output cpp
"""

import argparse
import sys
import numpy as np
from scipy.optimize import root


# ---------------------------------------------------------------------------
# Target functions
# ---------------------------------------------------------------------------

FUNCS = {
    "inv_sqrt": lambda x: 1.0 / np.sqrt(x),
    "sqrt":     np.sqrt,
    "exp":      np.exp,
    "log":      np.log,
    "tanh":     np.tanh,
    "sigmoid":  lambda x: 1.0 / (1.0 + np.exp(-x)),
}


# ---------------------------------------------------------------------------
# Chebyshev nodes on [a, b]
# ---------------------------------------------------------------------------

def chebyshev_nodes(a, b, n):
    k = np.arange(1, n + 1)
    t = np.cos((2 * k - 1) / (2 * n) * np.pi)
    return np.sort(0.5 * (b - a) * t + 0.5 * (b + a))


# ---------------------------------------------------------------------------
# Remez algorithm for rational (n, m) approximation
# ---------------------------------------------------------------------------

def rational_remez(func, a, b, n, m,
                   max_iter=50, tol=1e-12, grid_size=5000):
    """
    Returns (p_coeffs, q_coeffs, q_min, q_max).

    p_coeffs : length n+1, coefficients of numerator polynomial (ascending powers)
    q_coeffs : length m+1, coefficients of denominator polynomial (q[0]==1)
    q_min/q_max : range of Q over [a, b] (needed by eval_rational_approx)
    """
    num_points = n + m + 2  # equioscillation theorem requires n+m+2 nodes

    x_nodes = chebyshev_nodes(a, b, num_points)

    # Dense Chebyshev grid for error search
    k = np.arange(1, grid_size + 1)
    t = np.cos((2 * k - 1) / (2 * grid_size) * np.pi)
    x_dense = np.sort(0.5 * (b - a) * t + 0.5 * (b + a))

    p_coeffs = np.zeros(n + 1)
    q_coeffs = np.zeros(m + 1)
    q_coeffs[0] = 1.0

    print(f"Remez (n={n}, m={m}) on [{a}, {b}], func={func.__name__ if hasattr(func, '__name__') else '?'}")

    for iteration in range(max_iter):
        y_nodes = func(x_nodes)

        def equations(vars):
            p = vars[:n + 1]
            q_high = vars[n + 1: n + m + 1]
            E = vars[-1]
            q = np.concatenate(([1.0], q_high))
            P_val = np.polynomial.polynomial.polyval(x_nodes, p)
            Q_val = np.polynomial.polynomial.polyval(x_nodes, q)
            signs = (-1.0) ** np.arange(num_points)
            return P_val - y_nodes * Q_val * (1.0 + signs * E)

        guess = np.zeros(num_points)
        if iteration == 0:
            guess[0] = np.mean(y_nodes)
        else:
            guess[:n + 1] = p_coeffs
            guess[n + 1: n + m + 1] = q_coeffs[1:]

        sol = root(equations, guess, method="lm")

        if not sol.success and iteration > 0:
            print(f"  iter {iteration}: solver stalled, returning best so far.")
            break

        p_coeffs = sol.x[:n + 1]
        q_coeffs = np.concatenate(([1.0], sol.x[n + 1: n + m + 1]))
        E_curr = sol.x[-1]

        y_dense = func(x_dense)
        num_dense = np.polynomial.polynomial.polyval(x_dense, p_coeffs)
        den_dense = np.polynomial.polynomial.polyval(x_dense, q_coeffs)

        if np.min(np.abs(den_dense)) < 1e-9:
            print("  WARNING: denominator near zero — pole detected in [a, b].")

        approx = num_dense / den_dense
        rel_error = (approx - y_dense) / y_dense
        abs_rel_error = np.abs(rel_error)

        max_err_idx = np.argmax(abs_rel_error)
        max_err_val = abs_rel_error[max_err_idx]
        x_extremum = x_dense[max_err_idx]
        ext_sign = np.sign(rel_error[max_err_idx])

        diff = abs(max_err_val - abs(E_curr))
        print(f"  iter {iteration:3d}: |E|={abs(E_curr):.3e}  max_rel_err={max_err_val:.3e}  diff={diff:.3e}")

        if diff < tol:
            print(f"  Converged at iteration {iteration}.")
            break

        # Update reference nodes
        n_num = np.polynomial.polynomial.polyval(x_nodes, p_coeffs)
        n_den = np.polynomial.polynomial.polyval(x_nodes, q_coeffs)
        node_errs = (n_num / n_den - y_nodes) / y_nodes
        node_signs = np.sign(node_errs)

        idx = np.searchsorted(x_nodes, x_extremum)
        if idx == 0:
            if np.sign(ext_sign) == np.sign(node_signs[0]):
                x_nodes[0] = x_extremum
            else:
                x_nodes = np.insert(x_nodes, 0, x_extremum)[:-1]
        elif idx == num_points:
            if np.sign(ext_sign) == np.sign(node_signs[-1]):
                x_nodes[-1] = x_extremum
            else:
                x_nodes = np.append(x_nodes, x_extremum)[1:]
        else:
            s_left = node_signs[idx - 1]
            s_right = node_signs[idx]
            if np.sign(ext_sign) == np.sign(s_left):
                x_nodes[idx - 1] = x_extremum
            elif np.sign(ext_sign) == np.sign(s_right):
                x_nodes[idx] = x_extremum

        x_nodes = np.sort(x_nodes)

    # Final Q range over a dense linear grid
    x_final = np.linspace(a, b, 1000)
    den_final = np.polynomial.polynomial.polyval(x_final, q_coeffs)
    q_min = float(np.min(den_final))
    q_max = float(np.max(den_final))

    return p_coeffs, q_coeffs, q_min, q_max


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def fmt_vec(name, coeffs):
    inner = ",\n    ".join(f"{c:.15e}" for c in coeffs)
    return f"static const std::vector<double> {name} = {{\n    {inner}\n}};"


def print_cpp(p, q, q_min, q_max, label):
    print()
    print(fmt_vec(f"REMEZ_P_{label}", p))
    print(fmt_vec(f"REMEZ_Q_{label}", q))
    print(f"static constexpr double REMEZ_QMIN_{label} = {q_min:.15e};")
    print(f"static constexpr double REMEZ_QMAX_{label} = {q_max:.15e};")


def print_python(p, q, q_min, q_max):
    print()
    print("p_coeffs =", list(p))
    print("q_coeffs =", list(q))
    print(f"q_min = {q_min:.15e}")
    print(f"q_max = {q_max:.15e}")


def print_plain(p, q, q_min, q_max):
    print()
    print("Numerator P (ascending powers of x):")
    for i, c in enumerate(p):
        print(f"  p[{i}] = {c:.15e}")
    print("Denominator Q (ascending powers of x, q[0]=1):")
    for i, c in enumerate(q):
        print(f"  q[{i}] = {c:.15e}")
    print(f"Q range over [a,b]: min={q_min:.6e}  max={q_max:.6e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute Remez (n,m) rational minimax coefficients for a target function.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("a", type=float, help="Left endpoint of the interval")
    parser.add_argument("b", type=float, help="Right endpoint of the interval")
    parser.add_argument("n", type=int, help="Degree of numerator polynomial")
    parser.add_argument("m", type=int, help="Degree of denominator polynomial")
    parser.add_argument("--func", default="inv_sqrt",
                        choices=list(FUNCS.keys()),
                        help="Target function (default: inv_sqrt = 1/sqrt(x))")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum Remez iterations (default: 50)")
    parser.add_argument("--tol", type=float, default=1e-12,
                        help="Convergence tolerance on equioscillation error (default: 1e-12)")
    parser.add_argument("--grid", type=int, default=5000,
                        help="Dense grid size for extremum search (default: 5000)")
    parser.add_argument("--output", choices=["plain", "python", "cpp"], default="plain",
                        help="Output format (default: plain)")
    parser.add_argument("--label", default=None,
                        help="Label suffix for C++ variable names (default: auto)")

    args = parser.parse_args()

    if args.a >= args.b:
        print("Error: a must be < b", file=sys.stderr)
        sys.exit(1)

    func = FUNCS[args.func]
    # Attach a name for display
    func.__name__ = args.func

    p, q, q_min, q_max = rational_remez(
        func, args.a, args.b, args.n, args.m,
        max_iter=args.max_iter, tol=args.tol, grid_size=args.grid,
    )

    label = args.label or f"{args.func}_{args.n}_{args.m}"

    if args.output == "cpp":
        print_cpp(p, q, q_min, q_max, label.upper())
    elif args.output == "python":
        print_python(p, q, q_min, q_max)
    else:
        print_plain(p, q, q_min, q_max)


if __name__ == "__main__":
    main()
