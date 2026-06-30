# Dynamic Node Selection (Hybrid Polynomial Root-Finder)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, hybrid mathematical framework designed to accurately extract all real roots of a monic polynomial. The tool dynamically balances **local numerical optimization** with **global algebraic solvers** to ensure zero root-omission, featuring both a GitHub Actions automated workflow pipeline and a robust, stateful local Terminal CLI.

---

## 🚀 Core Methodology

Finding all roots of high-degree polynomials reliably within an interval can be notoriously tricky for standard local iterative methods. This project implements a two-phase **Dynamic Node Selection (DNS)** decision framework to guarantee completeness:

### Phase 1: Local Bracketed Exploitation (Modified Secant)
The algorithm samples the target domain $[a, b]$ to locate sign changes. For every bracket found, it initiates a single-point **Modified Secant Method** using a finite-difference slope estimate:

$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$

This converges quickly to high-precision local roots.

### Phase 2: Global Algebraic Exploration (Chebyshev-Frobenius Companion Matrix)
As a fallback and verification engine, the script constructs a canonical **Frobenius Companion Matrix** directly from the polynomial coefficients. The eigenvalues of this matrix correspond exactly to the roots of the polynomial:

$$\det(C - \lambda I) = 0$$

By computing the matrix eigenvalues using `numpy.linalg.eigvals`, the framework uncovers roots that may not have triggered a simple sign-change bracket (e.g., closely spaced roots or grazing touches).

### The DNS Decision Rule
* **Perfect Match:** If Phase 1 isolates a number of brackets equal to the polynomial's `degree`, the Secant phase is declared complete. Frobenius eigenvalues are strictly used for cross-check validation.
* **Deficit Resolution:** If Phase 1 finds fewer brackets than the polynomial's `degree`, the DNS engine calculates the missing root deficit, filters out the duplicates, and pulls the missing roots directly from the **Frobenius global fallback array**.

---

## ✨ Key Features

* 💻 **Interactive Local CLI Playground:** Run custom configurations, modify boundaries, degrees, or tolerances on the fly, and view step-by-step internal solver logging.
* 📊 **Batch Benchmark Suite:** Instantly run a standardized performance test across multiple configurations ranging from degree 2 up to degree 8.
* 💾 **Smart CSV Exporting:** Automatically saves execution metadata (`Degree`, `Interval Bounds`, `Tolerance`, `Roots Found`, and `Root Values`) to structured CSV files for downstream analysis or CI/CD logging.

---

## 🛠️ Installation & Setup

### Prerequisites
Make sure you have Python 3.8 or higher installed on your system. 

### 1. Clone the Repository
bash
git clone [https://github.com/pinkTshirt/dynamic-node-selection.git](https://github.com/pinkTshirt/dynamic-node-selection.git)
cd dynamic-node-selection

### 2. Install dependencies
The project relies on numpy

pip install numpy

*Run the code on terminal as* : python main.py

### upon launching, the terminal will say:
==================================================
  DYNAMIC NODE SELECTION — INTERACTIVE PLAYGROUND
==================================================
  [1] Run Solver with Current Variables
  [2] Modify Parameters / Toggle Variables
  [3] Run Full Standard Benchmark Suite (5 configs)
  [4] Save Last Custom Run to Local CSV
  [5] Exit
--------------------------------------------------
  CURRENT STATE: Degree=4 | Domain=[-3.0, 3.0] | Tol=1e-08
==================================================
Select an action (1-5):

*select and run the program*
