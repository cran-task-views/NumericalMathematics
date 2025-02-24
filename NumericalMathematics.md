---
name: NumericalMathematics
topic: Numerical Mathematics
maintainer: Hans W. Borchers, Robin Hankin, Serguei Sokol
email: hwb@mailbox.org
version: 2024-07-27
source: https://github.com/cran-task-views/NumericalMathematics/
---


This task view on numerical mathematics lists R packages and functions
that are useful for solving numerical problems in linear algebra and
analysis. It shows that R is a viable computing environment for
implementing and applying numerical methods, also outside the realm of
statistics.

The task view will *not* cover differential equations, optimization problems and solvers,
or packages and functions operating on times series, because all these topics are treated
extensively in the corresponding task views
`r view("DifferentialEquations")`,
`r view("Optimization")`, and
`r view("TimeSeries")`.
All these task views together will provide a good selection of what is available in R
for the area of numerical mathematics. The `r view("HighPerformanceComputing")` task view
with its many links for parallel computing may also be of interest.

**Contents**

- [Numerical Linear Algebra](#numerical-linear-algebra)
- [Special Functions](#special-functions)
- [Polynomials](#polynomials)
- [Differentiation and Integration](#differentiation-and-integration)
- [Interpolation and Approximation](#interpolation-and-approximation)
- [Root Finding and Fixed Points](#root-finding-and-fixed-points)
- [Discrete Mathematics and Number Theory](#discrete-mathematics-and-number-theory)
- [Multiprecision and Symbolic Calculations](#multiprecision-and-symbolic-calculations)
- [Python and SAGE Interfaces](#python-and-sage-interfaces)
- [MATLAB Octave Julia and other Interfaces](#matlab-octave-julia-and-other-interfaces)

The task view has been created to provide an overview of the topic. If
some packages are missing or certain topics in numerical math should be
treated in more detail, please contact the maintainer, either via
e-mail or by submitting an issue or pull request in the GitHub
repository linked above.

### Numerical Linear Algebra

As statistics is based to a large extent on linear algebra, many
numerical linear algebra routines are present in R, and some only
implicitly. Examples of explicitly available functions are vector and
matrix operations, matrix (QR) decompositions, solving linear equations,
eigenvalues/-vectors, singular value decomposition, or least-squares
approximation.

-   The recommended package
    `r pkg("Matrix", priority = "core")` provides classes and
    methods for dense and sparse matrices and operations on them, for
    example Cholesky and Schur decomposition, matrix exponential, or
    norms and conditional numbers for sparse matrices.
-   Recommended package `r pkg("MASS")` adds generalized
    (Penrose) inverses and null spaces of matrices.
-   `r pkg("expm")` computes the exponential, logarithm, and
    square root of square matrices, but also powers of matrices or the
    Frechet derivative. `expm()` is to be preferred to the function with
    the same name in `r pkg("Matrix")`.
-   `r pkg("QZ")` provides generalized eigenvalues and eigenvectors
    using a generalized Schur decomposition (QZ decomposition).
    It is able to compute the exponential of a matrix quite efficiently.
-   `r pkg("EigenR")` with complex Linear Algebra functions for determinant,
    rank, inverse, pseudo-inverse, kernel and image, QR decomposition, Cholesky 
    decomposition, (complex) Schur decomposition, and Hessenberg decomposition.
-   `r pkg("geigen")` calculates Generalized Eigenvalues, the Generalized Schur
    Decomposition and the Generalized Singular Value Decomposition of a pair
    of (complex) matrices.
-   `r pkg("SparseM")` provides classes and methods for
    sparse matrices and for solving linear and least-squares problems in
    sparse linear algebra
-   `r pkg("rmumps")` provides a wrapper for the MUMPS library, 
    solving large linear systems of equations applying a sparse direct solver.
-   `r pkg("sanic")` supports routines for solving (dense
    and sparse) large systems of linear equations; direct and iterative
    solvers from the Eigen C++ library are made available, including
    Cholesky, LU, QR, and Krylov subspace methods.
-   `r pkg("Rlinsolve")` is a collection of iterative
    solvers for sparse linear system of equations; stationary iterative
    solvers such as Jacobi or Gauss-Seidel, as well as nonstationary
    (Krylov subspace) methods are provided.
-   `r pkg("svd")` provides R bindings to state-of-the-art
    implementations of singular value decomposition (SVD) and
    eigenvalue/eigenvector computations. Package `r pkg("irlba")`
    will compute approximate singular values/vectors of large matrices.
-   Package `r pkg("PRIMME")` interfaces PRIMME, a C library
    for computing eigenvalues and eigenvectors of real symmetric or
    complex Hermitian matrices. It will find largest, smallest, or
    interior eigen-/singular values and will apply preconditioning to
    accelerate convergence.
-   Package `r pkg("rARPACK")`, a wrapper for the ARPACK
    library, is typically used to compute only a few
    eigenvalues/vectors, e.g., a small number of largest eigenvalues.
-   Package `r pkg("RSpectra")` interfaces the 'Spectra'
    library for large-scale eigenvalue decomposition and SVD problems.
-   `r pkg("optR")` uses elementary methods of linear
    algebra (Gauss, LU, CGM, Cholesky) to solve linear systems.
-   Package `r pkg("mbend")` for bending
    non-positive-definite (symmetric) matrices to positive-definiteness,
    using weighted and unweighted methods.
-   `r pkg("matrixcalc")` contains a collection of functions
    for matrix calculations, special matrices, and tests for matrix
    properties, e.g., (semi-)positive definiteness; mainly used for
    teaching and research purposes
-   `r pkg("matlib")` contains a collection of matrix
    functions for teaching and learning matrix linear algebra as used in
    multivariate statistical methods; mainly for tutorial purposes in
    learning matrix algebra ideas using R.
-   Package `r pkg("onion")` contains routines for
    manipulating quaternions and octonians (normed division algebras
    over the real numbers); quaternions can be useful for handling
    rotations in three-dimensional space.
-   Packages `r pkg("RcppArmadillo")` and
    `r pkg("RcppEigen")` enable the integration of the C++
    template libraries 'Armadillo' resp. 'Eigen' for linear algebra
    applications written in C++ and integrated in R using
    `r pkg("Rcpp")` for performance and ease of use.

### Special Functions

Many special mathematical functions are present in R, especially
logarithms and exponentials, trigonometric and hyperbolic functions, or
Bessel and Gamma functions. Many more special functions are available in
contributed packages.

-   Package `r pkg("gsl")` provides an interface to the
    'GNU Scientific Library' that contains implementations of many
    special functions, for example the Airy and Bessel functions,
    elliptic and exponential integrals, the hypergeometric function,
    Lambert's W function, and many more.
    `r pkg("RcppGSL")` provides an easy-to-use interface between
    'GSL' data structures and R, using concepts from 'Rcpp'.
-   Airy and Bessel functions, for real and complex numbers, are also
    computed in package `r pkg("Bessel")`, with
    approximations for large arguments.
-   Package `r pkg("pracma")` includes
    special functions, such as error functions and inverses, incomplete
    and complex gamma function, exponential and logarithmic integrals,
    Fresnel integrals, the polygamma and the Dirichlet and Riemann zeta
    functions.
-   The hypergeometric (and generalized hypergeometric) function, is
    computed in `r pkg("hypergeo")`, including
    transformation formulas and special values of the parameters.
-   Elliptic and modular functions are available in package
    `r pkg("elliptic")`, e.g., the Weierstrass P function and Jacobi's
    theta functions. There are tools for visualizing complex functions.
-   `r pkg("jacobi")` evaluates Jacobi theta and related functions:
    Weierstrass elliptic functions, Weierstrass sigma and zeta function,
    Klein j-function, Dedekind eta function, lambda modular function, 
    Jacobi elliptic functions, Neville theta functions, Eisenstein series.
    Complex values of the variable are supported.
-   `r pkg("Carlson")` evaluates Carlson elliptic and
    incomplete elliptic integrals (with compex arguments).
-   Package `r pkg("expint")` wraps C-functions from the GNU
    Scientific Library to calculate exponential integrals and the
    incomplete Gamma function, including negative values for its first
    argument.
-   `r pkg("fourierin")` computes Fourier integrals of
    functions of one and two variables using the Fast Fourier Transform.
-   `r pkg("logOfGamma")` uses approximations to compute the
    natural logarithms of the Gamma function for large values.
-   Package `r pkg("lamW")` implements both real-valued
    branches of the Lambert W function (using Rcpp).
-   `r pkg("flint")`; see section 'Discrete Mathematics and Number
    Theory'.

### Polynomials

Function polyroot() in base R determines all zeros of a polynomial,
based on the Jenkins-Traub algorithm. Linear regression function lm()
can perform polynomial fitting when using `poly()` in the model formula
(with option `raw = TRUE`).

-   Packages `r pkg("PolynomF", priority = "core")`
    (recommended) and `r pkg("polynom")` provide similar
    functionality for manipulating univariate polynomials, like
    evaluating polynomials (Horner scheme), or finding their roots.
    'PolynomF' generates orthogonal polynomials and provides graphical
    display features.
-   `r pkg("polyMatrix")` (based on 'polynom') implements
    basic matrix operations and provides thus an infrastructure for the
    manipulation of polynomial matrices.
-   Package `r pkg("MonoPoly")` fits univariate polynomials
    to given data, applying different algorithms.
-   For multivariate polynomials, package
    `r pkg("multipol")` provides various tools to manipulate
    and combine these polynomials of several variables.
-   Package `r pkg("mpoly")` facilitates symbolic
    manipulations on multivariate polynomials, including basic
    differential calculus operations on polynomials.
-   `r pkg("mvp")` enables fast manipulation of symbolic
    multivariate polynomials, using print and coercion methods from the
    'mpoly' package, but offers speed improvements.
-   Package `r pkg("orthopolynom")` consists of a collection
    of functions to construct orthogonal polynomials and their
    recurrence relations, among them Chebyshev, Hermite, and Legendre
    polynomials, as well as spherical and ultraspherical polynomials.
-   Symbolic calculation and evaluation of the Jack polynomials, zonal
    and Schur polynomials are available in package `r pkg("jack")`.
-   The Free Algebra in R package `r pkg("freealg")` handles
    multivariate polynomials with non-commuting indeterminates.
-   Package `r pkg("spray")` provides functionality for sparse
    multidimensional arrays, interpreted as multivariate polynomials.
-   Package `r pkg("qspray")` provides symbolic calculation and
    evaluation of multivariate polynomials with rational coefficients,
    plus some Groebner basis calculations.
-   Package `r pkg("minimaxApprox")` implements the algorithm of Remez (1962)
    for polynomial minimax approximation and that of Cody et al. (1968)
    for rational minimax approximations.
-   `r pkg("flint")`; see section 'Discrete Mathematics and Number
    Theory'.

### Differentiation and Integration

`D()` and `deriv()` in base R compute derivatives of simple expressions
symbolically. Function `integrate()` implements an approach for
numerically integrating univariate functions in R. It applies adaptive
Gauss-Kronrod quadrature and can handle singularities and unbounded
domains to a certain extent.

-   Package `r pkg("Deriv")` provides an
    extended solution for symbolic differentiation in R; the user can
    add custom derivative rules, and the output for a function will be
    an executable function again.
-   `r pkg("numDeriv", priority = "core")` sets the standard
    for numerical differentiation in R, providing numerical gradients,
    Jacobians, and Hessians, computed by simple finite differences,
    Richardson extrapolation, or the highly accurate complex step
    approach.
-   Package `r pkg("dual")` achieves automatic
    differentiation (for univariate functions) by employing dual
    numbers; for a mathematical function its value and its exact first
    derivative are returned.
-   Package `r github("Non-Contradiction/autodiffr")` (on Github)
    provides an R wrapper for the Julia packages ForwardDiff.jl
    and ReverseDiff.jl to do automatic differentiation for native
    R functions.
-   `r pkg("pracma")` contains functions for computing
    numerical derivatives, including Richardson extrapolation or complex
    step. `fderiv()` computes numerical derivatives of higher orders.
    `r pkg("pracma")` also has several routines for numerical
    integration: adaptive Lobatto quadrature, Romberg integration,
    Newton-Cotes formulas, Clenshaw-Curtis quadrature rules.
    `integral2()` integrates functions in two dimensions, also for
    polar coordinates or domains with variable interval limits.
-   `r pkg("cubature", priority = "core")` is a package for adaptive 
    multivariate integration over hypercubes in n-dimensional space, based 
    on the C-library 'cubature', resp. for deterministic and Monte-Carlo 
    integration based on library 'Cuba'. Function 'cubintegrate()' wraps 
    all the integration methods provided.
-   Package `r pkg("gaussquad")` contains a collection of
    functions to perform Gaussian quadrature, among them Chebyshev,
    Hermite, Laguerre, and Legendre quadrature rules, explicitly
    returning nodes and weights in each case. Function `gaussquad()` in
    package `r pkg("statmod")` does a similar job.
-   `r pkg("GramQuad")` allows for numerical integration
    based on Gram polynomials.
-   Package `r pkg("fastGHQuad")` provides a fast `r pkg("Rcpp")`-based 
    implementation of (adaptive) Gauss-Hermite quadrature.
-   `r pkg("mvQuad")` provides methods for generating
    multivariate grids that can be used for multivariate integration.
    These grids will be based on different quadrature rules such as
    Newton-Cotes or Gauss quadrature formulas.
-   Package `r pkg("SparseGrid")` provides another approach
    to multivariate integration in high-dimensional spaces. It creates
    sparse n-dimensional grids that can be used as with quadrature
    rules.
-   Package `r pkg("SphericalCubature")` employs
    `r pkg("cubature")` to integrate functions over unit
    spheres and balls in n-dimensional space;
    `r pkg("SimplicialCubature")` provides methods to
    integrate functions over m-dimensional simplices in n-dimensional
    space. Both packages comprise exact methods for polynomials.
-   Package `r pkg("polyCub")` holds some routines for
    numerical integration over polygonal domains in two dimensions.
-   Package `r pkg("Pade")` calculates the numerator and
    denominator coefficients of the Pade approximation, given the Taylor
    series coefficients of sufficient length.
-   `r pkg("calculus")` provides efficient functions for
    high-dimensional numerical and symbolic calculus, including accurate
    higher-order derivatives, Taylor series expansion, differential
    operators, and Monte-Carlo integration in orthogonal coordinate
    systems.
-   `r pkg("features")` extracts features from functional
    data, such as first and second derivatives, or curvature at critical
    points, while `r pkg("RootsExtremaInflections")` finds
    roots, extrema and inflection points of curves defined by discrete
    points.

### Interpolation and Approximation

Base R provides functions `approx()` for constant and linear
interpolation, and `spline()` for cubic (Hermite) spline interpolation,
while `smooth.spline()` performs cubic spline approximation. Base
package splines creates periodic interpolation splines in function
`periodicSpline()`.

-   Interpolation of irregularly spaced data is possible with the
    `r pkg("akima")` package: `aspline()` for univariate
    data, `bicubic()` or `interp()` for data on a 2D rectangular domain.
    (This package is distributed under ACM license and not available for
    commercial use.)
-   Package `r pkg("signal")` contains several *filters* to
    smooth discrete data, notably `interp1()` for linear, spline, and
    cubic interpolation, `pchip()` for piecewise cubic Hermite
    interpolation, and `sgolay()` for Savitzky-Golay smoothing.
-   Package `r pkg("pracma")` provides barycentric Lagrange
    interpolation (in 1 and 2 dimensions) in `barylag()` resp.
    `barylag2d()`, 1-dim. akima in `akimaInterp()`, and interpolation
    and approximation of data with rational functions, i.e. in the
    presence of singularities, in `ratinterp()` and `rationalfit()`.
-   The `r pkg("interp")` package provides bivariate data
    interpolation on regular and irregular grids, either linear or using
    splines. Currently the piecewise linear interpolation part is
    implemented. (It is intended to provide a free replacement for the
    ACM licensed `akima::interp` and `tripack::tri.mesh` functions.)
-   Package `r pkg("splines2")` provides basis matrices of B-splines,
    M-splines, I-splines, convex splines (C-splines), periodic
    splines, natural cubic splines, generalized Bernstein polynomials,
    and their integrals (except C-splines) and derivatives by
    closed-form recursive formulas.
-   `r pkg("bspline")` uses B-splines for creating functions interpolating
    and smooting 1D data. `fitsmbsp()` can optimize knot positions and
    impose monotonicity and positivity constraints. Produced functions
    can be differentiated with `dbsp()` or integrated with `ibsp()`.
-   `r pkg("tripack")` for triangulation of irregularly
    spaced data is a constrained two-dimensional Delaunay triangulation
    package providing both triangulation and generation of Voronoi
    mosaics of irregular spaced data.
-   `sinterp()` in package `r pkg("stinepack")` realizes
    interpolation based on piecewise rational functions by applying
    Stineman's algorithm. The interpolating function will be monotone
    in regions where the specified points change monotonically.
-   `Schumaker()` in package `r pkg("schumaker")` implements
    shape-preserving splines, guaranteed to be monotonic resp. concave
    or convex if the data is monotonic, concave, or convex.
-   `r pkg("ADPF")` uses least-squares polynomial regression
    and statistical testing to improve Savitzky-Golay smoothing.
-   Package `r pkg("conicfit")` provides several (geometric
    and algebraic) algorithms for fitting circles, ellipses, and conics
    in general.

### Root Finding and Fixed Points

`uniroot()`, implementing the Brent-Decker algorithm, is the basic
routine in R to find roots of univariate functions. There are
implementations of the bisection algorithm in several contributed
packages. For root finding with higher precision there is function
`unirootR()` in the multi-precision package
`r pkg("Rmpfr")`. For finding roots of univariate and multivariate
functions see the following packages:

-   Package `r pkg("itp")` implements the Interpolate, Truncate, Project
    (ITP) root-finding algorithm. The user provides a univariate (1-dim.)
    function and the endpoints of an interval where the function values
    have different signs.
-   Package `r pkg("rootSolve")` includes function
    `multiroot()` for finding roots of systems of nonlinear (and linear)
    equations; it also contains an extension `uniroot.all()` that
    attempts to find all zeros of a univariate function in an intervall
    (excepting quadratic zeros).
-   For solving nonlinear systems of equations the
    `r pkg("BB")` package provides Barzilai-Borwein spectral
    methods in `sane()`, including a derivative-free variant in
    `dfsane()`, and multi-start features with sensitivity analysis.
-   Package `r pkg("nleqslv")` solves nonlinear systems of
    equations using alternatively the Broyden or Newton method,
    supported by strategies such as line searches or trust regions.
-   `r pkg("ktsolve")` defines a common interface for
    solving a set of equations with `BB` or `nleqslv`.
-   `r pkg("FixedPoint")` provides algorithms for finding
    fixed point vectors of functions, including Anderson acceleration,
    epsilon extrapolation methods, or minimal polynomial methods .
-   Package `r pkg("daarem")` implements the DAAREM method
    for accelerating the convergence of any smooth, monotone, slow fixed
    point iteration.
-   Algorithms for accelerating the convergence of slow, monotone
    sequences from smooth contraction mappings such as the
    expectation-maximization (EM) algorithm are provided in packages
    `r pkg("SQUAREM")` resp. `r pkg("turboEM")`.

### Discrete Mathematics and Number Theory

Not so many functions are available for computational number theory.
Note that integers in double precision can be represented exactly up to
`2^53 - 1`, above that limit a multi-precision package such as
`r pkg("gmp")` is needed, see below.

-   Package `r pkg("numbers")` provides functions for
    factorization, prime numbers, twin primes, primitive roots, modular
    inverses, extended GCD, etc. Included are some number-theoretic
    functions like divisor functions or Euler's Phi function.
-   `r pkg("contfrac")` contains various utilities for evaluating
    continued fractions and partial convergents.
    `r pkg("contFracR")`Converts numbers to continued fractions and back again.
    A solver for Pell's Equation is provided.
-   `r pkg("magic")` creates and investigates magical
    squares and hypercubes, including functions for the manipulation and
    analysis of arbitrarily dimensioned arrays.
-   Package `r pkg("freegroup")` provides functionality for
    manipulating elements of a free group including juxtaposition,
    inversion, multiplication by a scalar, power operations, and Tietze
    forms.
-   The `r pkg("partitions")` package enumerates additive
    partitions of integers, including restricted and unequal partitions.
-   `r pkg("permutations")` treats permutations as
    invertible functions of finite sets and includes several
    mathematical operations on them.
-   Package `r pkg("combinat")` generates all permutations
    or all combinations of a certain length of a set of elements (i.e. a
    vector); it also computes binomial coefficients.
-   Package `r pkg("arrangements")` provides generators and
    iterators for permutations, combinations and partitions. The
    iterators allow users to generate arrangements in a fast and memory
    efficient manner. Permutations and combinations can be drawn
    with/without replacement and support multisets.
-   Package `r github("xoopR/set6")` (on Github) implements (as R6 classes)
    many forms of mathematical sets (sets, tuples, intervals) and allows
    for standard operations on them (unions, products, differences, etc.).
-   `r pkg("RcppAlgos")` provides flexible functions for
    generating combinations or permutations of a vector with or without
    constraints; the extension package
    `r pkg("RcppBigIntAlgos")` features a quadratic sieve
    algorithm for completely factoring large integers.
-   Package `r pkg("Zseq")` generates well-known integer
    sequences; the 'gmp' package is adopted for computing with
    arbitrarily large numbers. Every function has on its help page a
    hyperlink to the corresponding entry in the On-Line Encyclopedia of
    Integer Sequences ( [OEIS](https://oeis.org/) ).
-   Package `r pkg("primes")` provides quite fast (Rcpp)
    functions for identifying and generating prime numbers. And
    `r pkg("primefactr")` uses prime factorization for
    computations such as reducing ratios of large factorials.
-   Package `r pkg("frab")` provides methods to "add" two tables
    with the free Abelian group as the underlying structure.
-   `r pkg("flint")` is an R interface to FLINT
    (<https://flintlib.org/>), a C library for number theory.  FLINT
    provides C types and functions for arbitrary precision
    representation of and operations on standard rings (the integers,
    the integers modulo *n*, the rational, *p*-adic, real, and complex
    numbers) as well as vectors, matrices, polynomials, and power
    series over rings.  FLINT implements midpoint-interval (or "ball")
    arithmetic in the real and complex numbers, enabling computation
    in arbitrary precision with rigorous propagation of errors.  FLINT
    provides ball arithmetic implementations of many special
    mathematical functions, with high coverage of reference works such
    as the NIST Digital Library of Mathematical Functions
    (<https://dlmf.nist.gov/>).  The R interface is incomplete and
    extended "as needed"; users wanting an R interface to a certain C
    type or function are encouraged to submit a request using
    `bug.report(package = "flint")`.

### Multiprecision and Symbolic Calculations

-   Multiple precision arithmetic is available in R through package
    `r pkg("gmp")` that interfaces to the GMP C library.
    Examples are factorization of integers, a probabilistic prime number
    test, or operations on big rationals \-- for which linear systems of
    equations can be solved.
-   Multiple precision floating point operations and functions are
    provided through package `r pkg("Rmpfr")` using the MPFR
    and GMP libraries. Special numbers and some special functions are
    included, as well as routines for root finding, integration, and
    optimization in arbitrary precision.
-   `r pkg("Brobdingnag")` handles very large numbers by
    holding their logarithm plus a flag indicating their sign. (An
    excellent vignette explains how this is done using S4 methods.)
-   `r pkg("VeryLargeIntegers")` implements a
    multi-precision library that allows to store and manage arbitrarily
    big integers; it includes probabilistic primality tests and
    factorization algorithms.
-   `r pkg("bignum")` is a package for arbitrary-precision
    integer and floating-point numbers of 50 decimal digits of
    precision. The package utilizes the 'Boost.Multiprecision' C++
    library and is specifically designed to work with the 'tidyverse'
    collection of R packages.
-   Package `r pkg("Ryacas")` interfaces the computer
    algebra system 'Yacas'; it supports symbolic and arbitrary
    precision computations in calculus and linear algebra.
-   Package `r pkg("caracas")` (based on 'reticulate')
    accesses the symbolic algebra system 'SymPy'; supported are
    symbolic operations in linear algebra and calculus, such as
    eigenvalues, derivatives, integrals, limits, etc., computing special
    functions, or solving systems of equations.
-   Package `r pkg("symengine")` provides an interface to
    'SymEngine', a C++ library for fast symbolic calculations, such as
    manipulating mathematical expressions, finding exact derivatives,
    performing symbolic matrix computations, or solving ordinary
    differential equations (numerically).
-   Package `r pkg("rim")` provides an interface to the free and
    powerful computer algebra system 
    [Maxima](https://maxima.sourceforge.io/).
    Results can be output in LaTeX or MathML, and 2D and 3D plots will be 
    displayed directly. Maxima code chunks can be included in 'RMarkdown' 
    documents.
-   Package `r pkg("m2r")` provides a persistent interface to
    [Macauley2](http://www2.macaulay2.com/Macaulay2/), 
    an extended software program supporting research in algebraic
    geometry and commutative algebra. Macauley2 has to be installed
    independently, otherwise a Macauley2 process in the cloud will be
    instantiated.
-   There are several packages for working with algebras over the real numbers.
    `r pkg("clifford")` provides a suite of routines for
    arbitrary dimensional Clifford algebras and discusses special cases
    such as Lorentz transforms or quaternion multiplication.  Package
    `r pkg("weyl")` provides functionality for Weyl algebras, which have
    applications in quantum mechanics. Package `r pkg("stokes")` works
    with the algebra of diffrential $k$-forms as used in exterior calculus
    (these packages all use `r pkg("disordR")` discipline).  Package
    `r pkg("jordan")` provides functionality for working with Jordan algebras,
    which are commutative but non-associative algebras that obey the Jordan
    identity $(xy)xx=x(yxx)$.
-   `r pkg("flint")`; see section 'Discrete Mathematics and Number
    Theory'.

### Python and SAGE Interfaces

Python, through its modules 'NumPy', 'SciPy', 'Matplotlib',
'SymPy', and 'pandas', has elaborate and efficient numerical and
graphical tools available.

-   `r pkg("reticulate")` is an R interface to Python
    modules, classes, and functions. When calling Python in R data types
    are automatically converted to their equivalent Python types; when
    values are returned from Python to R they are converted back to R
    types. This package from the RStudio team is a kind of standard for
    calling Python from R.
-   `r pkg("feather")` provides bindings to read and write
    feather files, a lightweight binary data store designed for maximum
    speed. This storage format can also be accessed in Python, Julia, or
    Scala.
-   'pyRserve' is a Python module for connecting Python to an R
    process running `r pkg("Rserve")` as an RPC gateway.
    This R process can run on a remote machine, variable access and
    function calls will be delegated through the network.
-   `r pkg("XRPython")` (and 'XRJulia') are based on John
    Chambers' `r pkg("XR")` package and his "Extending R"
    book and allow for a structured integration of R with Python resp.
    Julia.

[SageMath](http://www.sagemath.org/) is an open source mathematics
system based on Python, allowing to run R functions, but also providing
access to systems like Maxima, GAP, FLINT, and many more math programs.
SageMath can be freely used through a Web interface at
[CoCalc](https://cocalc.com/) .

### MATLAB Octave Julia and other Interfaces

Interfaces to numerical computation software such as MATLAB (commercial)
or Octave (free) will be important when solving difficult numerical
problems. Unfortunately, at the moment there is no package allowing to
call Octave functions from within R.

-   The `r pkg("matlab")` emulation package contains about
    30 simple functions, replicating MATLAB functions, using the
    respective MATLAB names and being implemented in pure R.
-   Packages `r pkg("rmatio")` and
    `r pkg("R.matlab")` provides tools to read and write MAT
    files (the MATLAB data format) for versions 4 and 5. 'R.matlab'
    also enables a one-directional interface with a MATLAB v6 process,
    sending and retrieving objects through a TCP connection.

Julia is "a high-level, high-performance dynamic programming language
for numerical computing", which makes it interesting for optimization
problems and other demanding scientific computations in R.

-   `r pkg("JuliaCall")` provides seamless integration
    between R and Julia; the user can call Julia functions just like any
    R function, and R functions can be called in the Julia environment,
    both with reasonable automatic type conversion. [Notes on Julia
    Call](https://hwborchers.github.io/) provides an introduction of how
    to apply Julia functions with 'JuliaCall'.
-   `r pkg("JuliaConnectoR")` provides a functionally
    oriented interface for integrating Julia with R; imported Julia
    functions can be called as R functions; data structures are
    converted automatically.
-   Package `r pkg("XRJulia")` provides an interface from R
    to computations in the Julia language, based on the interface
    structure described in the book "Extending R" by John M. Chambers.

Java Math functions can be employed through the 'rjava' or 'rscala'
interfaces. Then package `r pkg("commonsMath")` allows calling Java 
JAR files of the Apache Commons Mathematics Library, a specialized library 
for all aspects of numerics, optimization, and differential equations.

Please note that commercial programs such as MATLAB, Maple, or
Mathematica have facilities to call R functions.


### Links
-   Textbook: [Hands-On Matrix Algebra Using R](http://www.worldscientific.com/worldscibooks/10.1142/7814)
-   Textbook: [Introduction to Scientific Programming and Simulation Using R](https://www.routledge.com/Introduction-to-Scientific-Programming-and-Simulation-Using-R/Jones-Maillardet-Robinson/p/book/9781466569997)
-   Textbook: [Numerical Methods in Science and Engineering Using R](https://www.routledge.com/Using-R-for-Numerical-Analysis-in-Science-and-Engineering/Bloomfield/p/book/9781439884485)
-   Textbook: [Computational Methods for Numerical Analysis with R](https://www.crcpress.com/Computational-Methods-for-Numerical-Analysis-with-R/II/p/book/9781498723633)
-   [MATLAB / R Reference (D. Hiebeler)](https://umaine.edu/mathematics/david-hiebeler/computing-software/matlab-r-reference/)
-   [Abramowitz and Stegun. Handbook of Mathematical Functions](http://people.math.sfu.ca/~cbm/aands/)
-   [Numerical Recipes: The Art of Numerical Computing](http://numerical.recipes/)
-   [E. Weisstein's Wolfram MathWorld](http://mathworld.wolfram.com/)
