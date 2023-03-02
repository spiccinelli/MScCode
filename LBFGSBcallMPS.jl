using LBFGSB: setulb
using LinearAlgebra

function LBFGSBcall(fun, ctrl, bound, nsites)
    nmax = 1024 # the dimension of the largest problem to be solved
    mmax = 17   # the maximum number of limited memory corrections

    task = fill(Cuchar(' '), 60)    # fortran's blank padding
    csave = fill(Cuchar(' '), 60)   # fortran's blank padding
    lsave = zeros(Cint, 4)
    isave = zeros(Cint, 44)
    dsave = zeros(Cdouble, 29)

    # suppress the default output
    iprint = -1

    # f/g is a DOUBLE PRECISION variable. If the routine setulb returns with task(1:2)= 'FG',
    # then f/g must be set by the user to contain the value of the function at the point x
    f, g = 0.0, 0.0

    wa = zeros(Cdouble, 2mmax*nmax + 5nmax + 11mmax*mmax + 8mmax)
    iwa = zeros(Cint, 3*nmax)

    # suppress both code-supplied stopping tests because we provide our own termination conditions
    factr = 0.0
    pgtol = 0.0

    x = copy(ctrl)   # copy initial guess, otherwise it gets modified in-place by the function call

    n = size(x,1)    # the dimension n of the sample problem
    m = 10           # the number of m of limited memory correction stored

    # provide nbd which defines the bounds on the variables:
    nbd = zeros(Cint, nmax)
    l = zeros(Cdouble, nmax)    # the lower bounds
    u = zeros(Cdouble, nmax)    # the upper bounds

    # set bounds on the variables
    for i = 1:n
        nbd[i] = 2
        l[i] = -bound
        u[i] =  bound
    end

    # start the iteration by initializing task
    task[1:5] = b"START"
    objfun = Vector{Float64}()              # stores the evolution of the objective function
    res = Vector{Float64}                   # stores the minimum found
    niter = 1                               # iteration number
    maxiter = 1000                          # max number of iterations

    SvN = Matrix{Float64}(undef, nsites-1, maxiter)
    recfide = Matrix{Float64}(undef, length(ctrl), maxiter)
    
    println("starting task ...")
    
    # ---------- start of the loop -------------

    @label CALLLBFGSB

    # call to the L-BFGS-B code
    setulb(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, task, iprint, csave, lsave, isave, dsave)

    # the minimization routine has returned to request the function f and gradient g values at the current x
    if task[1:2] == b"FG"
        f, g, S, fide = fun(x)
        # go back to the minimization routine
        @goto CALLLBFGSB
    end
    
    # the minimization routine has returned with a new iterate.
    # we test whether the following two stopping tests are satisfied:
    if task[1:5] == b"NEW_X"
        # 1. terminate if the total number of f and g evaluations exceeds 900
        isave[34] ≥ 900 && (task[1:4] = b"STOP"; terminate_info="TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT")

        # 2. terminate if  |proj g|/(1+|f|) < 1.0d-10
        dsave[13] ≤ 1e-10 * (1e0 + abs(f)) && (task[1:4] = b"STOP"; terminate_info="THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL")

        # 3. terminate for maxiter iterations of the algorithm
        niter ≥ maxiter && (task[1:4] = b"STOP"; terminate_info="REACHED MAX NR. OF ITERATIONS")

        print("running for iteration: ", niter, "\r")
        flush(stdout)

        push!(objfun, f)
        SvN[:,niter] = S
        recfide[:,niter] = fide
        niter += 1

        # if the run is to be terminated, print the information contained in task and save minimum
        if task[1:4] == b"STOP"
            res = x
            println("terminating task ...")
            println(terminate_info)
        end

        # go back to the minimization routine
        @goto CALLLBFGSB
    end

    # if the run is converged, print the information contained in task and save minimum
    if task[1:4] == b"CONV"
        res = x
        println("terminating task ...")
        println("CONVERGENCE REACHED")
    end
    
    # ---------- end of the loop -------------

    return objfun, res, SvN, recfide
end