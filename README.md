# MScThesis

In this repo I have collected the code developed for my master's thesis [Gradient-based quantum optimal control on
superconducting qubit systems](https://thesis.unipd.it/handle/20.500.12608/36024) developed jointly between the University of Padua and Forschungszentrum Jülich in the period from March to September 2022.
The code implements the methods shown in [Approximate Dynamics Lead to More Optimal Control: Efficient Exact Derivatives](https://arxiv.org/pdf/2005.09943.pdf) [[1](https://arxiv.org/pdf/2005.09943.pdf)] and [Achieving fast high-fidelity optimal control of many-body quantum dynamics](https://arxiv.org/abs/2008.06076) [[2](https://arxiv.org/abs/2008.06076)].


The work is dived as follows (files are listed in logical order, as reported in the thesis):

- `landau.ipynb` contains the code for the first simple example of a Landau-Zener transition as shown in [[1](https://arxiv.org/pdf/2005.09943.pdf)].
- `transmon_state.ipynb` implements the transmon system as in [[1](https://arxiv.org/pdf/2005.09943.pdf)]: the code realizes a CNOT gate on 2 neighboring qubits.
- `transmon_sys.ipynb` implements the transmon (qutrit) system: the code realizes a CNOT gate on neighbours qubits for an arbitrary system size (modulo computational feasibility) and evolves all basis states of the qubit subspace, with possibility of having local (on site) or global control(s).
- `jensen.py` contains the code to compute the fidelity, wall time and number of iteration for a scalable number of transmons over 10 different seeds. The script accepts parameters from the command line:
```
    -h, --help            show this help message and exit
    --dim DIM             number of sites
    --tgt_qubit_idx [TGT_QUBIT_IDX]
                            taget qubit index
    --T [T]               time window in SI-units (ns)
    --dt [DT]             time interval in non-dimensionalized numerical units
    --loc_ctrl            local control flag: if False, u^(j)_n=u_n
```
***
- `jensen_dense.ipynb` implements the code as in `transmon_sys.ipynb`, re-written in Julia for the dense (i.e. few qubits) case. At the end, a comparison with Krylov exponentiation is explored via the [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl) package. 
- `jensen_sparse.ipynb` implements the same code as in `jensen_dense.ipynb`, generalizing it to the sparse case.
- `LBFGSBcall.jl` implements the L-BFGS-B optimization routine, adapted from the [`LBFGSB.jl`](https://github.com/Gnimuc/LBFGSB.jl) library. The function stores during its evaluation values of interest (as e.g. the norm of the gradient and the evolution of the objective function).
- `TEBD.ipynb` implements Jensen's method for a many-body transmon system using Tensor Networks via Julia's [`ITensors`](https://github.com/ITensor/ITensors.jl) library and a particular version of the TEBD algorithm for MPS [[2](https://arxiv.org/abs/2008.06076)]. In the final part of the notebook automatic differentiation is explored. Performances can be further analyzed via the [`ProfileView.jl`](https://github.com/timholy/ProfileView.jl) library.
- `TEBD.jl` is the Julia script running the Tensor Networks simulations contained in `TEBD.ipynb`. Results are stored in the `TNresults` folder.
- `LBFGSBcallMPS.jl` implements the same routine of `LBFGSBcall.jl`, slightly modified to the case of MPS (it stores e.g. also the value of the Von Neumann entropy).
***
- `pyexpokit.py` implements a Python version of `expm` from Expokit, as showed [here](https://github.com/matteoacrossi/pyexpokit) [[3](https://github.com/matteoacrossi/pyexpokit)].
- `test_expokit.py` contains the routine to test the implementation of the matrix exponential `pyexpokit.py` as in [[3](https://github.com/matteoacrossi/pyexpokit)].
- `mps_ad_tfi.ipynb` computes the ground- and first excited states of a transverse field Ising model using automatic differentiation and a simple gradient descent routine. The code was developed by Niklas Tausendpfund.
- `quspin.ipynb` contains some unsuccessful trials using [QuSpin](https://github.com/QuSpin/QuSpin).
***
- `mpl2latex.py` contains some plotting routines largely based on M. Ballarin [work](https://github.com/mballarin97/Matplotlib2LaTeX).
- `scaling.ipynb` is the (messy) notebook containing most of the final plots of the thesis. Plot settings are taken from the [`SciencePlots`](https://github.com/garrettj403/SciencePlots) repository.

The scaling data and plots are stored in the folder of the same name.