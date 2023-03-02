import numpy as np
from numpy import array, ndarray
from numpy.linalg import norm
from scipy.sparse import dia_matrix, kron, identity
from scipy.sparse.linalg import expm
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from itertools import product
from functools import reduce
from scipy.optimize import fmin_l_bfgs_b
import argparse
from time import perf_counter

### simulation initialization ###

parser = argparse.ArgumentParser(description='jensen approximate dynamics simulation')
parser.add_argument('--dim', type=int, help='number of sites')
parser.add_argument('--tgt_qubit_idx', nargs='?', default=2, type=int, help='taget qubit index')
parser.add_argument('--T', nargs='?', default=60, type=float, help='time window in SI-units (ns)')
parser.add_argument('--dt', nargs='?', default=0.015, type=float, help='time interval in non-dimensionalized numerical units')
parser.add_argument('--loc_ctrl', default=False, action='store_true', help='local control flag: if False, u^(j)_n=u_n')

args = parser.parse_args()

# simulation parameters

# global dimension
dim = args.dim
# local dimension
d = 3
# Hilbert space dimension
D_H = d**dim
# subspace dimension for the gate transfer
subdim = 2**dim
# target qubit index
tgt_qubit_idx = args.tgt_qubit_idx - 1
# control qubit index
ctrl_qubit_idx = tgt_qubit_idx - 1

# target qubit cannot be greater than the system dimension
if tgt_qubit_idx > dim - 1:
    raise ValueError('target qubit index must be less than the number of sites.')

# time parameters
T = args.T
dt = args.dt

# rough estimation of the N_t parameter to catch bad grid sampling
if int(T / dt) <= 100:
    raise ValueError('ratio T / dt must be > 100:1')

# local/global controls
loc_ctrl = args.loc_ctrl

# Hamiltonian parameters
twopi = 2 * np.pi
omegas = np.array([4.918, 5.003, 4.866, 5.104, 4.902, 5.04, 4.973, 5.033, 4.767, 4.87, 4.999, 5.077, 5.198, 4.994, 5.083])[:dim] * twopi
deltas = np.array([-0.3091, -0.30763, -0.31125, -0.30768, -0.30965, -0.30739, -0.30076, -0.30851, -0.30398, -0.31089, -0.30886, -0.30693, -0.30578, -0.30757, -0.30708])[:dim] * twopi
omega_r = 7.5 * twopi
g_1 = 0.1 * twopi
Deltas = omegas - omega_r
omegas_tilde = omegas + (g_1**2 / Deltas)
Js = array([(g_1**2 * (Deltas[n] + Deltas[n+1])) / (Deltas[n] * Deltas[n+1]) for n in range(dim-1)])
Deltas = array([omegas_tilde[n+1] - omegas_tilde[n] for n in range(dim-1)])

# scaling constants
e_unit = np.abs(max(Js))
t_unit = 1 / e_unit
eps = np.finfo(float).eps

# re-scale Hamiltonian constants
Deltas /= e_unit
deltas /= e_unit
Js /= e_unit

# target qubit has zero detuning
Deltas = np.insert(Deltas, tgt_qubit_idx, 0)

### time grid and controls ###

def input_parameters(T: float, dt: float):
    T = T / t_unit
    Nt = T / dt
    res = Nt % 1
    Nt = int(Nt)
    N = Nt + 1

    print('eff time window:\t', np.round(T - res * dt,2))
    print('nr time steps:\t\t', Nt)

    return N, dt

N, delta_t = input_parameters(T=T, dt=dt)

N_trap = N
N_rect = N - 1

# initialize the control vector
def initialize_controls(n_iter: int, loc_ctrl: bool) -> ndarray:
    u_thr = 0.2 * twopi / e_unit # 200 MHz
    if loc_ctrl:
        n_ctrl = dim
    else:
        n_ctrl = 1
    return np.random.uniform(-u_thr, u_thr, size=(n_iter, n_ctrl))

# filters out values 100 * machine precision
filter_eps = lambda x: np.where(np.abs(x) > 100 * eps, x, 0.)

### Hamiltonian ###

# creation and annihilation operators
def annihilation_sparse(dimension: int) -> csr_matrix:
    offdiag_elements = np.sqrt(range(dimension))
    return dia_matrix((offdiag_elements, [1]), shape=(dimension, dimension)).tocsr()
def creation_sparse(dimension: int) -> csr_matrix:
    return annihilation_sparse(dimension).transpose().tocsr()

bdgb = creation_sparse(d) @ annihilation_sparse(d)
b = annihilation_sparse(d)
bdg = creation_sparse(d)

# wrapper to store the matrices in the correct format
Id = lambda x: identity(x, format='csr')
Kron = lambda x,y: kron(x, y, format='csr')

# compute operators
def op_train(idxs, args):
    train = np.repeat(Id(d), dim)
    for idx, arg in zip(idxs, args):
        train[idx] = arg
    return reduce(Kron, train)

# compute eigenvalue/vectors in dense representation, controls are on all qubits
Hc = np.sum([op_train([n], [b + bdg]) for n in range(dim)]).toarray()
e, R = eigh(Hc)
e, R = filter_eps(e), filter_eps(R)

# convert back to sparse representation, build rotation function
R = csr_matrix(R)
rotate = lambda x: R.T @ x @ R

# save only diagonal elements
Hc = e

# drift Hamiltonian before rotation
drift1 = np.sum([Deltas[n] * op_train([n], [bdgb]) for n in range(dim)])
drift2 = np.sum([1/2 * deltas[n] * op_train([n], [bdgb @ (bdgb - Id(d))]) for n in range(dim)])
drift3 = np.sum([Js[n] * (op_train([n, n+1], [bdg, b]) + op_train([n, n+1], [b, bdg])) for n in range(dim-1)])

Hd = (drift1 + drift2 + drift3)
print('before rotation: ', 100 * Hd.nnz / np.product(Hd.shape))

# drift Hamiltonian after rotation
drift1 = np.sum([Deltas[n] * rotate(op_train([n], [bdgb])) for n in range(dim)])
drift2 = np.sum([1/2 * deltas[n] * rotate(op_train([n], [bdgb @ (bdgb - Id(d))])) for n in range(dim)])
drift3 = np.sum([Js[n] * (rotate(op_train([n, n+1], [bdg, b]) + op_train([n, n+1], [b, bdg]))) for n in range(dim-1)])

Hd = (drift1 + drift2 + drift3)
print('after rotation: ', 100 * Hd.multiply(np.abs(Hd) > 100 * eps).nnz / np.product(Hd.shape))

# exponential of drift Hamiltonian is computed once: sparse is here not convenient
Hd = Hd.toarray()
expHd = expm(-1j * Hd * delta_t)
expHdconj = np.conj(expHd.T)

# compute (rotated) sub-space basis states
def state_train(rep):
    kets = array([array([1,0,0]), array([0,1,0])])
    state_list = array([kets[n] for n in rep])
    return (R.T.toarray() @ reduce(np.kron, state_list))

# create all possible combinations of states
reps = list(product([0,1], repeat=dim))

# initialize and construct initial and target states
ini_states = np.zeros((subdim, D_H), dtype=complex)
tgt_states = np.zeros((subdim, D_H), dtype=complex)

for idx, rep in enumerate(reps):
    rep = list(rep)
    ini_states[idx] = state_train(rep)
    # flip target qubit if CNOT
    if rep[ctrl_qubit_idx] == 1:
        rep[tgt_qubit_idx] = (rep[tgt_qubit_idx] + 1) % 2
    tgt_states[idx] = state_train(rep)

### integration rules ###

class trapezoid:

    def __init__(self, ctrl):
        self.ctrl = ctrl                                                                # controls
        self.N = self.ctrl.shape[0]                                                     # nr of evolutions = nr controls = Nt + 1 = N
        self.D_H = D_H                                                                  # local dimension
        self.subdim = subdim                                                            # sub-dimension of qubit subspace
        self.delta_t = delta_t                                                          # time step
        self.evolved_psi, self.evolved_chi = self.time_evolution()                      # evoluted wave functions
        self.overlap, self.coverlap = self.transfer_amplitude()                         # overlap, conjugated overlap
        
    def time_evolution(self):

        evolved_psi = np.zeros((self.N, self.subdim, self.D_H), dtype=complex)          # psi evolution vector
        evolved_chi = np.zeros((self.N, self.subdim, self.D_H), dtype=complex)          # chi evolution vector
        trotter = np.zeros((self.N-1, 2, self.D_H), dtype=complex)                      # trotterized operators

        evolved_psi[0]  = ini_states                                                    # initialize evolutions
        evolved_chi[-1] = tgt_states

        for n in range(1, self.N):                                                      # iterate over the control vector
            Hc_next = np.sum(self.ctrl[n]) * Hc
            Hc_curr = np.sum(self.ctrl[n-1]) * Hc

            a = np.exp(-1j * Hc_next * self.delta_t / 2)
            c = np.exp(-1j * Hc_curr * self.delta_t / 2)
            
            trotter[n-1] = a, c                                                         # save operator
            evolve_psi = lambda x: a * (expHd @ (c * x))                                # build trotterized propagator
            evolved_psi[n] = array([evolve_psi(psi) for psi in evolved_psi[n-1]])       # evolve psi
            evolved_psi[n] = array([psi / norm(psi) for psi in evolved_psi[n]])         # impose normalization
            
        for n in range(self.N-1, 0, -1):
            a, c = np.conj(trotter[n-1])
            evolve_chi = lambda x: c * (expHdconj @ (a * x))                            # build trotterized propagator
            evolved_chi[n-1] = array([evolve_chi(chi) for chi in evolved_chi[n]])       # evolve chi
            evolved_chi[n-1] = array([chi / norm(chi) for chi in evolved_chi[n-1]])     # impose normalization
        
        return evolved_psi, evolved_chi

    def transfer_amplitude(self):
        over = array([np.dot(np.conj(chi), psi) for chi, psi in zip(self.evolved_chi[-1], self.evolved_psi[-1])])
        return over, np.conj(over)                                                      # transfer amplitude

    def F(self):
        return (np.abs(1/self.subdim * np.sum(self.overlap)))**2                        # fidelity

    def gradient(self):
        theta = np.conj(np.sum(self.overlap))
        g = np.zeros(self.N)
        for k in range(self.subdim):
            sandwich = array([np.conj(self.evolved_chi[n][k]) @ (Hc * self.evolved_psi[n][k]) for n in range(self.N)])
            sandwich[0]  *= 1/2
            sandwich[-1] *= 1/2
            g += np.real(1j * theta * sandwich)
        return g * (self.delta_t/self.subdim**2)                                        # gradient

class rectangle:

    def __init__(self, ctrl):
        self.ctrl = ctrl                                                                # controls
        self.N = self.ctrl.shape[0] + 1                                                 # nr of evolutions = nr of controls + 1 = Nt + 1 = N
        self.D_H = D_H                                                                  # local dimension
        self.subdim = subdim                                                            # sub-dimension of qubit subspace
        self.delta_t = delta_t                                                          # time step
        self.evolved_psi, self.evolved_chi = self.time_evolution()                      # evoluted wave functions
        self.overlap, self.coverlap = self.transfer_amplitude()                         # overlap, conjugated overlap
        
    def time_evolution(self):

        evolved_psi = np.zeros((self.N, self.subdim, self.D_H), dtype=complex)          # psi evolution vector
        evolved_chi = np.zeros((self.N, self.subdim, self.D_H), dtype=complex)          # chi evolution vector
        trotter = np.zeros((self.N-1, self.D_H), dtype=complex)                         # trotterized operators

        evolved_psi[0] = ini_states                                                     # initialize evolutions
        evolved_chi[-1] = tgt_states

        for n in range(1, self.N):                                                      # iterate over the control vector
            Hc_curr = np.sum(self.ctrl[n-1]) * Hc
            a = np.exp(-1j * Hc_curr * self.delta_t / 2)
            
            trotter[n-1] = a                                                            # save operator
            evolve_psi = lambda x: a * (expHd @ (a * x))                                # build trotterized propagator
            evolved_psi[n] = array([evolve_psi(psi) for psi in evolved_psi[n-1]])       # evolve psi
            evolved_psi[n] = array([psi / norm(psi) for psi in evolved_psi[n]])         # impose normalization
            
        for n in range(self.N-1, 0, -1):
            a = np.conj(trotter[n-1])
            evolve_chi = lambda x: a * (expHdconj @ (a * x))                            # build trotterized propagator
            evolved_chi[n-1] = array([evolve_chi(chi) for chi in evolved_chi[n]])       # evolve chi
            evolved_chi[n-1] = array([chi / norm(chi) for chi in evolved_chi[n-1]])     # impose normalization
        
        return evolved_psi, evolved_chi

    def transfer_amplitude(self):
        over = array([np.dot(np.conj(chi), psi) for chi, psi in zip(self.evolved_chi[-1], self.evolved_psi[-1])])
        return over, np.conj(over)                                                      # transfer amplitude

    def F(self):
        return (np.abs(1/self.subdim * np.sum(self.overlap)))**2                        # fidelity

    def gradient(self):
        theta = np.conj(np.sum(self.overlap))
        g = np.zeros(self.N-1)
        for k in range(self.subdim):
            sandwich = array([np.conj(self.evolved_chi[n][k]) @ (Hc * self.evolved_psi[n][k]) for n in range(self.N)])
            neighbor_sum = array([sandwich[n+1] + sandwich[n] for n in range(self.N-1)])
            g += np.real(1j * theta * 1/2 * neighbor_sum)
        return g * (self.delta_t/self.subdim**2)                                        # gradient

### optimal control ###

# functions for the optimization algorithm
JF_trap = lambda u: 1/2 * (1 - trapezoid(u).F())
gradient_trap = lambda u: trapezoid(u).gradient()
JF_rect = lambda u: 1/2 * (1 - rectangle(u).F())
gradient_rect = lambda u: rectangle(u).gradient()

# control bounds
bound = 0.2 * 2 * np.pi / e_unit

# info convergence function
def printinfo(dict, f):
    if dict['warnflag']==0:
        print(' SIMULATION CONVERGED')
    elif dict['warnflag']==1:
        print(' EXCEEDED NUMBER OF FUNC. EVALUATION')
    else:
        print(' SIMULATION NOT CONVERGED')
    print(' - number of iterations:\t', dict['nit'])
    print(' - final fidelity:\t', 1-2*f)

# storing statistics
trap_fidelity_reg, rect_fidelity_reg = [], []
trap_iterations_reg, rect_iterations_reg = [], []
trap_time_reg, rect_time_reg = [], []

np.random.seed(2027650)
kk = 0
nseeds = 0
# time threshold to store meaningful results
time_threshold = 60

while kk < 10:
    print('\nrunning for iteration #', kk+1)

    # initialize control vectors
    control_start = initialize_controls(n_iter=N_trap, loc_ctrl=loc_ctrl)
    bounds_trap = [(-bound, bound) for _ in range(np.prod(control_start.shape))]
    bounds_rect = bounds_trap[:-dim] if loc_ctrl else bounds_trap[:-1]

    start = perf_counter()
    u_trap, f_trap, d_trap = fmin_l_bfgs_b(func=JF_trap, x0=control_start, fprime=gradient_trap, bounds=bounds_trap, m=17, factr=10.0, maxfun=900)
    end = perf_counter()
    t1 = end - start

    start = perf_counter()
    u_rect, f_rect, d_rect = fmin_l_bfgs_b(func=JF_rect, x0=control_start[:-1], fprime=gradient_rect, bounds=bounds_rect, m=17, factr=10.0, maxfun=900)
    end = perf_counter()
    t2 = end - start

    nseeds += 1

    if t1 > time_threshold and t2 > time_threshold:
        print('\ntrapezoidal rule')
        printinfo(d_trap, f_trap)
        print(' - time: ', t1)
        trap_time_reg.append(t1)
        trap_iterations_reg.append(d_trap['nit'])
        trap_fidelity_reg.append(1-2*f_trap)

        print('\nrectangular rule')
        printinfo(d_rect, f_rect)
        print(' - time: ', t2)
        rect_time_reg.append(t2)
        rect_iterations_reg.append(d_rect['nit'])
        rect_fidelity_reg.append(1-2*f_rect)

        kk += 1

print('time:\n')
print(np.mean(trap_time_reg), '\t', np.std(trap_time_reg), '\t', np.mean(rect_time_reg), '\t', np.std(rect_time_reg))
print('niter:\n')
print(np.mean(trap_iterations_reg), '\t', np.mean(rect_iterations_reg))
print('fidelity:\n')
print(np.mean(trap_fidelity_reg), '\t', np.mean(rect_fidelity_reg))
print('number of unsuccessfull runs:\t', (nseeds-kk)/nseeds)
