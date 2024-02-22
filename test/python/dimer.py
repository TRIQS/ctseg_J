# dimer test from github.com/triqs/benchmarks/dimer

from triqs.gf import Gf, BlockGf, MeshImFreq, fit_hermitian_tail, iOmega_n, inverse, Fourier
from triqs.gf.tools import make_zero_tail
from triqs.gf.gf_factories import make_gf_from_fourier
from triqs.operators import c, c_dag
from triqs.operators.util import U_matrix_kanamori, h_int_density
from itertools import product

from triqs_ctseg import Solver

from numpy import matrix, array, block, diag, eye
from numpy.linalg import inv

# ==== System Parameters ====
beta = 10.                       # Inverse temperature
mu = 0.0                        # Chemical potential
eps = array([0.0, 0.1])         # Impurity site energies
eps_bath = array([0.27, -0.4])  # Bath site energies
U = 1.                          # Density-density interaction
J = 0.2                         # Hunds coupling
block_names = ['up', 'dn']
n_orb = len(eps)
n_orb_bath = len(eps_bath)

# Non-interacting impurity Hamiltonian in matrix representation
h_0_mat = diag(eps - mu)

# Bath Hamiltonian in matrix representation
h_bath_mat = diag(eps_bath)

# Coupling matrix
V_mat = matrix([[1., 1.],
                [1., 1.]])

# ==== Local Hamiltonian ====
c_dag_vec = {s: matrix([[c_dag(s, o) for o in range(n_orb)]]) for s in block_names}
c_vec = {s: matrix([[c(s, o)] for o in range(n_orb)]) for s in block_names}

h_0 = sum(c_dag_vec[s] * h_0_mat * c_vec[s] for s in block_names)[0, 0]

Umat, Upmat = U_matrix_kanamori(n_orb, U_int=U, J_hund=J)
h_int = h_int_density(block_names, range(n_orb), Umat, Upmat, off_diag=True)

h_imp = h_0 + h_int

# ==== Bath & Coupling Hamiltonian ====
c_dag_bath_vec = {s: matrix([[c_dag(s, o) for o in range(n_orb, n_orb + n_orb_bath)]]) for s in block_names}
c_bath_vec = {s: matrix([[c(s, o)] for o in range(n_orb, n_orb + n_orb_bath)]) for s in block_names}

h_bath = sum(c_dag_bath_vec[s] * h_bath_mat * c_bath_vec[s] for s in block_names)[0, 0]
h_coup = sum(c_dag_vec[s] * V_mat * c_bath_vec[s] + c_dag_bath_vec[s] * V_mat.H * c_vec[s] for s in block_names)[0, 0]

# ==== Total impurity Hamiltonian ====
h_tot = h_imp + h_coup + h_bath

# ==== Green function structure ====
gf_struct = [(s, n_orb) for s in block_names]

# ==== Non-Interacting Impurity Green function  ====
n_iw = int(10 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
h_tot_mat = block([[h_0_mat, V_mat],
                   [V_mat.H, h_bath_mat]])
for bl, iw in product(block_names, iw_mesh):
    G0_iw[bl][iw] = inv(iw.value * eye(2*n_orb) - h_tot_mat)[:n_orb, :n_orb]

# ==== Hybridization Function ====
Delta = G0_iw.copy()
Delta['up'] << iOmega_n - h_0_mat - inverse(G0_iw['up'])
Delta['dn'] << iOmega_n - h_0_mat - inverse(G0_iw['dn'])

# --------- Construct the CTSET solver ----------
constr_params = {
    'beta': beta,
    'gf_struct': gf_struct,
    'n_tau': 10001,
}
S = Solver(**constr_params)

# --------- fill Delta_tau from G0_iw ----------
for name, g0 in Delta:
    known_moments = make_zero_tail(Delta[name], 1)
    tail, err = fit_hermitian_tail(Delta[name], known_moments)
    S.Delta_tau[name] << make_gf_from_fourier(Delta[name], S.Delta_tau.mesh, tail).real

# --------- Solve! ----------
n_cycles = int(1e+5)
solve_params = {
    'h_int': h_int,
    'hartree_shift': [0.0, 0.0, 0.1, 0.1],
    'n_warmup_cycles': 5000,
    'n_cycles': n_cycles,
    'length_cycle': 100,
}
S.solve(**solve_params)

G_iw = BlockGf(mesh=Delta.mesh, gf_struct=gf_struct)
for name, gf in G_iw:
    gf << Fourier(S.results.G_tau[name])
