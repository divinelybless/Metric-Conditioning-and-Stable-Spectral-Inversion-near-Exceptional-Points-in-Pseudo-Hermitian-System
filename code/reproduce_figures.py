from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'manuscript' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def H_two(gamma, g=1.0):
    return np.array([[1j*gamma, g], [g, -1j*gamma]], dtype=complex)


def A_three(eps):
    return np.array([[1., 1., 0.], [0., 1.+eps, 1.], [0., 0., 2.]])


def V_three(eps):
    return np.array([[1., 1., 1.], [0., eps, 1.], [0., 0., 1.-eps]])


def eta_asymptotic(eps):
    V = V_three(eps)
    W = np.diag([eps/2, eps/2, 1.])
    Vi = np.linalg.inv(V)
    return Vi.T @ W @ Vi


def eta_constructive(eps):
    B = np.array([[eps, -1., 1.], [0., 1., -1./(1.-eps)], [0., 0., 1./(1.-eps)]])
    return B.T @ B


def C_chain(N, eps):
    d = np.array([1., 1.+eps] + [2.+(j-3)/(N-2) for j in range(3, N+1)])
    return np.diag(d) + np.diag(np.ones(N-1), 1)

# Figure 1: two-level conditioning
gamma = np.linspace(0, 0.99, 400)
kappa = (1+gamma)/(1-gamma)
plt.figure(figsize=(7.1,4.35))
plt.semilogy(gamma, kappa, label=r'$\kappa_2(H)=\kappa_2(\eta_*)$')
plt.semilogy(gamma, np.ones_like(gamma), '--', label=r'$\kappa_2(h_*)=1$')
plt.xlabel(r'$\gamma$'); plt.ylabel('Condition number')
plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
plt.savefig(OUT/'conditioning.pdf', bbox_inches='tight'); plt.close()

# Figure 2: inverse norms and singularity radius
eps = np.maximum(1e-12, 1-gamma**2)
Hinv = 1/(1-gamma)
hinv = 1/np.sqrt(eps)
radius = 1-gamma
plt.figure(figsize=(7.1,4.35))
plt.semilogy(gamma, Hinv, label=r'$\|H^{-1}\|_2$')
plt.semilogy(gamma, hinv, '--', label=r'$\|h_*^{-1}\|_2$')
plt.semilogy(gamma, radius, ':', label=r'$r_{\rm sing}(H)$')
plt.xlabel(r'$\gamma$'); plt.ylabel('Magnitude')
plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
plt.savefig(OUT/'inversion_and_radius.pdf', bbox_inches='tight'); plt.close()

# Figure 3: metric-condition-number sandwich
e = np.geomspace(1e-3, 0.5, 220)
lower = 8/e**2
upper_simple = 135/(4*e**2)
kas = np.array([np.linalg.cond(eta_asymptotic(x)) for x in e])
plt.figure(figsize=(7.1,4.35))
plt.loglog(e, lower, label=r'Lower bound $8/\varepsilon^2$')
plt.loglog(e, kas, '--', label='Asymptotically optimal feasible metric')
plt.loglog(e, upper_simple, ':', label=r'Constructive bound $135/(4\varepsilon^2)$')
plt.xlabel(r'$\varepsilon$'); plt.ylabel('Metric condition number')
plt.grid(alpha=.25, which='both'); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT/'three_level_metric_sandwich.pdf', bbox_inches='tight'); plt.close()

# Figure 4: best-certificate asymptotics versus exact radius
r3 = np.array([np.linalg.svd(A_three(x), compute_uv=False)[-1] for x in e])
cert_bound = e/(2*np.sqrt(2))
cert_as = np.array([1/np.sqrt(np.linalg.cond(eta_asymptotic(x))) for x in e])
plt.figure(figsize=(7.1,4.35))
plt.semilogx(e, r3, label='Exact singularity radius')
plt.semilogx(e, cert_bound, '--', label=r'Upper bound $\varepsilon/(2\sqrt{2})$')
plt.semilogx(e, cert_as, ':', linewidth=2.2, label='Certificate from asymptotically optimal metric')
plt.xlabel(r'$\varepsilon$'); plt.ylabel('Euclidean radius / certificate')
plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT/'three_level_certificate_v8.pdf', bbox_inches='tight'); plt.close()

# Figure 5: growing-dimensional chain
e2 = np.geomspace(1e-4, 0.5, 100)
plt.figure(figsize=(7.3,4.7))
for N in (3,8,32):
    radii = [np.linalg.svd(C_chain(N,float(x)), compute_uv=False)[-1] for x in e2]
    plt.loglog(e2, radii, label=rf'$r_{{\rm sing}}(C_{{{N},\varepsilon}})$')
plt.loglog(e2, e2/(2*np.sqrt(2)), 'k--', label=r'Metric-certificate upper bound $\varepsilon/(2\sqrt{2})$')
plt.axhline(1/np.sqrt(6), linestyle=':', label=r'Uniform lower bound $1/\sqrt{6}$')
plt.xlabel(r'$\varepsilon$'); plt.ylabel('Euclidean invertibility radius / certificate')
plt.grid(alpha=.25, which='both'); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT/'growing_chain_radius.pdf', bbox_inches='tight'); plt.close()

# Numerical checks used in the paper
for x in (0.5,0.1,0.01,0.001):
    assert np.linalg.cond(eta_asymptotic(x)) >= 8/x**2*(1-1e-10)
    assert np.linalg.norm(np.linalg.inv(A_three(x)),2) <= np.sqrt(15)/2 + 1e-12
for N in (3,8,32,64):
    for x in (0.5,0.1,0.01):
        C = C_chain(N,x)
        assert np.linalg.norm(C,2) <= 4 + 1e-12
        assert np.linalg.norm(np.linalg.inv(C),2) <= np.sqrt(6) + 1e-10
print(f'Wrote five manuscript figures to {OUT}')
