"""Leading fractional-current transverse equilibrium and normal-form checks.

This is a reduced-order research calculation, not a full MHD/kinetic solver.
Run with OPENBLAS_NUM_THREADS=1. The companion manuscript defines all units.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import root
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)


def nonplanar_geometry(n=201, nb=96):
    """Independent first-order vacuum QS solve; no pyQSC import."""
    nfp, eta = 2, 0.95
    period = 2*np.pi/nfp
    phi = period*np.arange(n)/n
    R = 1+.09*np.cos(2*phi)
    Rp, Rpp, Rppp = -.18*np.sin(2*phi), -.36*np.cos(2*phi), .72*np.sin(2*phi)
    Zp, Zpp, Zppp = -.18*np.cos(2*phi), .36*np.sin(2*phi), .72*np.cos(2*phi)
    v = np.stack((Rp, R, Zp), axis=1)
    acc = np.stack((Rpp-R, 2*Rp, Zpp), axis=1)
    jerk = np.stack((Rppp-3*Rp, 3*Rpp-R, Zppp), axis=1)
    speed = np.linalg.norm(v, axis=1)
    cross = np.cross(v, acc)
    kap = np.linalg.norm(cross, axis=1)/speed**3
    tau = np.einsum('ij,ij->i', cross, jerk)/np.sum(cross**2, axis=1)
    ell = np.mean(speed)
    wave = np.fft.fftfreq(n, 1/n)*nfp
    D = np.fft.ifft(1j*wave[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0).real
    Db = (ell/speed)[:, None]*D
    x = eta/kap
    def residual(y):
        sig, nu = y[:-1], y[-1]
        return np.r_[Db@sig+nu*(x**4+1+sig**2)+2*x*x*ell*tau, sig[0]]
    sol = root(residual, np.r_[np.zeros(n), .21], tol=1e-11)
    err = float(np.max(abs(residual(sol.x))))
    if err > 2e-10:
        raise RuntimeError(f'First-order solve residual {err:g}')
    sigma, nu = sol.x[:-1], sol.x[-1]
    c = np.fft.fft(speed)/n
    prim = np.zeros(n, complex)
    prim[1:] = c[1:]/(1j*wave[1:])
    primitive = np.fft.ifft(prim*n).real
    varphi = phi+(primitive-primitive[0])/ell
    target = period*np.arange(nb)/nb
    def interpolate(values):
        return CubicSpline(np.r_[varphi, period], np.r_[values, values[0]], bc_type='periodic')(target)
    x, sig = interpolate(x), interpolate(sigma)
    E = np.zeros((nb, 2, 2))
    E[:, 0, 0], E[:, 1, 0], E[:, 1, 1] = x, sig/x, 1/x
    return E, float(nu), float(ell), {'sigma_residual': err, 'eta_per_m': eta, 'n_geometry': n, 'nfp': nfp}


def fractional_source(E, nu, ntheta=128, ngamma=256, nfp=2):
    """Physical-plane Poisson solve and pullback to normalized flux coordinates."""
    nb = len(E)
    theta = 2*np.pi*np.arange(ntheta)/ntheta
    gamma = 2*np.pi*np.arange(ngamma)/ngamma
    eg = np.stack((np.cos(gamma), np.sin(gamma)), axis=1)
    u = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    mg = np.fft.fftfreq(ngamma, 1/ngamma)
    P = np.zeros((nb, ntheta))
    poisson_error = circulation_error = 0.
    for j, mat in enumerate(E):
        Q = np.linalg.inv(mat).T@np.linalg.inv(mat)
        source = np.einsum('gi,ij,gj->g', eg, Q, eg)**.25
        fc = np.fft.fft(source)/ngamma/(25/4-mg**2)
        residual = np.fft.ifft((25/4-mg**2)*fc*ngamma).real-source
        poisson_error = max(poisson_error, float(np.max(abs(residual))))
        y = u@mat.T
        rho = np.linalg.norm(y, axis=1)
        angle = np.arctan2(y[:, 1], y[:, 0])
        phase = np.exp(1j*angle[:, None]*mg[None, :])
        fa, fga = (phase@fc).real, (phase@(1j*mg*fc)).real
        P[j] = rho**2.5*fa
        er = y/rho[:, None]
        et = np.stack((-er[:, 1], er[:, 0]), axis=1)
        grad = rho[:, None]**1.5*(2.5*fa[:, None]*er+fga[:, None]*et)
        b = np.stack((-grad[:, 1], grad[:, 0]), axis=1)
        tangent = np.stack((-np.sin(theta), np.cos(theta)), axis=1)@mat.T
        circ = 2*np.pi*np.mean(np.sum(b*tangent, axis=1))
        circulation_error = max(circulation_error, abs(circ-4*np.pi/5)/(4*np.pi/5))
    k = np.fft.fftfreq(nb, 1/nb)*nfp
    m = np.fft.fftfreq(ntheta, 1/ntheta)
    pc = np.fft.fft2(P)/P.size
    divisor = k[:, None]+nu*m[None, :]
    divisor[0, 0] = 1.
    sc = -pc/(1j*divisor)
    sc[0, 0] = 0.
    s = np.fft.ifft2(sc*P.size).real
    st = np.fft.ifft2(1j*m[None, :]*sc*P.size).real
    sphi = np.fft.ifft2(1j*k[:, None]*sc*P.size).real
    homerr = float(np.max(abs(sphi+nu*st+P-P.mean())))
    ut = np.stack((-np.sin(theta), np.cos(theta)), axis=1)
    h = st[:, :, None]*u[None, :, :]-2.5*s[:, :, None]*ut[None, :, :]
    physical_h = np.einsum('bij,btj->bti', E, h)
    return {'P': P, 'pc': pc, 'sc': sc, 'k': k, 'm': m, 'theta': theta, 's': s, 'st': st,
            'h': physical_h, 'Pbar': float(P.mean()), 'iota_factor': float(2.5*P.mean()),
            'poisson_error': poisson_error, 'circulation_relative_error': circulation_error,
            'homological_error': homerr,
            'geometry_rms_factor': float(np.sqrt(np.mean(np.sum(physical_h**2, axis=2))))}


def integrate_normal_form(data, nu, epsilon, periods=64, nfp=2):
    """Direct field-line integration, independent of the normal-form solution."""
    pc, sc = data['pc'], data['sc']
    active = np.abs(pc)>2e-13
    kk = np.broadcast_to(data['k'][:, None], pc.shape)[active]
    mm = np.broadcast_to(data['m'][None, :], pc.shape)[active]
    pp, ss = pc[active], sc[active]
    def evaluate(theta, phi):
        ex = np.exp(1j*(mm*theta+kk*phi))
        return ((pp@ex).real, ((1j*mm*pp)@ex).real,
                (ss@ex).real, ((1j*mm*ss)@ex).real)
    theta_bar = .317
    _, _, s0, st0 = evaluate(theta_bar, 0.)
    A0, t0 = .5+epsilon*st0, theta_bar-2.5*epsilon*s0
    def rhs(phi, y):
        theta, action = y
        if action <= 0:
            raise RuntimeError('Orbit left the positive-action domain')
        rad = np.sqrt(2*action)
        P, Pt, _, _ = evaluate(theta, phi)
        return [nu+2.5*epsilon*np.sqrt(rad)*P, -epsilon*rad**2.5*Pt]
    end = periods*2*np.pi/nfp
    sol = solve_ivp(rhs, (0, end), [t0, A0], method='DOP853', rtol=2e-12, atol=2e-13,
                    max_step=.3)
    if not sol.success:
        raise RuntimeError(sol.message)
    corrected, actions = [], []
    for phi, theta, action in zip(sol.t, sol.y[0], sol.y[1]):
        _, _, ss0, st = evaluate(theta, phi)
        rad = np.sqrt(2*action)
        corrected.append(theta+2.5*epsilon*np.sqrt(rad)*ss0)
        actions.append(action-epsilon*rad**2.5*st)
    measured = (corrected[-1]-corrected[0])/end
    factor = (measured-nu)/epsilon
    return {'epsilon': epsilon, 'periods': periods, 'rhs_evaluations': sol.nfev,
            'measured_iota_factor': float(factor), 'predicted_iota_factor': data['iota_factor'],
            'relative_factor_error': float(abs(factor/data['iota_factor']-1)),
            'corrected_action_range': float(np.ptp(actions)),
            'retained_Fourier_coefficients': len(pp)}


def main():
    results = {'scope': 'Leading transverse Maxwell/field-line equilibrium for prescribed fractional parallel current. Not full MHD or kinetic validation.'}
    nb = 96
    circle = np.broadcast_to(np.eye(2), (nb, 2, 2)).copy()
    ellipse = np.broadcast_to(np.diag([1.5, 1/1.5]), (nb, 2, 2)).copy()
    E, nu, ell, geom = nonplanar_geometry(nb=nb)
    datasets = {}
    for name, mat in [('circle', circle), ('ellipse', ellipse), ('nonplanar_QS', E)]:
        data = fractional_source(mat, nu)
        datasets[name] = data
        results[name] = {k: data[k] for k in ['Pbar', 'iota_factor', 'poisson_error', 'circulation_relative_error', 'homological_error', 'geometry_rms_factor']}
        results[name]['iota_half_per_lambda_half'] = ell*data['iota_factor']
        assert data['poisson_error'] < 1e-10
        assert data['circulation_relative_error'] < 2e-7
        assert data['homological_error'] < 1e-9
    assert abs(datasets['circle']['iota_factor']-.4)<1e-12
    assert datasets['circle']['geometry_rms_factor']<1e-12
    results['nonplanar_geometry'] = dict(geom, iotaN=nu, ell_m=ell,
        definition='R=1+0.09 cos(2 phi), Z=-0.09 sin(2 phi), etabar=0.95, B0=1, I2=0')
    Ehi, nuhi, ellhi, _ = nonplanar_geometry(n=301, nb=144)
    hi = fractional_source(Ehi, nuhi, ntheta=192, ngamma=384)
    results['resolution'] = {'coarse_iota_factor': datasets['nonplanar_QS']['iota_factor'],
        'fine_iota_factor': hi['iota_factor'],
        'relative_change': abs(hi['iota_factor']/datasets['nonplanar_QS']['iota_factor']-1)}
    assert results['resolution']['relative_change'] < 1e-6
    results['direct_integration'] = {}
    for name in ['ellipse', 'nonplanar_QS']:
        rows = [integrate_normal_form(datasets[name], nu, eps) for eps in [2e-3, 1e-3, 5e-4]]
        results['direct_integration'][name] = rows
        assert rows[-1]['relative_factor_error'] < .01
        assert rows[-1]['corrected_action_range'] < rows[0]['corrected_action_range']/8
    (OUT/'fractional_bootstrap_results.json').write_text(json.dumps(results, indent=2)+'\n')
    np.savez(OUT/'fractional_bootstrap_arrays.npz', E=E, nu=nu, ell=ell,
             P=datasets['nonplanar_QS']['P'], s=datasets['nonplanar_QS']['s'],
             h=datasets['nonplanar_QS']['h'], theta=datasets['nonplanar_QS']['theta'])
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for name, label in [('circle', 'Circular section'), ('ellipse', 'Fixed ellipse'), ('nonplanar_QS', 'Nonplanar QS reference')]:
        ax.plot(datasets[name]['theta'], datasets[name]['P'][0], label=label)
    ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$P(\theta,\varphi=0)$'); ax.legend(frameon=False)
    fig.savefig(FIG/'fractional_source.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for name, label in [('ellipse','Fixed ellipse'), ('nonplanar_QS','Nonplanar QS reference')]:
        rows = results['direct_integration'][name]
        ax.loglog([r['epsilon'] for r in rows], [r['relative_factor_error'] for r in rows], 'o-', label=label)
    ax.set_xlabel(r'$\varepsilon_b=\ell\lambda_{1/2}\sqrt{r_*}$')
    ax.set_ylabel('Relative error in transform correction'); ax.legend(frameon=False)
    fig.savefig(FIG/'fractional_transform_check.pdf'); plt.close(fig)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
