"""Reproduce the new section's algebra and independent response checks.

This does not execute ESSOS or VMEX and does not validate a finite-radius
free-boundary equilibrium. The geometrical example solves the bundled
independent Boozer-projection equations using NumPy/SciPy.
"""
from pathlib import Path
import json
import sys
import numpy as np
import sympy as sp
from numpy.testing import assert_allclose
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'reference' / 'checks'))
from reference_equilibrium import reference_case

results = {}

def zero(name: str, expression) -> None:
    simplified = sp.simplify(expression)
    is_zero = (all(entry == 0 for entry in simplified)
               if isinstance(simplified, sp.MatrixBase) else simplified == 0)
    assert is_zero, (name, simplified)
    results[name] = 'exact symbolic identity'

R, B, I, mu, p2, r, a = sp.symbols('R B I mu p2 r a', nonzero=True, real=True)
beta = -mu*p2/I**2
nu = R*I/B
X0 = (beta+sp.Rational(3,4))/R
Xc = -(beta+sp.Rational(5,4))/(2*R)
Ys = Xc
bs = -4*mu*p2/(nu*B**2)
zero('tokamak_closure', 2*nu*X0+3*nu/(2*R)-R*bs/2-3*I/B)
zero('tokamak_jacobian_constraint', Ys+1/(2*R)+Xc+X0)
zero('tokamak_midpoint', X0+Xc-(beta+sp.Rational(1,4))/(2*R))
shift = (a*a-r*r)*(beta+sp.Rational(1,4))/(2*R)
zero('tokamak_Grad_Shafranov', sp.diff(r*(I*r)**2*sp.diff(shift,r),r)
     -r/R*(2*mu*r*sp.diff(p2*(r*r-a*a),r)-(I*r)**2))
th, c0, cc, ss = sp.symbols('theta X20 X2c Y2s', real=True)
normal = (c0+cc*sp.cos(2*th))*sp.cos(th)+ss*sp.sin(2*th)*sp.sin(th)
zero('circle_normal_harmonics', sp.expand_trig(normal-(c0+(cc+ss)/2)*sp.cos(th)
     -(cc-ss)/2*sp.cos(3*th)))

x, sig, xp, sigp, ell, tau, v, cp, eta, a = sp.symbols(
    'x sig xp sigp ell tau nu cp eta a', real=True, nonzero=True)
J = sp.Matrix([[0,-1],[1,0]])
D = (1+x*x)**2+sig*sig
for sg in (-1,1):
    for sf in (-1,1):
        chi = sg*sf
        E = sp.Matrix([[x,0],[chi*sig/x,chi/x]])
        Ep = E.diff(x)*xp+E.diff(sig)*sigp
        A = sp.Matrix([[(xp/x+v*sig)/ell, -chi*v*x*x/ell],
                        [chi*(sigp+v*(1+sig*sig))/(ell*x*x),-(xp/x+v*sig)/ell]])
        zero(f'conjugacy_{sg}_{sf}', E.inv()*(ell*A*E-Ep)-v*J)
        forcing = sp.Matrix([2*sg*cp*a*a*ell*eta*x*sig/(v*D),
                             -2*sf*cp*a*a*ell*eta*x*(1+x*x)/(v*D)])/(sg*B)
        Cp = 2*cp*a*a*ell*ell*eta/(B*v)
        target = Cp/D*sp.Matrix([sig,-x*x*(1+x*x)-sig*sig])
        zero(f'forcing_{sg}_{sf}', E.inv()*ell*forcing-target)
        M = E.inv().T*E.inv()
        expected = sp.Matrix([[(1+sig*sig)/(x*x),-chi*sig],[-chi*sig,x*x]])
        zero(f'flux_metric_{sg}_{sf}', M-expected)
        Mp = M.diff(x)*xp+M.diff(sig)*sigp
        zero(f'flux_transport_{sg}_{sf}', Mp+(ell*A).T*M+M*(ell*A))

# The direct flux critical point has one common Hessian determinant.
Au, Av, C, hu, hv = sp.symbols('Au Av C hu hv', real=True)
H = sp.Matrix([[2*Au, C], [C, 2*Av]])
critical = -H.inv()*sp.Matrix([hu, hv])
zero('direct_flux_critical_point', H*critical+sp.Matrix([hu,hv]))
zero('direct_flux_uncoupled_sign', critical.subs(C,0)
     -sp.Matrix([-hu/(2*Au),-hv/(2*Av)]))

# Manufactured harmonics: independent closed solution, FFT, collocation,
# and the periodic real-space Green function. They test the response only.
n = 257
ph = 2*np.pi*np.arange(n)/n
nu, Cp, m = .371, -.0073, 3
F0, Fc, Fs = .55,.09,-.07
F = F0+Fc*np.cos(m*ph)+1j*Fs*np.sin(m*ph)
zexact = Cp*(F0/nu+(nu*Fc+m*Fs)/(nu*nu-m*m)*np.cos(m*ph)
                +1j*(m*Fc+nu*Fs)/(nu*nu-m*m)*np.sin(m*ph))
k = np.fft.fftfreq(n,1/n)
zfft = np.fft.ifft(Cp*np.fft.fft(F)/(nu-k))
assert_allclose(zfft,zexact,rtol=1e-13,atol=1e-15)
Dph = np.fft.ifft(1j*k[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real
zcoll = np.linalg.solve(Dph-1j*nu*np.eye(n),-1j*Cp*F)
assert_allclose(zcoll,zexact,rtol=2e-11,atol=2e-14)
fun = lambda t: np.exp(-1j*nu*t)*(F0+Fc*np.cos(m*t)+1j*Fs*np.sin(m*t))
integral = quad(lambda t: fun(t).real,0,2*np.pi,epsabs=1e-13)[0]+1j*quad(
    lambda t: fun(t).imag,0,2*np.pi,epsabs=1e-13)[0]
z0green = -1j*Cp*np.exp(2j*np.pi*nu)/(1-np.exp(2j*np.pi*nu))*integral
assert_allclose(z0green,zexact[0],atol=2e-15)
results['manufactured_FFT_max_error_m'] = float(np.max(abs(zfft-zexact)))
results['manufactured_collocation_max_error_m'] = float(np.max(abs(zcoll-zexact)))
results['periodic_Green_error_m'] = float(abs(z0green-zexact[0]))

# A nonplanar vacuum axis, not a free-boundary equilibrium comparison.
rows=[]
for n in (51,101,151):
    q=reference_case(nphi=n,I2=0.,p2=0.)
    x,sig,nu,ell=q.X1c,q.sigma,q.iotaN,q.d_l_d_varphi
    dv, ds=q.d_d_varphi,q.d_d_varphi/ell
    eta, B0, chi=q.etabar,q.B0,q.sG*q.spsi
    mu0,p2,a=4e-7*np.pi,-6e5,.03
    Cp=2*mu0*p2*a*a*ell*ell*eta/(B0*B0*nu)
    F=1-1/(1+x*x+1j*sig)
    z=np.linalg.solve(dv-1j*nu*np.eye(n),-1j*Cp*F)
    u=x*z.real; vv=chi*(sig*z.real+z.imag)/x
    xp,sp_=dv@x,dv@sig
    AA=(xp/x+nu*sig)/ell
    AB=-chi*nu*x*x/ell
    BA=chi*(sp_+nu*(1+sig*sig))/(ell*x*x)
    cp=mu0*p2/B0; den=(1+x*x)**2+sig*sig
    fn=2*cp*a*a*ell*eta*x*sig/(nu*den*B0)
    fb=-2*chi*cp*a*a*ell*eta*x*(1+x*x)/(nu*den*B0)
    operator=np.block([[ds-np.diag(AA),-np.diag(AB)],
                       [-np.diag(BA),ds+np.diag(AA)]])
    uv=np.linalg.solve(operator,np.r_[fn,fb])
    relative=float(np.max(abs(np.r_[u,vv]-uv))/np.max(abs(uv)))
    weight=q.d_l_d_phi/np.sum(q.d_l_d_phi)
    factor=float(np.dot(weight,F.real))
    predicted=-2*mu0*p2*a*a/(B0*B0)*(ell*eta)**2/nu**2*factor
    variation=float(-np.dot(weight,q.curvature*u))
    parity=max(np.max(abs(u-u[(-np.arange(n))%n])),
               np.max(abs(vv+vv[(-np.arange(n))%n])))
    # Differentiate the actual length of the perturbed Cartesian curve.
    # Work with periodic cylindrical vectors: derivative includes basis rotation.
    xic=u[:,None]*q.normal_cylindrical+vv[:,None]*q.binormal_cylindrical
    dxic=q.d_d_phi@xic
    dxic[:,0]-=xic[:,1];dxic[:,1]+=xic[:,0]
    base=np.stack((q.R0p,q.R0,q.Z0p),1)
    def length(alpha):
        return 2*np.pi*np.mean(np.linalg.norm(base+alpha*dxic,axis=1))
    step=1e-4
    lfinite=(length(step)-length(-step))/(2*step*q.axis_length)
    rows.append(dict(nphi=n,iotaN=float(nu),
          response_relative_difference=relative,
          length_slope_over_L_predicted=float(predicted),
          length_variation_over_L=float(variation),
          length_finite_difference_over_L=float(lfinite),
          length_identity_relative_error=float(abs(variation-predicted)/abs(predicted)),
          symmetry_max_error_m=float(parity),
          first_order_sigma_residual=q.reference_residuals['sigma']))
    assert relative < 3e-7
    assert abs(variation-predicted)<1e-9
    assert abs(lfinite-predicted)<3e-9
    assert parity<1e-9
results['nonplanar_reference'] = rows
results['scope'] = 'Algebra and linear-response tests only; no ESSOS or VMEX execution.'
(HERE/'shafranov_checks.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
