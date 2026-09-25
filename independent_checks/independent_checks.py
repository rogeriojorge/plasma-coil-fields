"""Independent checks of selected near-axis equations; no full 3-D MHD solver.
Run: OPENBLAS_NUM_THREADS=1 python independent_checks.py
Requires numpy, scipy, sympy. Writes independent_results.json next to this file.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import root
from scipy.interpolate import CubicSpline
from scipy.special import ellipe
from numpy.polynomial.legendre import leggauss
import sympy as sp

OUT=Path(__file__).resolve().parent
results={}
# A first-order, nonplanar, vacuum QS reference, solved independently.
n=151; nfp=2; phi=2*np.pi*np.arange(n)/(n*nfp); period=2*np.pi/nfp
eta=.95; B0=1.; mu0=4*np.pi*1e-7; p2=-6e5; a=.03
R=1+.09*np.cos(nfp*phi); Z=-.09*np.sin(nfp*phi)
Rp=-.09*nfp*np.sin(nfp*phi); Zp=-.09*nfp*np.cos(nfp*phi)
Rpp=-.09*nfp**2*np.cos(nfp*phi); Zpp=.09*nfp**2*np.sin(nfp*phi)
Rppp=.09*nfp**3*np.sin(nfp*phi); Zppp=.09*nfp**3*np.cos(nfp*phi)
v=np.stack((Rp,R,Zp),axis=1); acc=np.stack((Rpp-R,2*Rp,Zpp),axis=1)
jerk=np.stack((Rppp-3*Rp,3*Rpp-R,Zppp),axis=1)
speed=np.linalg.norm(v,axis=1); ell=np.mean(speed); cross=np.cross(v,acc)
kappa=np.linalg.norm(cross,axis=1)/speed**3
normal=np.cross(cross/np.linalg.norm(cross,axis=1)[:,None],v/speed[:,None])
tau=np.einsum('ij,ij->i',cross,jerk)/np.sum(cross**2,axis=1)
k=np.fft.fftfreq(n,1/n)*nfp
Dphi=np.fft.ifft(1j*k[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real
Db=(ell/speed)[:,None]*Dphi
x=eta/kappa

def sigma_eq(y):
    sig,nu=y[:-1],y[-1]
    return np.r_[Db@sig+nu*(x**4+1+sig**2)+2*x*x*ell*tau,sig[0]]
sol=root(sigma_eq,np.r_[np.zeros(n),.2],tol=1e-11)
assert np.max(abs(sigma_eq(sol.x)))<1e-9
sigma,nu=sol.x[:-1],sol.x[-1]
weight=speed/speed.sum()
# Recompute the pressure response using an IVP monodromy in PHYSICAL (u,v),
# rather than just reuse the complex Fourier formula.
F=1-1/(1+x*x+1j*sigma)
Cp=2*mu0*p2*a*a*ell*ell*eta/(B0*B0*nu)
z=np.linalg.solve(Db-1j*nu*np.eye(n),-1j*Cp*F)
uv_closed=np.stack((x*z.real,(sigma*z.real+z.imag)/x),axis=1)
xp=Db@x; sigp=Db@sigma
A=np.zeros((n,2,2)); A[:,0,0]=(xp/x+nu*sigma)/ell
A[:,0,1]=-nu*x*x/ell
A[:,1,0]=(sigp+nu*(1+sigma*sigma))/(ell*x*x)
A[:,1,1]=-A[:,0,0]
den=(1+x*x)**2+sigma*sigma
f=np.stack((sigma,-1-x*x),axis=1)*(2*mu0*p2*a*a*ell*eta*x/(B0**2*nu*den))[:,None]
# Interpolate whole differential operator in geometrical angle, preserving periodicity.
As=CubicSpline(np.r_[phi,period],np.concatenate((speed[:,None,None]*A,(speed[0]*A[0])[None])),bc_type='periodic')
fs=CubicSpline(np.r_[phi,period],np.vstack((speed[:,None]*f,speed[0]*f[0])),bc_type='periodic')
def rhs(t,y):
    mat=As(t); M=y[:4].reshape(2,2); h=y[4:]
    return np.r_[(mat@M).ravel(),mat@h+fs(t)]
ivp=solve_ivp(rhs,(0,period),np.r_[np.eye(2).ravel(),np.zeros(2)],rtol=2e-11,atol=1e-13,dense_output=True,max_step=period/200)
assert ivp.success
end=ivp.y[:,-1]; w0=np.linalg.solve(np.eye(2)-end[:4].reshape(2,2),end[4:])
y=ivp.sol(phi); uv_ivp=np.einsum('nij,j->ni',y[:4].T.reshape(n,2,2),w0)+y[4:].T
rel=np.max(abs(uv_ivp-uv_closed))/np.max(abs(uv_closed))
beta=-2*mu0*p2*a*a/B0**2
length=beta*(ell*eta)**2/nu**2*(weight@F.real)
length_ivp=-weight@(kappa*uv_ivp[:,0])
krms=np.sqrt(weight@kappa**2); xir=np.sqrt(weight@np.sum(uv_closed**2,axis=1))
results['pressure_response']={'method':'physical-frame monodromy IVP vs complex periodic collocation',
    'beta_axis':float(beta),'iotaN':float(nu),'axis_length':float(2*np.pi*ell),
    'relative_max_difference':float(rel),'length_slope_formula':float(length),
    'length_slope_IVP':float(length_ivp),'mean_ReF':float(weight@F.real),
    'rms_displacement_m_per_alpha':float(xir),
    'lower_bound_rms_displacement_m_per_alpha':float(length/krms),
    'max_displacement_over_a_per_alpha':float(np.max(np.linalg.norm(uv_closed,axis=1))/a)}
assert rel<1e-5
assert abs(length_ivp-length)<1e-7
assert xir>=length/krms

# Trapped fraction: independent quadrature with CORRECT Boozer flux weighting.
# Angular mean <Q>=mean(Q/B^2)/mean(1/B^2), B=1+eps cos(theta).
# Substitute lambda=(1-t^2)/(1+eps) to tame the lambda endpoint.
g,w=leggauss(800); th=np.pi*(g+1); w=w/2

def trapped(eps):
    b=1+eps*np.cos(th); norm=np.sum(w/b**2); b2avg=1/norm
    def integrand(t):
        lam=(1-t*t)/(1+eps)
        average=np.sum(w*np.sqrt(np.maximum(0,1-lam*b))/b**2)/norm
        return lam*2*t/((1+eps)*average)
    integ=quad(integrand,0,1,epsabs=4e-12,epsrel=4e-12,limit=200)[0]
    return 1-.75*b2avg*integ
# Angular average of sqrt(u^2+1-cos theta) via complete elliptic integral.
def sqrt_average(u): return 2/np.pi*np.sqrt(u*u+2)*ellipe(2/(u*u+2))
def match_integrand(u):
    return u*(1/sqrt_average(u)-1/np.sqrt(u*u+1))
# Beyond u=50, difference contributes ~1/(48 U^3) to the integral.
# Use infinite quadrature too and compare, with numerical cancellation controlled.
I=quad(match_integrand,0,50,epsabs=2e-12,epsrel=2e-12,limit=200)[0]
# Asymptotic integrand=1/(16u^4)-5/(32u^6)+O(u^-8).
I+=1/(48*50**3)-1/(32*50**5)
Ct=1.5*(1-I)
rows=[]
for eps in [1e-2,3e-3,1e-3,3e-4,1e-4]:
    ft=trapped(eps);rows.append({'epsilon':eps,'ft':float(ft),'ft_over_sqrt_epsilon':float(ft/np.sqrt(eps))})
results['trapped_fraction']={'Ct_matched':float(Ct),'quadrature':'800-point Gauss-Legendre angle, adaptive lambda, Boozer B^-2 weight','rows':rows}
assert abs(Ct-1.462424956)<2e-8
assert abs(rows[-1]['ft_over_sqrt_epsilon']-Ct)<1e-4

# Exact cylindrical Ampere-law regularity calculation with a
# collisionless j_parallel proportional to sqrt(r), no numerical extrapolation.
r=sp.symbols('r',positive=True); jh,mu=sp.symbols('jhalf mu',nonzero=True)
Btheta=sp.Rational(2,5)*mu*jh*r**sp.Rational(3,2)
hess=sp.diff(Btheta,r,2)
assert sp.simplify(hess-sp.Rational(3,10)*mu*jh/sp.sqrt(r))==0
results['fractional_current_axis_regularity']={'j_z':'jhalf*r^(1/2)','B_theta':str(Btheta),'d2_Btheta_dr2':str(hess),'conclusion':'C1 but not C2 field at axis when jhalf != 0; no finite ordinary on-axis Hessian.'}
# Classical shifted-circle GS radial ODE, independently differentiated.
R0,I2,p2s,aa=sp.symbols('R0 I2 p2 aa',nonzero=True)
betap=-mu*p2s/I2**2
Delta=(aa**2-r**2)*(betap+sp.Rational(1,4))/(2*R0)
gs=sp.diff(r*(I2*r)**2*sp.diff(Delta,r),r)-r/R0*(2*mu*r*sp.diff(p2s*(r*r-aa*aa),r)-(I2*r)**2)
assert sp.simplify(gs)==0
results['circular_tokamak']={'delta':str(Delta),'Grad_Shafranov_residual':str(sp.simplify(gs))}
results['scope']='Selected independent algebra, kinetic-integral, and linear-response checks. No full 3-D equilibrium or kinetic solver, no coil optimization, no proof of all manuscript claims.'
(OUT/'independent_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

# Plot a derivative coefficient, not a finite-pressure equilibrium.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(OUT.parent/'paper.mplstyle')
fig, ax = plt.subplots(figsize=(3.4,2.3), constrained_layout=True)
ax.plot(phi, uv_closed[:,0]/a, label=r'$u/a$ (normal)')
ax.plot(phi, uv_closed[:,1]/a, label=r'$v/a$ (binormal)')
ax.set_xlabel(r'cylindrical angle $\phi$')
ax.set_ylabel(r'displacement per unit $\alpha$ / $a$')
ax.legend(frameon=False)
(OUT/'figures').mkdir(exist_ok=True)
fig.savefig(OUT/'figures'/'linear_response.pdf')
plt.close(fig)
