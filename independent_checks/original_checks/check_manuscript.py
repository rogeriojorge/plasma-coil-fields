"""Independent algebra checks for the revised manuscript.

No ESSOS, VMEX or pyQSC_JAX import is made. The equilibrium arrays are
calculated by the bundled NumPy Fourier-collocation reference. These checks
are not a replay of the native PR tests or of the free-boundary benchmark.
"""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.integrate import quad
from numpy.testing import assert_allclose

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'reference'))
sys.path.insert(0, str(HERE / 'reference' / 'checks'))
from reference_equilibrium import reference_case
from plasma_field import local_plasma_jet, total_jet, MU0
from zero_current import qsc_data, zero_current_jet, zero_current_hessian


def closed_local(q, a):
    x, sig, k = q.X1c, q.sigma, q.curvature
    ell, chi = abs(q.G0) / q.B0, q.sG*q.spsi
    cp, I2, io = MU0*q.p2/q.B0, q.I2, q.iotaN
    xp, sp = q.d_d_varphi@x, q.d_d_varphi@sig
    den = (1+x*x)**2+sig*sig
    A, B = q.X2s-chi*q.Y2c, q.X2c+chi*q.Y2s
    bt = a*a*(q.sG*cp+I2/ell*(io+chi*ell*q.torsion+((1+x*x)*sp-2*sig*x*xp)/den))
    bn = (-I2*a*a*k*sig*x*x/(2*den)+2*q.sG*cp*a*a*ell*q.etabar*x*sig/(io*den)
          +2*I2*a*a*x*x/den**2*((sig*sig-(1+x*x)**2)*A-2*sig*(1+x*x)*B))
    bb = (chi*I2*a*a*k/2*(np.log(8*ell/a)-.5-.5*np.log(den/(4*x*x))+x*x*(1+x*x)/den)
          -2*q.spsi*cp*a*a*ell*q.etabar*x*(1+x*x)/(io*den)
          -2*chi*I2*a*a*x*x/den**2*((sig*sig-(1+x*x)**2)*B+2*sig*(1+x*x)*A))
    return np.stack((bt,bn,bb),axis=-1)


def angular_local(q, a, ntheta=8192):
    """Unreduced current/shape integral, as in the PR's angular algorithm."""
    theta = 2*np.pi*(np.arange(ntheta)+.137)/ntheta
    co, si = np.cos(theta), np.sin(theta)
    ell, chi = abs(q.G0)/q.B0, q.sG*q.spsi
    j, C2 = 2*chi*q.I2, q.G2+(q.iota-q.iotaN)*q.I2
    ds = q.d_d_varphi/ell
    x,y,sig = q.X1c,q.Y1s,q.sigma
    c = np.stack((np.full_like(x,-chi*q.spsi*q.B0*q.beta_1s),
                  j*(ds@x-q.torsion*y*sig),j*(ds@(y*sig)+q.torsion*x)-2*chi*C2*y/ell),-1)
    s = np.stack((np.zeros_like(x),-j*q.torsion*y+2*chi*C2*x/ell,
                  j*(ds@y)+2*chi*C2*y*sig/ell),-1)
    result=[]
    for idx in (0,7,23,44):
        E=np.stack((np.zeros(ntheta),x[idx]*co,y[idx]*(si+sig[idx]*co)),-1)
        X=q.X20[idx]+q.X2c[idx]*np.cos(2*theta)+q.X2s[idx]*np.sin(2*theta)
        Y=q.Y20[idx]+q.Y2c[idx]*np.cos(2*theta)+q.Y2s[idx]*np.sin(2*theta)
        Z=q.Z20[idx]+q.Z2c[idx]*np.cos(2*theta)+q.Z2s[idx]*np.sin(2*theta)
        displacement=np.stack((Z,X,Y),-1)
        current=c[idx]*co[:,None]+s[idx]*si[:,None]
        E2=np.sum(E*E,axis=-1)
        integ=(np.cross([j,0,0],displacement)+np.cross(current,E))/E2[:,None]
        integ-=2*np.sum(E*displacement,axis=-1)[:,None]*np.cross([j,0,0],E)/E2[:,None]**2
        val=-a*a*np.mean(integ,axis=0)/2
        den=(1+x[idx]**2)**2+sig[idx]**2
        val+=j*a*a*q.curvature[idx]/4*np.array([
            0,-chi*sig[idx]*x[idx]**2/den,
            np.log(8*ell/a)-.5-.5*np.log(den/(4*x[idx]**2))+x[idx]**2*(1+x[idx]**2)/den])
        result.append(val)
    return np.array(result)


def compact_zero_gradient(q,a):
    x,sig=q.X1c,q.sigma
    ell,cp,chi=abs(q.G0)/q.B0,MU0*q.p2/q.B0,q.sG*q.spsi
    ds=q.d_d_varphi/ell;xs,ss=ds@x,ds@sig
    w=1+x*x+1j*chi*sig;ws=2*x*xs+1j*chi*ss
    ft=4*q.spsi*cp*ell*q.etabar*x/(q.iotaN*w)
    fn=2j*q.sG*cp*x*x/w;fb=2*q.sG*cp*(1+1j*chi*sig)/w
    fns=2j*q.sG*cp*(2*x*xs*w-x*x*ws)/w**2
    fbs=2*q.sG*cp*(1j*chi*ss*w-(1+1j*chi*sig)*ws)/w**2
    qt=x*x/(2*w*w)*(q.spsi*cp*(8*q.Z2s-ell/q.iotaN*(10*q.etabar**2-8*q.B2c/q.B0))
         +8j*q.sG*cp*q.Z2c-4*ft*(q.X2c+chi*q.Y2s+1j*(q.Y2c-chi*q.X2s)))
    cur=q.curvature*x*x*ft/(4*w)
    gnn=a*a*(-(qt+q.curvature*ft/4+cur).imag-fbs.real/2-q.torsion*(fn.real+fb.imag)/2)
    gnb=a*a*(fns.real/2-q.torsion*fb.real/2+q.torsion*fn.imag/2-(qt+cur).real)
    b=closed_local(q,a)
    gtt=ds@b[:,0]-q.curvature*b[:,1]
    gtn=ds@b[:,1]+q.curvature*b[:,0]-q.torsion*b[:,2]
    gtb=ds@b[:,2]+q.torsion*b[:,1]
    return np.stack((np.stack((gtt,gtn,gtb),-1),np.stack((gtn,gnn,gnb),-1),
                     np.stack((gtb,gnb,-gtt-gnn),-1)),axis=1)


def explicit_compatibility_residuals(q):
    """Appendix B equations, evaluated independently of matrix assembly."""
    x,y,s=q.X1c,q.Y1s,q.sigma
    ell=abs(q.G0)/q.B0;D=q.d_d_varphi;io=q.iotaN;chi=q.sG*q.spsi
    ys=y*s
    X0,Xc,Xs,Y0,Yc,Ys,Z0,Zc,Zs=[getattr(q,n) for n in
        ('X20','X2c','X2s','Y20','Y2c','Y2s','Z20','Z2c','Z2s')]
    c=(x*(D@Xs)-(D@x)*Xs+ys*(D@Ys)-(D@ys)*Ys-y*(D@(Y0+Yc))+(D@y)*(Y0+Yc)
       -io*(x*(X0+3*Xc)+ys*(Y0+3*Yc)+3*y*Ys)
       +2*ell*q.torsion*(-y*(X0+Xc-s*Xs)-x*Ys)+2*ell*q.curvature*x*Zs
       -chi*ell*q.beta_1s/2-3*q.sG*ell*q.I2*q.etabar/q.B0)
    d=(x*(D@(X0-Xc))-(D@x)*(X0-Xc)+ys*(D@(Y0-Yc))-(D@ys)*(Y0-Yc)
       -y*(D@Ys)+(D@y)*Ys-io*(3*x*Xs+y*(Y0-3*Yc)+3*ys*Ys)
       +2*ell*q.torsion*(ys*(X0-Xc)-y*Xs-x*(Y0-Yc))
       +2*ell*q.curvature*x*(Z0-Zc))
    return max(np.max(abs(c)),np.max(abs(d)))


def main():
    report={'scope':'NumPy algebra and independently assembled collocation checks; no native PR or VMEX run'}
    fields=[];gradients=[];hessians=[];maxwell=[]
    for sG in (-1,1):
        for spsi in (-1,1):
            for I2 in (0.,.6):
                q=reference_case(nphi=81,I2=I2,p2=-6e5,sG=sG,spsi=spsi)
                a=.01;closed=closed_local(q,a)[[0,7,23,44]];angular=angular_local(q,a)
                err=float(np.max(abs(closed-angular))); fields.append(err)
                assert_allclose(closed,angular,atol=2e-12,rtol=2e-10)
                local,D,H=local_plasma_jet(q,a)
                Dt,Ht=total_jet(q)
                Hc=Ht-H
                maxwell.append(float(max(np.max(abs(Hc-Hc.swapaxes(1,2))),
                                          np.max(abs(Hc-Hc.swapaxes(2,3))))))
                if I2==0:
                    unred=zero_current_jet(qsc_data(q),a)['D'].swapaxes(1,2)
                    compact=compact_zero_gradient(q,a)
                    gradients.append(float(np.max(abs(compact-unred))))
                    assert_allclose(compact,unred,atol=1e-10,rtol=2e-7)
                    simple=zero_current_hessian(qsc_data(q))
                    hessians.append(float(np.max(abs(simple-H))))
                    assert_allclose(simple,H,atol=2e-10,rtol=1e-9)
    report['max_closed_local_field_vs_8192_point_angular_integral_T']=max(fields)
    report['max_compact_zero_gradient_vs_unreduced_potential_curl_T_per_m']=max(gradients)
    report['max_zero_hessian_vs_general_cubic_polynomial_T_per_m2']=max(hessians)
    report['max_external_hessian_permutation_error_T_per_m2']=max(maxwell)
    report['equilibria_checked']=8
    q=reference_case(nphi=31,I2=.9,p2=-2e5,rc=(1.,),zs=(0.,),nfp=1,etabar=1.,B2c=0.)
    a=.01;val=closed_local(q,a)[0]
    exact=np.array([a*a*(MU0*q.p2+q.I2**2)/q.B0,0,
                    q.I2*a*a/2*(np.log(8/a)-1.25-MU0*q.p2/q.I2**2)])
    assert_allclose(val,exact,atol=1e-12,rtol=1e-12)
    report['axisymmetric_field_formula_max_error_T']=float(np.max(abs(val-exact)))
    # Both finite endpoints and the removed bounded interval are retained.
    errors=[]
    for delta in (.04,.08,.16):
        a=.004;dm=1.1;dp=.7
        inner=2*np.arcsinh(delta/a)
        outer=np.arcsinh(dm/a)+np.arcsinh(dp/a)-inner
        second=2*np.log(2*delta/a)+a*a/(2*delta*delta)
        second+=np.log(dm*dp/delta**2)+a*a/4*(1/dm**2+1/dp**2-2/delta**2)
        exact=np.arcsinh(dm/a)+np.arcsinh(dp/a)
        errors.append(abs(exact-second))
        assert abs(inner+outer-exact)<1e-14
    assert max(errors)-min(errors)<2e-14
    report['toy_second_order_error_independent_of_cutoff']=errors
    # Cubic Poisson and exterior matching: algebraic coefficients, random ellipses.
    rng=np.random.default_rng(328)
    from plasma_field import _cubic
    maxerr=0.
    for _ in range(50):
        x=rng.uniform(.5,2);s=rng.uniform(-1,1);chi=rng.choice([-1,1]);an,ab=rng.normal(size=2)
        c=_cubic(an,ab,x,s,chi)
        maxerr=max(maxerr,abs(6*c[0]+2*c[2]-2*np.pi*an),abs(2*c[1]+6*c[3]-2*np.pi*ab))
    report['cubic_poisson_max_residual']=maxerr
    assert maxerr<2e-14
    residuals=[explicit_compatibility_residuals(reference_case(nphi=121,I2=i,p2=-6e5)) for i in (0.,.6)]
    assert max(residuals)<1e-8
    report['explicit_second_order_compatibility_max_abs_residual_per_m']=float(max(residuals))
    Path(HERE/'algebra_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
