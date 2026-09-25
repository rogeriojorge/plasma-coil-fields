"""Independent discretization of the direct Boozer-projection derivation.

This is NOT pyQSC. It supplies a duck-typed comparison object with the documented
Qsc data fields so that the plasma-field evaluation can be tested independently.
The default physical inputs are exactly the published 'r2 section 5.3' example.
"""
from types import SimpleNamespace
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import root
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plasma_field import MU0


def reference_case(nphi=151, I2=.9, p2=-600000., rc=(1.,.09), zs=(0.,-.09),
                   nfp=2, etabar=.95, B2c=-.7, B0=1., sG=1, spsi=1):
    if nphi % 2 == 0: raise ValueError('Use an odd nphi.')
    q=SimpleNamespace(nphi=nphi,nfp=nfp,rc=np.array(rc),rs=np.zeros(len(rc)),
        zs=np.array(zs),zc=np.zeros(len(zs)),etabar=etabar,B2c=B2c,B2s=0.,B0=B0,
        sG=sG,spsi=spsi,I2=I2,p2=p2,order='r2',lasym=False)
    ph=np.arange(nphi)*2*np.pi/(nphi*nfp);q.phi=ph
    freq=np.arange(len(rc))*nfp;co=np.cos(ph[:,None]*freq);si=np.sin(ph[:,None]*freq)
    R=co@q.rc;Z=si@q.zs;Rp=(-si*freq)@q.rc;Zp=(co*freq)@q.zs
    Rpp=(-co*freq**2)@q.rc;Zpp=(-si*freq**2)@q.zs
    Rppp=(si*freq**3)@q.rc;Zppp=(-co*freq**3)@q.zs
    v=np.stack((Rp,R,Zp),1);vv=np.stack((Rpp-R,2*Rp,Zpp),1)
    vvv=np.stack((Rppp-3*Rp,3*Rpp-R,Zppp),1)
    speed=np.linalg.norm(v,axis=1);L=np.mean(speed);cr=np.cross(v,vv)
    k=np.linalg.norm(cr,axis=1)/speed**3;tau=np.einsum('ni,ni->n',cr,vvv)/np.sum(cr*cr,axis=1)
    t=v/speed[:,None];b=cr/np.linalg.norm(cr,axis=1)[:,None];nor=np.cross(b,t)
    q.R0,q.Z0,q.R0p,q.Z0p=R,Z,Rp,Zp
    q.d_l_d_phi=speed;q.d_l_d_varphi=L;q.axis_length=2*np.pi*L;q.G0=sG*B0*L
    q.curvature=k;q.torsion=tau;q.tangent_cylindrical=t
    q.normal_cylindrical=nor;q.binormal_cylindrical=b
    ff=np.fft.fftfreq(nphi,1/nphi)*nfp
    dphi=np.fft.ifft(1j*ff[:,None]*np.fft.fft(np.eye(nphi),axis=0),axis=0).real
    d=dphi*(L/speed)[:,None];q.d_d_phi=dphi;q.d_d_varphi=d
    chi=sG*spsi;x=etabar/k;y=chi/x
    def sigres(vv):
        sig=vv[:-1];io=vv[-1]
        return np.r_[d@sig+io*(x**4+1+sig*sig)-2*x*x*q.G0/B0*(I2/B0-spsi*tau),sig[0]]
    guess=np.r_[np.zeros(nphi),np.mean(2*x*x*q.G0/B0*(I2/B0-spsi*tau)/(1+x**4))]
    sol=root(sigres,guess,tol=2e-11)
    if np.max(abs(sigres(sol.x)))>1e-9:raise RuntimeError(sol.message)
    sig,io=sol.x[:-1],sol.x[-1];q.iotaN=io;q.sigma=sig;q.X1c=x;q.Y1s=y;q.Y1c=y*sig
    # Helicity is only used to report iota, not to solve the helical equations.
    normalangle=np.unwrap(np.r_[np.arctan2(nor[:,2],nor[:,0]),np.arctan2(nor[0,2],nor[0,0])])
    helicity=int(np.rint((normalangle[-1]-normalangle[0])/(2*np.pi)))*chi
    q.helicity=helicity;q.iota=io-helicity*nfp
    pp=MU0*p2;q.G2=-pp*q.G0/B0**2-q.iota*I2
    beta=-4*spsi*pp*q.G0*etabar/(io*B0**3) if pp else 0.;q.beta_1s=beta
    Z0=-d@(x*x+y*y*(1+sig*sig))/(8*L)
    Zs=-(d@(2*y*y*sig)-2*io*(x*x+y*y*(sig*sig-1)))/(8*L)
    Zc=-(d@(x*x+y*y*(sig*sig-1))+4*io*y*y*sig)/(8*L)
    xp,yp,ysp=d@x,d@y,d@(y*sig)
    qc=xp-L*tau*y*sig;qs=-io*x-L*tau*y
    bc=ysp+io*y+L*tau*x;bs=yp-io*y*sig
    Xs=(d@Zs-2*io*Zc+(qc*qs+bc*bs)/(2*L))/(L*k)
    Xc=(d@Zc+2*io*Zs+L*B2c/B0-L*etabar**2/2+(qc*qc-qs*qs+bc*bc-bs*bs)/(4*L))/(L*k)
    def second(vv):
        X0,Y0=vv[:nphi],vv[nphi:]
        Ys=chi*(-k/2+(-Xc+sig*Xs-X0)/x**2)
        Yc=Y0+chi*(Xs+sig*Xc-sig*X0)/x**2
        cosine=x*(d@Xs)-xp*Xs+y*sig*(d@Ys)-ysp*Ys \
               -y*(d@(Y0+Yc))+yp*(Y0+Yc) \
               -io*(x*(X0+3*Xc)+y*sig*(Y0+3*Yc)+3*y*Ys) \
               +2*L*tau*(-y*(X0+Xc-sig*Xs)-x*Ys) \
               +2*L*k*x*Zs-chi*L*beta/2-3*sG*L*I2*etabar/B0
        sine=x*(d@(X0-Xc))-xp*(X0-Xc)+y*sig*(d@(Y0-Yc))-ysp*(Y0-Yc) \
             -y*(d@Ys)+yp*Ys \
             -io*(3*x*Xs+y*(Y0-3*Yc)+3*y*sig*Ys) \
             +2*L*tau*(y*sig*(X0-Xc)-y*Xs-x*(Y0-Yc)) \
             +2*L*k*x*(Z0-Zc)
        return np.r_[cosine,sine],Ys,Yc
    bvec=second(np.zeros(2*nphi))[0]
    M=np.column_stack([second(e)[0]-bvec for e in np.eye(2*nphi)])
    vals=np.linalg.solve(M,-bvec);res,Ys,Yc=second(vals)
    X0,Y0=vals[:nphi],vals[nphi:]
    q.B20=B0*(k*X0-(d@Z0)/L+etabar**2/2-pp/B0**2-(qc*qc+qs*qs+bc*bc+bs*bs)/(4*L*L))
    for name,val in zip(('X20','X2c','X2s','Y20','Y2c','Y2s','Z20','Z2c','Z2s'),(X0,Xc,Xs,Y0,Yc,Ys,Z0,Zc,Zs)):
        setattr(q,name,val)
    q.reference_residuals={'sigma':float(np.max(abs(sigres(sol.x)))),
                           'second_order':float(np.max(abs(res)))}
    return q
