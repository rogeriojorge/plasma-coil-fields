"""First nonzero plasma B and gradient for I2=0.

All vector components use (t,n,b). D is derivative-direction first.
Pass xp=jax.numpy for a differentiable array-only implementation.
Assumptions: fixed smooth Taylor coefficients, stellarator symmetry,
nonzero iotaN, and no extra resonant homogeneous current harmonics.
"""
from __future__ import annotations
import numpy as np

MU0 = 4e-7 * np.pi


def qsc_data(q):
    """Read the documented qsc data outside any JAX tracing context."""
    if q.I2 != 0:
        raise ValueError('This higher-accuracy formula requires I2=0.')
    if abs(q.iotaN) < 1e-12:
        raise ValueError('The selected finite-pressure branch requires nonzero iotaN.')
    names = ('B0','G0','p2','B2c','etabar','iotaN','sG','spsi','sigma',
             'curvature','torsion','X1c','Y1s','d_d_varphi',
             'X20','X2c','X2s','Y20','Y2c','Y2s','Z20','Z2c','Z2s')
    return {name: np.asarray(getattr(q, name)) for name in names}


def zero_current_jet(data, a, xp=np):
    """Return B, D, and the needed transverse potential derivatives.

    B error is O(B0*(a/Rg)^4*log(Rg/a)).
    D error is O((B0/Rg)*(a/Rg)^4*log(Rg/a)).
    Array derivatives propagate through ALL entries of data and a.
    This function does not solve equilibrium or select a physical edge.
    """
    d = {k: xp.asarray(v) for k, v in data.items()}
    x,y,sig=d['X1c'],d['Y1s'],d['sigma']
    k,tau=d['curvature'],d['torsion']
    B0,G0,io=d['B0'],d['G0'],d['iotaN']
    L=xp.abs(G0)/B0;chi=d['sG']*d['spsi'];Bb=d['spsi']*B0
    ds=d['d_d_varphi']/L;xs=ds@x;ys=ds@y;yss=ds@(y*sig)
    G2=-MU0*d['p2']*G0/B0**2
    beta=-4*d['spsi']*MU0*d['p2']*G0*d['etabar']/(io*B0**3)
    beta2=MU0*d['p2']*G0/(Bb*B0**2*io)*(1.5*d['etabar']**2-2*d['B2c']/B0)
    w=(1+x*x+1j*chi*sig)[:,None]
    F=MU0*d['p2']/B0*xp.stack((4*d['spsi']*L*d['etabar']*x/io,
                          2j*d['sG']*x*x,2*d['sG']*(1+1j*chi*sig)),axis=-1)/w
    Vt=chi*(-4*Bb*beta2+Bb*beta*k*x-8*G2*d['Z2s']/L)-8j*G2*d['Z2c']/L
    Vn=-chi*Bb*beta*(xs-tau*y*sig)-8*chi*G2*d['X2s']/L-1j*Bb*beta*tau*y-8j*G2*d['X2c']/L
    Vb=-chi*Bb*beta*(yss+tau*x)-8*chi*G2*d['Y2s']/L+1j*Bb*beta*ys-8j*G2*d['Y2c']/L
    V=xp.stack((Vt,Vn,Vb),axis=-1)
    shape=(d['X2c']+chi*d['Y2s']+1j*(d['Y2c']-chi*d['X2s']))[:,None]
    Q=x[:,None]**2/(2*w*w)*(V-4*F*shape)
    An=a*a/2*xp.real(F);Ab=-a*a/2*xp.imag(F)
    Ann=a*a*xp.real(Q+k[:,None]*F/2+k[:,None]*x[:,None]**2*F/(4*w))
    Anb=-a*a*xp.imag(Q+k[:,None]*F/4+k[:,None]*x[:,None]**2*F/(4*w))
    Abb=-a*a*xp.real(Q+k[:,None]*x[:,None]**2*F/(4*w))
    B=xp.stack((An[:,2]-Ab[:,1],Ab[:,0],-An[:,0]),axis=-1)
    rt=xp.stack((ds@B[:,0]-k*B[:,1],ds@B[:,1]+k*B[:,0]-tau*B[:,2],
                 ds@B[:,2]+tau*B[:,1]),axis=-1)
    rn=xp.stack((Ann[:,2]-Anb[:,1],
                 Anb[:,0]-ds@An[:,2]-tau*An[:,1]+tau*Ab[:,2],
                 ds@An[:,1]+k*An[:,0]-tau*An[:,2]-tau*Ab[:,1]-Ann[:,0]),axis=-1)
    rb=xp.stack((Anb[:,2]-Abb[:,1],
                 Abb[:,0]-ds@Ab[:,2]-tau*Ab[:,1]-tau*An[:,2],
                 ds@Ab[:,1]+k*Ab[:,0]-tau*Ab[:,2]+tau*An[:,1]-Anb[:,0]),axis=-1)
    return dict(B=B,D=xp.stack((rt,rn,rb),axis=1),An=An,Ab=Ab,Ann=Ann,Anb=Anb,Abb=Abb)


def cartesian_targets(data, a, frame, Btotal, Gtotal, xp=np):
    """Subtract plasma arrays using ESSOS/JAX output-first Jacobians.

    frame[N,alpha,i] has t,n,b rows and fixed Cartesian columns.
    Gtotal[N,i,j] is d(B_i)/d(x_j). No batch .T operation is used.
    """
    out=zero_current_jet(data,a,xp)
    frame=xp.asarray(frame)
    Bp=xp.einsum('nai,na->ni',frame,out['B'])
    Dp=xp.einsum('nai,nab,nbj->nij',frame,out['D'],frame)
    Gp=xp.swapaxes(Dp,-1,-2)
    return dict(Bplasma=Bp,gradBplasma=Gp,
                Bcoil=xp.asarray(Btotal)-Bp,gradBcoil=xp.asarray(Gtotal)-Gp)


def zero_current_hessian(data, xp=np):
    """Leading zero-I2 magnetic Hessian, in (t,n,b), derivative-first.

    H[N,k,j,i] = d_k d_j B_i. All leading entries with a t derivative
    vanish. Transverse entries have error O((B0/Rg^2)*epsilon^2 log).
    This is not the higher-order derivative of zero_current_jet's D.
    """
    d={k:xp.asarray(v) for k,v in data.items()}
    x,sig=d['X1c'],d['sigma'];g,p=d['sG'],d['spsi']
    L=xp.abs(d['G0'])/d['B0'];k=d['curvature'];io=d['iotaN']
    c=MU0*d['p2']/d['B0'];den=(1+x*x)**2+sig*sig
    pref=4*c*L*k/(io*den**2)
    cross=2*g*sig*x**4*(1+x*x)
    square=x**4*((1+x*x)**2-sig*sig)
    nn=xp.stack((-2*g*c*(1+sig*sig)/x**2,pref*cross,
                 pref*p*(den**2-square)),-1)
    nb=xp.stack((2*p*c*sig,-pref*p*square,-pref*cross),-1)
    bb=xp.stack((-2*g*c*x*x,-pref*cross,pref*p*square),-1)
    z=xp.zeros_like(nn)
    return xp.stack((xp.stack((z,z,z),1),xp.stack((z,nn,nb),1),
                     xp.stack((z,nb,bb),1)),1)
