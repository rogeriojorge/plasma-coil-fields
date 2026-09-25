"""Leading near-axis plasma/coil field jets in a stellarator-symmetric QS branch.

The source is the current inside 0 <= r <= a, with fixed near-axis Taylor
coefficients. Field values retain a^2 log(a) and a^2; finite-I2 gradients and Hessians
retain their leading a^0 terms. For I2=0 the gradient retains the first
nonzero a^2 coefficient, calculated in zero_current.py. No theta quadrature or finite-radius surface
is used. Input is a qsc.Qsc object calculated with order='r2' or 'r3'.
All returned tensor indices are derivative-first and Cartesian.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.polynomial.legendre import leggauss

MU0 = 4e-7 * np.pi


def _frame(q: Any) -> np.ndarray:
    """Rows are t,n,b; columns are Cartesian x,y,z on q.phi (geometric angle)."""
    phi = np.asarray(q.phi)
    c, s = np.cos(phi), np.sin(phi)
    rows = []
    for name in ('tangent_cylindrical', 'normal_cylindrical', 'binormal_cylindrical'):
        v = np.asarray(getattr(q, name))
        rows.append(np.column_stack((v[:, 0]*c-v[:, 1]*s,
                                     v[:, 0]*s+v[:, 1]*c, v[:, 2])))
    return np.stack(rows, axis=1)


def axis_integral(q: Any, nquad: int = 160) -> np.ndarray:
    """Bounded circular-comparison integral, in observation cylindrical components.

    The integral is over the FULL axis, using geometric cylindrical angle phi.
    Hence its companion logarithm is log(8*(ds/dphi)/a), not log(8*L/a).
    Pairing +/- angular separation removes the opposite one-sided limits.
    Stable Fourier differences avoid subtraction of nearby Cartesian points.
    """
    if nquad < 8:
        raise ValueError('Use at least 8 Gauss nodes; verify quadrature convergence.')
    phi = np.asarray(q.phi, dtype=float)
    k = np.arange(len(q.rc)) * q.nfp
    rc, rs, zc, zs = [np.asarray(getattr(q, x), float) for x in ('rc','rs','zc','zs')]
    u, weights = leggauss(nquad)
    h = (u + 1) * np.pi / 2
    weights = weights * np.pi / 2
    result = np.zeros((phi.size, 3))
    for sign in (1, -1):
        step = sign * h
        phase = (phi[:, None, None] + step[None, :, None]) * k
        co, si = np.cos(phase), np.sin(phase)
        R = co @ rc + si @ rs
        Rp = (-si*k) @ rc + (co*k) @ rs
        Zp = (-si*k) @ zc + (co*k) @ zs
        mid = (phi[:, None, None] + step[None, :, None]/2) * k
        half = np.sin(step[:, None]*k/2)
        dcos = 2*np.sin(mid)*half
        dsin = -2*np.cos(mid)*half
        dR = dcos @ rc + dsin @ rs
        dZ = dcos @ zc + dsin @ zs
        ch, sh = np.cos(step), np.sin(step)
        separation = np.stack((dR+2*R*np.sin(step/2)**2, -R*sh, dZ), axis=-1)
        source_tangent = np.stack((Rp*ch-R*sh, Rp*sh+R*ch, Zp), axis=-1)
        distance = np.linalg.norm(separation, axis=-1)
        if np.any(distance <= 0):
            raise ValueError('Degenerate axis or coincident nonlocal source points.')
        integrand = np.cross(source_tangent, separation)/distance[..., None]**3
        integrand -= (np.asarray(q.curvature)[:, None, None]
                      *np.asarray(q.binormal_cylindrical)[:, None, :]
                      /(4*np.sin(h/2)[None, :, None]))
        result += np.einsum('nqi,q->ni', integrand, weights)
    return result


def _cubic(alpha_n, alpha_b, x, sigma, chi):
    """Coefficients (u^3,u^2 v,u v^2,v^3) of the resolved affine log potential."""
    den = (1+x*x)**2 + sigma*sigma
    hc = (x**4-1-sigma*sigma)/den
    hs = -2*chi*sigma*x*x/den
    real = (2*hc+hc*hc-hs*hs)*alpha_n + 2*hs*(1-hc)*alpha_b
    imag = 2*hs*(1+hc)*alpha_n + (-2*hc+hc*hc-hs*hs)*alpha_b
    return np.stack((np.pi*(3*alpha_n-real)/12,
                     np.pi*(alpha_b+imag)/4,
                     np.pi*(alpha_n+real)/4,
                     np.pi*(3*alpha_b-imag)/12), axis=-1)


def local_plasma_jet(q: Any, a: float):
    """Local B value (without global/log curvature), D, H in t,n,b components."""
    n = len(q.sigma)
    L, B0 = abs(q.G0)/q.B0, q.B0
    chi, j = q.sG*q.spsi, 2*q.sG*q.spsi*q.I2
    x, sig = np.asarray(q.X1c), np.asarray(q.sigma)
    k, tau = np.asarray(q.curvature), np.asarray(q.torsion)
    io, I2, pp = q.iotaN, q.I2, MU0*q.p2
    if abs(io) < 1e-12 and abs(pp*q.etabar) > 0:
        raise ValueError('The finite-pressure branch is singular at iota_N=0.')
    d = np.asarray(q.d_d_varphi)
    xp, sigp = d@x, d@sig
    den = (1+x*x)**2 + sig*sig
    X0,Xc,Xs,Y0,Yc,Ys = [np.asarray(getattr(q,z)) for z in
                          ('X20','X2c','X2s','Y20','Y2c','Y2s')]
    G2N = -pp*q.G0/B0**2-io*I2
    beta = 0.0 if pp == 0 else -4*q.spsi*pp*q.G0*q.etabar/(io*B0**3)
    pressure_n = np.zeros(n) if pp == 0 else 2*q.sG*pp*a*a*L*q.etabar*x*sig/(io*B0*den)
    pressure_b = np.zeros(n) if pp == 0 else -2*q.spsi*pp*a*a*L*q.etabar*x*(1+x*x)/(io*B0*den)
    shape_n = 2*I2*a*a*x*x/den**2*((sig*sig-(1+x*x)**2)*(Xs-chi*Yc)
                                -2*sig*(1+x*x)*(Xc+chi*Ys))
    shape_b = -2*chi*I2*a*a*x*x/den**2*((sig*sig-(1+x*x)**2)*(Xc+chi*Ys)
                                     +2*sig*(1+x*x)*(Xs-chi*Yc))
    Bt = a*a*(q.sG*pp/B0+I2/L*(io+chi*L*tau
                  +((1+x*x)*sigp-2*sig*x*xp)/den))
    B = np.stack((Bt, pressure_n+shape_n, pressure_b+shape_b), axis=-1)
    Dp = np.zeros((n,3,3))
    Dp[:,1,1]=j*chi*sig*x*x/den
    Dp[:,1,2]=j*(1+x*x+sig*sig)/den
    Dp[:,2,1]=-j*x*x*(1+x*x)/den
    Dp[:,2,2]=-Dp[:,1,1]
    # Actual affine source coefficients in physical normal/binormal coordinates.
    src_n = np.stack((-chi*q.spsi*B0*beta/x,
                       j*xp/(L*x)-2*chi*G2N*sig/L,
                       chi*j*sigp/(L*x*x)+j*tau-2*G2N*(1+sig*sig)/(L*x*x)),axis=-1)
    src_b = np.stack((np.zeros(n),-j*tau+2*G2N*x*x/L,
                       -j*xp/(L*x)+2*chi*G2N*sig/L),axis=-1)
    shape_alpha_n = 2*((1+sig*sig)*X0+(1-sig*sig)*Xc-2*sig*Xs)/(x*x) \
                    +2*chi*(Ys-sig*Y0+sig*Yc)
    shape_alpha_b = 2*chi*(Xs-sig*X0+sig*Xc)+2*x*x*(Y0-Yc)
    p = Xc+sig*Xs-chi*x*x*Ys
    qq = chi*Xs-chi*sig*Xc+x*x*Yc
    f=(1+x*x)**3-3*(1+x*x)*sig*sig
    g=chi*(3*(1+x*x)**2*sig-sig**3)
    lc=-4*np.pi*x*x*(p*f-qq*g)/(3*den**3)
    ls=-4*np.pi*x*x*(p*g+qq*f)/(3*den**3)
    boundary = np.stack((lc,3*ls,-3*lc,-ls),axis=-1)
    U2 = np.stack((np.pi*(1+x*x+sig*sig)/den,
                   -2*np.pi*chi*sig*x*x/den,
                   np.pi*x*x*(1+x*x)/den),axis=-1)
    uxU2 = np.column_stack((U2, np.zeros(n)))
    cubic = np.stack([-_cubic(src_n[:,i],src_b[:,i],x,sig,chi)/(2*np.pi)
                       for i in range(3)],axis=1)
    cubic[:,0] += j*k[:,None]/(4*np.pi)*(_cubic(np.ones(n),np.zeros(n),x,sig,chi)-uxU2)
    cubic[:,0] += j/(2*np.pi)*(_cubic(shape_alpha_n,shape_alpha_b,x,sig,chi)-boundary)
    m = -j*U2/(2*np.pi)
    Hp = np.zeros((n,3,3,3))
    Hp[:,1,1,0]=6*cubic[:,2,0]-2*cubic[:,1,1]
    Hp[:,1,2,0]=2*cubic[:,2,1]-2*cubic[:,1,2]
    Hp[:,2,2,0]=2*cubic[:,2,2]-6*cubic[:,1,3]
    Hp[:,1,1,1]=2*cubic[:,0,1]
    Hp[:,1,2,1]=2*cubic[:,0,2]
    Hp[:,2,2,1]=6*cubic[:,0,3]
    Hp[:,1,1,2]=2*k*m[:,0]-6*cubic[:,0,0]
    Hp[:,1,2,2]=k*m[:,1]-2*cubic[:,0,1]
    Hp[:,2,2,2]=2*k*m[:,2]-2*cubic[:,0,2]
    Hp[:,2,1]=Hp[:,1,2]
    connection=np.zeros((n,3,3));connection[:,0,1]=k;connection[:,1,0]=-k
    connection[:,1,2]=tau;connection[:,2,1]=-tau
    tangent=np.tensordot(d/L,Dp,axes=(1,0)) \
       -np.einsum('nad,ndg->nag',connection,Dp) \
       -np.einsum('ngd,nad->nag',connection,Dp)
    Hp[:,0]=tangent;Hp[:,:,0]=tangent
    return B,Dp,Hp


def total_jet(q: Any):
    """Total NAE gradient and Hessian, in physical t,n,b components."""
    n=len(q.sigma);d=np.asarray(q.d_d_varphi);L=abs(q.G0)/q.B0
    x,y,sig=np.asarray(q.X1c),np.asarray(q.Y1s),np.asarray(q.sigma)
    k,tau=np.asarray(q.curvature),np.asarray(q.torsion);io=q.iotaN
    connection=np.zeros((n,3,3));connection[:,0,1]=k;connection[:,1,0]=-k
    connection[:,1,2]=tau;connection[:,2,1]=-tau
    def derivative(v):
        return d@v+L*np.einsum('ni,nij->nj',v,connection)
    d1=np.stack((np.zeros(n),x,y*sig),axis=-1)
    d2=np.stack((np.zeros(n),np.zeros(n),y),axis=-1)
    h11=2*np.stack((q.Z20+q.Z2c,q.X20+q.X2c,q.Y20+q.Y2c),axis=-1)
    h12=2*np.stack((q.Z2s,q.X2s,q.Y2s),axis=-1)
    h22=2*np.stack((q.Z20-q.Z2c,q.X20-q.X2c,q.Y20-q.Y2c),axis=-1)
    V0=np.tile([L,0,0],(n,1));V1=derivative(d1)+io*d2;V2=derivative(d2)-io*d1
    V11=derivative(h11)+2*io*h12;V12=derivative(h12)+io*(h22-h11);V22=derivative(h22)-2*io*h12
    p0=q.B0**2/q.G0;p1=2*q.etabar*p0
    G2i=-MU0*q.p2*q.G0/q.B0**2
    p11=(q.B0**2*q.etabar**2+2*q.B0*(q.B20+q.B2c))/q.G0-q.B0**2*G2i/q.G0**2
    p22=2*q.B0*(q.B20-q.B2c)/q.G0-q.B0**2*G2i/q.G0**2
    B1=np.stack((p0*derivative(V0),p1*V0+p0*V1,p0*V2),axis=1)
    B2=np.zeros((n,3,3,3));B2[:,0,0]=p0*derivative(derivative(V0))
    B2[:,0,1]=p1*derivative(V0)+p0*derivative(V1);B2[:,1,0]=B2[:,0,1]
    B2[:,0,2]=p0*derivative(V2);B2[:,2,0]=B2[:,0,2]
    B2[:,1,1]=2*p11[:,None]*V0+2*p1*V1+p0*V11
    B2[:,1,2]=p1*V2+p0*V12;B2[:,2,1]=B2[:,1,2]
    B2[:,2,2]=2*p22[:,None]*V0+p0*V22
    Q=np.zeros((n,3,3));Q[:,0,0]=1/L;Q[:,1,1]=1/x;Q[:,2,1]=-sig/x;Q[:,2,2]=1/y
    xm=np.zeros_like(B2);xm[:,0,0,1]=L*L*k
    xm[:,0,1]=derivative(d1);xm[:,1,0]=xm[:,0,1]
    xm[:,0,2]=derivative(d2);xm[:,2,0]=xm[:,0,2]
    xm[:,1,1]=h11;xm[:,1,2]=h12;xm[:,2,1]=h12;xm[:,2,2]=h22
    Dt=np.einsum('nai,naj->nji',B1,Q)
    Ht=np.einsum('nabi,nak,nbj->nkji',B2,Q,Q) \
        -np.einsum('nai,nal,ncbl,nck,nbj->nkji',B1,Q,xm,Q,Q)
    return Dt,Ht


def plasma_coil_jet(q: Any, a: float, nquad: int = 160) -> dict[str,np.ndarray]:
    """Return plasma/coil B, D, H and axis coordinates on the native Qsc grid.

    No Maxwell symmetry is projected into the answer. Inspect the returned
    unprojected residuals and repeat with larger nphi and nquad.
    """
    if not np.isfinite(a) or a <= 0:
        raise ValueError('a must be a positive flux radius in the axis length units.')
    if getattr(q,'order','r1') not in ('r2','r3'):
        raise ValueError("Construct Qsc with order='r2' or order='r3'.")
    if getattr(q,'lasym',False) or abs(q.B2s)>1e-14:
        raise ValueError('This implementation assumes stellarator symmetry and B2s=0.')
    frame=_frame(q)
    local,Dp,Hp=local_plasma_jet(q,a)
    if q.I2 == 0:
        from zero_current import qsc_data, zero_current_jet
        Dp=zero_current_jet(qsc_data(q),a)['D']
    geom_cyl=axis_integral(q,nquad) if q.I2 != 0 else np.zeros((len(q.phi),3))
    ph=np.asarray(q.phi);c,s=np.cos(ph),np.sin(ph)
    geom=np.column_stack((geom_cyl[:,0]*c-geom_cyl[:,1]*s,
                          geom_cyl[:,0]*s+geom_cyl[:,1]*c,geom_cyl[:,2]))
    chi=q.sG*q.spsi;j=2*chi*q.I2;x=q.X1c;sig=q.sigma
    den=(1+x*x)**2+sig*sig
    local[:,1] += -q.I2*a*a*q.curvature*sig*x*x/(2*den)
    local[:,2] += j*a*a*q.curvature/4*(np.log(8*q.d_l_d_phi/a)-.5
                        -.5*np.log(den/(4*x*x))+x*x*(1+x*x)/den)
    Bp=j*a*a*geom/4+np.einsum('na,nai->ni',local,frame)
    Bt=q.sG*q.B0*frame[:,0]
    Dt,Ht=total_jet(q)
    rotate_D=lambda v: np.einsum('nai,nbj,nab->nij',frame,frame,v)
    rotate_H=lambda v: np.einsum('nai,nbj,nck,nabc->nijk',frame,frame,frame,v)
    return dict(phi=ph,axis=np.column_stack((q.R0*c,q.R0*s,q.Z0)),
                B_plasma=Bp,B_coil=Bt-Bp,
                D_plasma=rotate_D(Dp),D_coil=rotate_D(Dt-Dp),
                H_plasma=rotate_H(Hp),H_coil=rotate_H(Ht-Hp),
                axis_integral=geom,
                gradient_order=np.asarray(2 if q.I2 == 0 else 0),
                hessian_order=np.asarray(0))
