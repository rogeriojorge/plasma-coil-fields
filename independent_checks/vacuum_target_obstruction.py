"""Symbolic local vacuum-target obstruction for divergence-free field Hessians.
No stellarator asymptotics are needed for the tensor identity.
"""
from pathlib import Path
import itertools, json
import sympy as s
out=Path(__file__).resolve().parent
inds=range(3); H={}
for i in inds:
    for j in inds:
        for k in range(j,3):
            H[i,j,k]=H[i,k,j]=s.Symbol(f'h{i}{j}{k}',real=True)
# Differentiated solenoidality: sum_i H[i,i,k]=0.
sub={H[0,0,0]:-H[1,0,1]-H[2,0,2],
     H[1,1,1]:-H[0,0,1]-H[2,1,2],
     H[2,2,2]:-H[0,0,2]-H[1,1,2]}
H={t:v.subs(sub,simultaneous=True) for t,v in H.items()}
S={(i,j,k):sum(H[t] for t in itertools.permutations((i,j,k)))/6 for i,j,k in itertools.product(inds,repeat=3)}
tr=[sum(S[i,j,j] for j in inds) for i in inds]
P={(i,j,k):S[i,j,k]-(int(i==j)*tr[k]+int(i==k)*tr[j]+int(j==k)*tr[i])/5 for i,j,k in S}
assert all(s.expand(sum(P[i,j,j] for j in inds))==0 for i in inds)
assert all(s.expand(P[i,j,k]-P[j,i,k])==0 for i,j,k in P)
C=s.Matrix(3,3,lambda k,l:sum(s.LeviCivita(k,j,i)*H[i,j,l] for i in inds for j in inds))
assert s.expand(s.trace(C))==0
sym=(C+C.T)/2; skew=(C-C.T)/2
norm=lambda T:sum(v*v for v in T)
distance=s.expand(sum((H[t]-P[t])**2 for t in H))
rhs=s.Rational(2,3)*norm(sym)+s.Rational(4,5)*norm(skew)
identity=s.expand(distance-rhs)
assert identity==0
# Orthogonality to every harmonic cubic potential field Hessian follows from
# symmetrization plus trace removal. Check distance^2 = ||H||^2 - ||P||^2.
assert s.expand(distance-sum(v*v for v in H.values())+sum(v*v for v in P.values()))==0
# Straight-cylinder example Bz=cp*(a^2-x^2-y^2).
cp=s.Symbol('cp',real=True)
hc={(i,j,k):s.Integer(0) for i,j,k in H}
hc[2,0,0]=hc[2,1,1]=-2*cp
symc={(i,j,k):sum(hc[t] for t in itertools.permutations((i,j,k)))/6 for i,j,k in hc}
tc=[sum(symc[i,j,j] for j in inds) for i in inds]
pc={(i,j,k):symc[i,j,k]-(int(i==j)*tc[k]+int(i==k)*tc[j]+int(j==k)*tc[i])/5 for i,j,k in hc}
floor=s.simplify(sum((hc[t]-pc[t])**2 for t in hc))
assert s.simplify(floor-s.Rational(32,5)*cp**2)==0
result={'identity_symbolic_residual':str(identity),
 'assumptions':['H_ijk = partial_j partial_k B_i','partial_i B_i=0 and its differentiated constraints','Frobenius norm counts all Cartesian indices','external coils are outside a neighborhood of the observation point'],
 'projection':'P_ijk = Sym(H)_ijk - (delta_ij t_k + delta_ik t_j + delta_jk t_i)/5, t_i=Sym(H)_ijj',
 'C_definition':'C_kl = epsilon_kji H_ijl = mu0 partial_l J_k',
 'distance_squared':'||H-P||_F^2 = (2/3)||Sym(C)||_F^2 + (4/5)||Skew(C)||_F^2',
 'coil_lower_bound':'Every realizable external-coil Hessian Hc satisfies ||H-Hc||_F^2 >= ||H-P||_F^2.',
 'cylinder_H_norm_squared':'8 cp^2','cylinder_minimum_squared_error':str(floor),'cylinder_relative_error_floor':str(s.sqrt(s.Rational(4,5))),
 'qualification':'An exact LOCAL compatibility lower bound, not a sufficiency result for a global realizable coil set. Projecting the total Hessian does not recover the actual free-space plasma subtraction, whose harmonic part is not fixed by local Maxwell constraints.',
 'novelty':'Derived in this audit as a proposed strengthening; no claim that harmonic tensor projection itself is new.'}
(out/'vacuum_target_obstruction.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
