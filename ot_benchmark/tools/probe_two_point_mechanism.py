"""High precision two-point identities, not a new OT solver benchmark."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import mpmath as mp
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sn(x):return mp.nstr(x,28)
def case(z,delta,perturb):
    z=mp.mpf(z);delta=mp.mpf(delta);k=mp.exp(-z);A=mp.mpf('.5');B=1-A;b1=A+delta;b2=B-delta;s=b1/b2
    m11=s*(B+A*k*k);m12=s*k;m21=k;m22=A+B*k*k
    diff=m11-m22;disc=mp.sqrt(diff*diff+4*m12*m21)
    rp=(diff+disc)/(2*m21);rm=-m12/(m21*rp)
    rho=(m11*m22-m12*m21)/(m21*rp+m22)**2
    r0=mp.exp(mp.mpf(perturb))
    def step(r):
        u1=A/(r+k);u2=B/(k*r+1)
        v1=b1/(u1+k*u2);v2=b2/(k*u1+u2)
        return v1/v2,[[u1*v1,u1*k*v2],[u2*k*v1,u2*v2]]
    def closed(n):
        w=(r0-rp)/(r0-rm)*mp.power(rho,n)
        return (rp-w*rm)/(1-w)
    # Independent primal root using stable quadratic formula.
    q=k*k;linear=delta+q*(1-delta)
    t=2*q*A*(B-delta)/(linear+mp.sqrt(linear**2+4*(1-q)*q*A*(B-delta)))
    optimal=[[A-t,t],[t+delta,B-delta-t]]
    _,scaled=step(rp)
    assert max(abs(scaled[i][j]-optimal[i][j]) for i in range(2) for j in range(2))<mp.mpf('1e-130')
    # Derivative of the log-ratio map agrees with analytic projective derivative.
    deriv=mp.diff(lambda h:mp.log(step(mp.exp(h))[0]),mp.log(rp))
    assert abs(deriv-rho)<mp.mpf('1e-130')
    direct=r0;err=mp.mpf(0)
    for n in range(1,31):
        direct,_=step(direct);err=max(err,abs(mp.log(direct/closed(n))))
    assert err<mp.mpf('1e-130')
    det=optimal[0][0]*optimal[1][1]-optimal[0][1]*optimal[1][0]
    assert abs(rho-det**2/(A*B*b1*b2))<mp.mpf('1e-130')
    checkpoints=[]
    for n in [1,100,10000,1000000]:
        rn,p=step(closed(n-1))
        assert abs(mp.log(rn/closed(n)))<mp.mpf('1e-130')
        residual=abs(sum(p[0])-A)+abs(sum(p[1])-B)
        checkpoints.append(dict(iteration=n,potential_difference_error=sn(abs(mp.log(rn/rp))/z),raw_row_l1=sn(residual),net_to_first=sn(p[1][0]-p[0][1])))
    return dict(inverse_epsilon=sn(z),delta=sn(delta),initial_log_ratio=perturb,kernel_cross_weight=sn(k),optimal_reverse_flow=sn(t),optimal_log_ratio=sn(mp.log(rp)),one_minus_local_rate=sn(1-rho),direct_30_max_log_error=sn(err),checkpoints=checkpoints)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dps',type=int,default=600);args=parser.parse_args();o=args.output;o.mkdir(parents=True,exist_ok=False)
    mp.mp.dps=args.dps
    write(o/'protocol.json',dict(dps=args.dps,inverse_epsilon=[8,32,128,1000],deltas=['0','1e-6','1e-5','1e-4','1e-3'],perturbed_balanced_initial_log_ratio='0.1',role='analytic model check; no empirical solver speed or held-out generalization claim'))
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as f:f.write(__file__,Path(__file__).name)
    rows=[]
    for z in [8,32,128,1000]:
        for d in ['0','1e-6','1e-5','1e-4','1e-3']:
            row=case(z,d,'0');rows.append(row);print(z,d,row['one_minus_local_rate'],row['checkpoints'][-1]['potential_difference_error'],flush=True)
        rows.append(case(z,'0','0.1'))
    write(o/'summary.json',dict(identity_checks_passed=True,rows=rows))
    write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
