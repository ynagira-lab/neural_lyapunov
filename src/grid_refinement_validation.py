"""Grid refinement validation of the saved Step60 Neural Lyapunov model.
Run in src:
%run grid_refinement_validation.py --module neural_lyapunov_initial_training.py --weights out/neural_lyapunov_step60_weights.pt --grids 201 401 801 --maxiter 100
No training. Numerical sampling only, NOT a stability certificate.
Requires quadratic_neural_comparison.py and baseline_checkpoint_training.py in same folder.
"""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import differential_evolution
from scipy.ndimage import label
import matplotlib.pyplot as plt
from quadratic_neural_comparison import load_initial_controller, make_p, sublevel
from baseline_checkpoint_training import LyapunovNet, values

HERE=Path(__file__).resolve().parent

def resolve(p):
    p=Path(p)
    return p if p.is_absolute() else HERE/p

def metric(v,vnext,norm2,boundary,nonzero,area,alpha):
    margin=vnext-v+alpha*norm2
    c,a,n,inside=sublevel(v,margin,boundary,nonzero,area)
    return dict(c=c,area=a,inside=inside,violations_inside=n,margin=margin,
                violations_total=int(np.count_nonzero((margin>1e-8)&nonzero)))

def quad_v(x,p):
    return np.einsum('ni,ij,nj->n',x,p,x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--module',default='neural_lyapunov_initial_training.py')
    ap.add_argument('--weights',default='out/neural_lyapunov_step60_weights.pt')
    ap.add_argument('--grids',nargs='+',type=int,default=[201,401,801])
    ap.add_argument('--maxiter',type=int,default=100)
    ap.add_argument('--alpha',type=float,default=.05)
    ap.add_argument('--range',dest='x_range',type=float,default=2.)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--out',default='out')
    args=ap.parse_args()
    out=resolve(args.out);out.mkdir(parents=True,exist_ok=True)
    f,p_dare=load_initial_controller(resolve(args.module))
    torch.set_num_threads(min(torch.get_num_threads(),4))
    net=LyapunovNet()
    net.load_state_dict(torch.load(resolve(args.weights),map_location='cpu',weights_only=True))
    net.eval()
    rows=[]
    for n in args.grids:
        if n<3: raise ValueError('grid must be >=3')
        axis=np.linspace(-args.x_range,args.x_range,n)
        xx,yy=np.meshgrid(axis,axis)
        x=np.column_stack([xx.ravel(),yy.ravel()])
        fx=np.asarray(f(x),dtype=float)
        norm2=(x*x).sum(axis=1)
        nonzero=norm2>1e-20
        b=np.zeros((n,n),dtype=bool)
        b[[0,-1],:]=True;b[:,[0,-1]]=True
        boundary=b.ravel()
        cell=(2*args.x_range/(n-1))**2
        v=values(net,x);vn=values(net,fx)
        neural=metric(v,vn,norm2,boundary,nonzero,cell,args.alpha)
        def quad_metric(p):
            return metric(quad_v(x,p),quad_v(fx,p),norm2,boundary,nonzero,cell,args.alpha)
        # Same optimization method, parameterization and seed as quadratic comparison and baseline checkpoint.
        opt=differential_evolution(lambda t:-quad_metric(make_p(t))['area'],
            [(-np.pi/2,np.pi/2),(-3.,3.)],seed=args.seed,maxiter=args.maxiter,
            tol=1e-6,polish=True)
        p_opt=make_p(opt.x)
        q=quad_metric(p_opt)
        d=quad_metric(p_dare)
        # Connected components: 8-neighbor topology of the sampled sublevel set.
        inside=neural['inside'].reshape(n,n)
        comps,num=label(inside,structure=np.ones((3,3),dtype=int))
        component_sizes=np.bincount(comps.ravel())[1:]
        boundary_contact=bool(np.any(inside & b))
        # Shape test: best quadratic least-squares fit to the Neural contour.
        # A fitted ellipse is descriptive, not a Lyapunov or area-optimal certificate.
        fig,ax=plt.subplots(figsize=(8,7))
        ax.contour(xx,yy,quad_v(x,p_dare).reshape(n,n),levels=[d['c']],colors=['tab:blue'],linewidths=1.5)
        ax.contour(xx,yy,quad_v(x,p_opt).reshape(n,n),levels=[q['c']],colors=['tab:green'],linewidths=2)
        contour=ax.contour(xx,yy,v.reshape(n,n),levels=[neural['c']],colors=['tab:red'],linewidths=2)
        segments=contour.allsegs[0]
        ellipse_rmse=np.nan
        if segments:
            seg=max(segments,key=len)
            # Fit x^T H x ~ 1 for origin-centered ellipse (same center as quadratic).
            z=np.column_stack([seg[:,0]**2,2*seg[:,0]*seg[:,1],seg[:,1]**2])
            coeff=np.linalg.lstsq(z,np.ones(len(z)),rcond=None)[0]
            H=np.array([[coeff[0],coeff[1]],[coeff[1],coeff[2]]])
            if np.min(np.linalg.eigvalsh(H))>0:
                ellipse_rmse=float(np.sqrt(np.mean((z@coeff-1)**2)))
                t=np.linspace(0,2*np.pi,500)
                circle=np.stack([np.cos(t),np.sin(t)])
                eig,U=np.linalg.eigh(H)
                pts=(U@np.diag(1/np.sqrt(eig))@circle).T
                ax.plot(pts[:,0],pts[:,1],ls='--',color='black',lw=1.5,label='Neural contour ellipse fit')
        ax.plot([],[],color='tab:blue',label='DARE %.4f'%d['area'])
        ax.plot([],[],color='tab:green',label='Optimized P %.4f'%q['area'])
        ax.plot([],[],color='tab:red',label='Neural Step60 %.4f'%neural['area'])
        ax.set(xlim=(-args.x_range,args.x_range),ylim=(-args.x_range,args.x_range),
               xlabel='State x1',ylabel='State x2',title=f'Grid refinement | N={n}: numerical sublevel comparison')
        ax.set_aspect('equal');ax.grid(True);ax.legend(fontsize=9)
        fig.tight_layout();fig.savefig(out/f'grid_refinement_{n}_contours.png',dpi=170);plt.close(fig)
        rows.append([n,d['area'],q['area'],neural['area'],neural['area']/q['area'],
                     neural['c'],q['c'],neural['violations_inside'],neural['violations_total'],
                     int(num),int(component_sizes.max()) if len(component_sizes) else 0,
                     boundary_contact,ellipse_rmse,p_opt[0,0],p_opt[0,1],p_opt[1,1]])
        print(f'Grid {n}: DARE={d["area"]:.6f}, Optimized={q["area"]:.6f}, Neural={neural["area"]:.6f}, ratio={neural["area"]/q["area"]:.5f}')
        print(f'  Neural: c={neural["c"]:.8f}, sampled violations_inside={neural["violations_inside"]}, components={num}, boundary_contact={boundary_contact}, ellipse_fit_rmse={ellipse_rmse:.5g}')
        print('  P_opt (det=1):',p_opt.tolist())
    path=out/'grid_refinement_validation_metrics.csv'
    with path.open('w',newline='',encoding='utf-8') as fp:
        writer=csv.writer(fp)
        writer.writerow(['grid','dare_area','optimized_p_area','neural_area','neural_to_opt_ratio',
            'neural_c','optimized_c','neural_violations_inside','neural_violations_total',
            'neural_components_8connected','largest_component_grid_points','touches_domain_boundary',
            'origin_centered_ellipse_fit_level_rmse','p_opt_11','p_opt_12','p_opt_22'])
        writer.writerows(rows)
    print('Saved:',path)
    print('CAUTION: zero sampled violations is NOT continuous-domain proof; 8-connected components are grid approximations.')
    print('CAUTION: boundary contact and ellipse fit are descriptive diagnostics, not formal ROA certification.')

if __name__=='__main__':main()
