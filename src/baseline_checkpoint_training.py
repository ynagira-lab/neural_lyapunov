"""Baseline checkpoint: replay baseline training steps 1-100 and diagnose area collapse.
Numerical grid experiment, NOT a formal stability/ROA certificate.

Run from src (Spyder):
%run baseline_checkpoint_training.py --module neural_lyapunov_initial_training.py --weights out/neural_lyapunov_initial_weights.pt --steps 100 --grid 201 --maxiter 100

Uses the SAME closed-loop system and quadratic-comparison sublevel scoring rule.
Saves into src/out regardless of current working directory.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution
from quadratic_neural_comparison import load_initial_controller, make_p, quadratic_metrics, sublevel

HERE = Path(__file__).resolve().parent

class LyapunovNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
    def forward(self, x):
        z = torch.zeros((1,2), dtype=x.dtype, device=x.device)
        return (x*x).sum(dim=1,keepdim=True) + (self.net(x)-self.net(z)).square()

def values(net, a, batch=4096):
    chunks=[]
    net.eval()
    with torch.no_grad():
        for i in range(0,len(a),batch):
            chunks.append(net(torch.as_tensor(a[i:i+batch],dtype=torch.float32)).cpu().numpy().ravel())
    return np.concatenate(chunks).astype(float)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--module',default='neural_lyapunov_initial_training.py')
    ap.add_argument('--weights',default='out/neural_lyapunov_initial_weights.pt')
    ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--grid',type=int,default=201)
    ap.add_argument('--maxiter',type=int,default=80,help='Differential evolution iterations for Quadratic baseline')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--alpha',type=float,default=0.05)
    ap.add_argument('--range',dest='x_range',type=float,default=2.0)
    ap.add_argument('--train-samples',type=int,default=8192)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--area-weight',type=float,default=1.0)
    ap.add_argument('--violation-weight',type=float,default=100.0)
    ap.add_argument('--boundary-weight',type=float,default=100.0)
    ap.add_argument('--temperature',type=float,default=0.12)
    ap.add_argument('--out',default='out')
    ap.add_argument('--inspect-steps',default='1,25,50,60,70,75,80,90,100')
    args=ap.parse_args()
    if args.grid<3 or args.steps<1: raise ValueError('grid>=3 and steps>=1 required')
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    def resolve(p):
        p=Path(p)
        return p if p.is_absolute() else HERE/p
    out=resolve(args.out); out.mkdir(parents=True,exist_ok=True)
    module=resolve(args.module); weights=resolve(args.weights)
    if not weights.is_file(): raise FileNotFoundError(f'initial weights not found: {weights}. Run initial Neural training first.')
    f,p_dare=load_initial_controller(module)
    axis=np.linspace(-args.x_range,args.x_range,args.grid)
    xx,yy=np.meshgrid(axis,axis)
    x=np.column_stack([xx.ravel(),yy.ravel()]); fx=np.asarray(f(x),dtype=np.float64)
    norm2=np.sum(x*x,axis=1); nonzero=norm2>1e-20
    boundary=np.zeros((args.grid,args.grid),dtype=bool)
    boundary[0,:]=boundary[-1,:]=boundary[:,0]=boundary[:,-1]=True
    boundary=boundary.ravel()
    cell_area=(2*args.x_range/(args.grid-1))**2
    def qm(p): return quadratic_metrics(p,x,fx,norm2,boundary,nonzero,cell_area,args.alpha)
    result=differential_evolution(lambda t:-qm(make_p(t))[1],
                                  [(-np.pi/2,np.pi/2),(-3,3)],seed=args.seed,
                                  maxiter=args.maxiter,tol=1e-6,polish=True)
    p_opt=make_p(result.x)
    dare=qm(p_dare); quad=qm(p_opt)

    net=LyapunovNet()
    net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    def score():
        v=values(net,x); vf=values(net,fx)
        m=sublevel(v,vf-v+args.alpha*norm2,boundary,nonzero,cell_area)
        return m,v
    before,v_before=score()
    best_area=before[1]; best_step=0
    best_state={k:v.detach().clone() for k,v in net.state_dict().items()}

    # Fixed sampled states + boundary points for differentiable surrogate.
    xt_np=rng.uniform(-args.x_range,args.x_range,(args.train_samples,2))
    # Include the complete evaluation grid as a training pool, to discourage
    # solutions exploiting sparse samples (still NOT continuous verification).
    pool=np.vstack([xt_np,x])
    pool_fx=np.asarray(f(pool),dtype=np.float64)
    pool_t=torch.as_tensor(pool,dtype=torch.float32)
    pool_f_t=torch.as_tensor(pool_fx,dtype=torch.float32)
    norm_t=(pool_t*pool_t).sum(dim=1,keepdim=True)
    boundary_t=torch.as_tensor(x[boundary],dtype=torch.float32)
    # Optimized level is trainable. Initialize at initial training's selected level.
    log_c=nn.Parameter(torch.tensor(np.log(max(before[0],1e-5)),dtype=torch.float32))
    optimizer=torch.optim.Adam(list(net.parameters())+[log_c],lr=args.lr)
    history=[]
    inspect_steps = {int(t.strip()) for t in args.inspect_steps.split(',') if t.strip()}
    inspect_steps.add(50); inspect_steps.add(100)
    inspect_steps = {t for t in inspect_steps if 1 <= t <= args.steps}
    snapshots = {}
    step60_state = None
    batch_size=min(4096,len(pool))
    print('Quadratic-comparison-equivalent baseline areas: DARE=%.6f, Optimized P=%.6f, Neural Initial=%.6f' % (dare[1],quad[1],before[1]))
    print('Starting baseline area-aware training; scores are NUMERICAL ONLY')
    for step in range(1,args.steps+1):
        net.train(); optimizer.zero_grad()
        idx=rng.integers(0,len(pool),size=batch_size)
        xb=pool_t[idx]; fb=pool_f_t[idx]; nb=norm_t[idx]
        c=log_c.exp()
        v=net(xb); dv=net(fb)-v+args.alpha*nb
        # Soft inclusion: smooth proxy for number of states in {V <= c}.
        inside=torch.sigmoid((c-v)/args.temperature)
        area_proxy=inside.mean()*(2*args.x_range)**2
        # Penalize margin violations inside the proposed level set.
        # Normalize margin by 1+||x||^2 for numerical conditioning.
        viol=(inside*torch.relu(dv)/(1+nb)).mean()
        # Keep level strictly below boundary values (0.99 initial-training rule
        # is re-applied during hard scoring, not optimized directly).
        vb=net(boundary_t)
        boundary_loss=torch.relu(c-0.99*vb).square().mean()
        loss=(-args.area_weight*area_proxy
              +args.violation_weight*viol
              +args.boundary_weight*boundary_loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0)
        optimizer.step()
        if step == 60:
            # Capture weights immediately AFTER the optimizer update.
            step60_state = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if step in inspect_steps or step==1 or step%50==0 or step==args.steps:
            metric, v_now=score()
            vf_now=values(net,fx)
            margin=vf_now-v_now+args.alpha*norm2
            bad=(margin>1e-8)&nonzero
            boundary_min=float(v_now[boundary].min())
            bad_min=float(v_now[bad].min()) if bad.any() else float('inf')
            level_from_boundary=0.99*boundary_min
            level_from_bad=0.99*bad_min
            limiting='violation' if level_from_bad < level_from_boundary else 'boundary'
            inside=v_now<=metric[0]
            snapshots[step]={
                'v':v_now.copy(), 'margin':margin.copy(), 'bad':bad.copy(),
                'c_eval':metric[0], 'c_train':float(log_c.detach().exp().item()),
                'area':metric[1], 'bad_total':int(bad.sum()),
                'bad_inside':int((bad&inside).sum()),
                'boundary_min':boundary_min,'bad_min':bad_min,
                'limiting':limiting,'surrogate':float(area_proxy.item()),
                'viol_loss':float(viol.item()),'boundary_loss':float(boundary_loss.item())
            }
            if step==1 or step%50==0 or step==args.steps:
                history.append((step,metric[1],float(area_proxy.item()),float(viol.item()),float(boundary_loss.item())))
            if metric[1]>best_area and metric[2]==0:
                best_area=metric[1]; best_step=step
                best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
            if step==1 or step%50==0 or step==args.steps:
                print(f'step={step:4d} grid_area={metric[1]:.4f} best={best_area:.4f} surrogate={area_proxy.item():.4f} violation={viol.item():.3g}')
    # Verify the saved state by loading it into a fresh model and scoring again.
    if step60_state is None or 60 not in snapshots:
        raise RuntimeError('Step 60 was not reached; run with --steps >= 60.')
    checkpoint_path = out/'neural_lyapunov_step60_weights.pt'
    torch.save(step60_state, checkpoint_path)
    reloaded = LyapunovNet()
    reloaded.load_state_dict(torch.load(checkpoint_path,map_location='cpu',weights_only=True))
    v60=values(reloaded,x)
    vf60=values(reloaded,fx)
    m60=sublevel(v60,vf60-v60+args.alpha*norm2,boundary,nonzero,cell_area)
    ref=snapshots[60]
    print('\n=== Step 60 save / reload verification ===')
    print('Before save: area=%.6f c=%.9f' % (ref['area'],ref['c_eval']))
    print('After reload: area=%.6f c=%.9f violations_inside=%d' % (m60[1],m60[0],m60[2]))
    print('Maximum absolute V difference:',np.max(np.abs(v60-ref['v'])))
    if not (np.isclose(m60[1],ref['area'],atol=1e-9) and np.isclose(m60[0],ref['c_eval'],atol=1e-7)):
        raise RuntimeError('Step 60 reloaded model does not match in-memory score.')
    with (out/'baseline_checkpoint_metrics.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.writer(fp)
        w.writerow(['step','grid','x_range','alpha','grid_area','level_c','violations_inside','total_grid_violations','quadratic_optimized_area'])
        w.writerow([60,args.grid,args.x_range,args.alpha,m60[1],m60[0],m60[2],ref['bad_total'],quad[1]])
    print('Checkpoint:',checkpoint_path)
    print('Metrics:',out/'baseline_checkpoint_metrics.csv')
    # Diagnostic snapshots are taken AFTER optimizer.step(), as in baseline training.
    diagnostic_rows=[]
    for step, snap in sorted(snapshots.items()):
        diagnostic_rows.append([
            step,snap['area'],snap['c_eval'],snap['c_train'],
            snap['boundary_min'],snap['bad_min'] if np.isfinite(snap['bad_min']) else '',
            snap['limiting'],snap['bad_total'],snap['bad_inside'],
            snap['surrogate'],snap['viol_loss'],snap['boundary_loss']
        ])
    with (out/'baseline_checkpoint_diagnostics.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.writer(fp)
        w.writerow(['step','grid_area','c_eval','c_train','min_boundary_V',
                    'min_bad_V','limiting_factor','total_grid_bad_points',
                    'bad_points_inside_selected_level','surrogate_area',
                    'train_violation_loss','train_boundary_loss'])
        w.writerows(diagnostic_rows)
    print('\\n=== Baseline checkpoint diagnostic replay ===')
    for row in diagnostic_rows:
        print('step=%3d area=%8.4f c_eval=%9.6f c_train=%9.6f bad=%5d limiter=%s' %
              (row[0],row[1],row[2],row[3],row[7],row[6]))
    fig, axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
    for ax, step in zip(axes,[50,100]):
        if step not in snapshots:
            ax.set_visible(False)
            continue
        snap=snapshots[step]
        ax.contour(xx,yy,snap['v'].reshape(args.grid,args.grid),
                   levels=[snap['c_eval']],colors=['tab:blue'],linewidths=2)
        badxy=x[snap['bad']]
        if len(badxy):
            ax.scatter(badxy[:,0],badxy[:,1],s=3,c='tab:red',alpha=.45,
                       rasterized=True,label='Decrease violations')
        ax.set(title=f"Step {step}: area={snap['area']:.4f}, c={snap['c_eval']:.4g}",
               xlabel='State x1',ylabel='State x2',
               xlim=(-args.x_range,args.x_range),ylim=(-args.x_range,args.x_range))
        ax.set_aspect('equal'); ax.grid(True)
        if len(badxy): ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('Baseline checkpoint: sublevel sets and violations (grid-based)')
    fig.tight_layout()
    fig.savefig(out/'baseline_checkpoint_violation_maps.png',dpi=180)
    plt.close(fig)
    steps_sorted=sorted(snapshots)
    fig,axs=plt.subplots(2,1,figsize=(8,7),sharex=True)
    axs[0].plot(steps_sorted,[snapshots[t]['area'] for t in steps_sorted],marker='o',label='Hard grid area')
    axs[0].plot(steps_sorted,[snapshots[t]['surrogate'] for t in steps_sorted],marker='.',label='Surrogate area')
    axs[0].legend(); axs[0].set_ylabel('Area'); axs[0].grid(True)
    axs[1].plot(steps_sorted,[snapshots[t]['c_eval'] for t in steps_sorted],marker='o',label='c_eval')
    axs[1].plot(steps_sorted,[snapshots[t]['c_train'] for t in steps_sorted],marker='.',label='c_train')
    axs[1].legend(); axs[1].set_ylabel('Level c'); axs[1].set_xlabel('Training step'); axs[1].grid(True)
    fig.tight_layout()
    fig.savefig(out/'baseline_checkpoint_area_level.png',dpi=180)
    plt.close(fig)
    print('Diagnostic outputs saved to:',out)
    return
    net.load_state_dict(best_state)
    after,v_after=score()
    print('\n=== Baseline common-grid comparison (NOT a certificate) ===')
    rows=[('Quadratic (DARE)',dare),('Quadratic (Optimized P)',quad),('Neural (Initial)',before),('Neural (Baseline best)',after)]
    for name,m in rows:
        print(f'{name:27s} area={m[1]:.6f} c={m[0]:.6f} violations_inside={m[2]}')
    print('Best training step:',best_step,'; area change:',after[1]-before[1])
    print('P optimized (det=1):\n',p_opt)
    with (out/'area_training_comparison_metrics.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.writer(fp); w.writerow(['method','level_c','grid_area_estimate','grid_margin_violations'])
        for name,m in rows: w.writerow([name,m[0],m[1],m[2]])
    with (out/'area_training_training_history.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.writer(fp); w.writerow(['step','grid_area','surrogate_area','violation_loss','boundary_loss']); w.writerows(history)
    torch.save(best_state,out/'area_training_neural_area_weights.pt')
    fig,ax=plt.subplots(figsize=(8,7))
    for name,m,v,color in [
        ('DARE',dare,dare[4],'tab:blue'),
        ('Optimized Quadratic',quad,quad[4],'tab:green'),
        ('Neural Initial',before,v_before,'tab:orange'),
        ('Neural Baseline',after,v_after,'tab:red')]:
        ax.contour(xx,yy,v.reshape(args.grid,args.grid),levels=[m[0]],colors=[color],linewidths=2)
        ax.plot([],[],color=color,label=f'{name}: {m[1]:.4f}')
    ax.set(xlim=(-args.x_range,args.x_range),ylim=(-args.x_range,args.x_range),
           xlabel='State x1',ylabel='State x2',title='Neural area optimization (grid-based)')
    ax.set_aspect('equal'); ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(out/'area_training_sublevel_comparison.png',dpi=200); plt.close(fig)
    if history:
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot([r[0] for r in history],[r[1] for r in history])
        ax.axhline(quad[1],linestyle='--',label='Optimized Quadratic')
        ax.set(xlabel='Step',ylabel='Grid area',title='Neural area optimization progression')
        ax.legend(); ax.grid(True); fig.tight_layout(); fig.savefig(out/'area_training_area_history.png',dpi=200); plt.close(fig)
    print('Saved to:',out)

if __name__=='__main__': main()
