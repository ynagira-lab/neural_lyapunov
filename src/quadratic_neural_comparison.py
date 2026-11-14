"""Apples-to-apples grid comparison (NOT a formal certificate).

Uses the same closed-loop map, margin, 0.99 threshold, boundary grid rule,
and grid-count area for DARE, optimized quadratic, and Neural.

Run (from the src folder):
 python quadratic_neural_comparison.py --module neural_lyapunov_initial_training.py --weights out/neural_lyapunov_initial_weights.pt
If the weights are missing, omit --weights to retrain the initial Neural network.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.linalg import solve_discrete_are


def load_initial_controller(path):
    source = Path(path).read_text(encoding='utf-8')
    marker = '# 8. Origin check'
    pos = source.find(marker)
    if pos < 0:
        raise ValueError('Cannot find section 8 in initial-training script')
    prefix = source[:source.rfind('# ============================================================', 0, pos)]
    scope = {'__name__': 'initial_training_definitions'}
    exec(compile(prefix, str(path), 'exec'), scope)
    return scope['closed_loop_np'], scope['P']


def make_p(theta):
    angle, log_ratio = theta
    co, si = np.cos(angle), np.sin(angle)
    rot = np.array([[co, -si], [si, co]])
    return rot @ np.diag([np.exp(log_ratio), np.exp(-log_ratio)]) @ rot.T


def sublevel(v, margin, boundary, nonzero, cell_area):
    # Identical to initial-training numerical_sublevel: same 1e-8 tolerance and 0.99.
    bad = (margin > 1e-8) & nonzero
    upper = float(np.min(v[boundary]))
    if np.any(bad):
        upper = min(upper, float(np.min(v[bad])))
    c = max(0., 0.99 * upper)
    inside = v <= c
    return c, float(np.count_nonzero(inside) * cell_area), int(np.count_nonzero(inside & bad)), inside


def quadratic_metrics(p, x, fx, norm2, boundary, nonzero, cell_area, alpha):
    v = np.einsum('ni,ij,nj->n', x, p, x)
    vn = np.einsum('ni,ij,nj->n', fx, p, fx)
    return (*sublevel(v, vn-v+alpha*norm2, boundary, nonzero, cell_area), v)


def neural_values(x, fx, weights, seed, samples, epochs, learning_rate, alpha, x_range):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)

    class LyapunovNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        def forward(self, a):
            zero = torch.zeros((1,2), dtype=a.dtype, device=a.device)
            return torch.sum(a*a, dim=1, keepdim=True) + (self.net(a)-self.net(zero))**2

    net = LyapunovNet()
    if weights:
        state = torch.load(weights, map_location='cpu', weights_only=True)
        net.load_state_dict(state)
        print('Neural weights loaded:', weights)
    else:
        # Reproduce initial-training RNG sequence: numpy seed set before the initial-training draw.
        from scipy.linalg import solve_discrete_are  # no random state consumed
        train_x = np.random.uniform(-x_range,x_range,(samples,2))
        # The training map is supplied by the same initial-training module.
        train_fx = neural_values.controller(train_x)
        xt = torch.tensor(train_x, dtype=torch.float32)
        ft = torch.tensor(train_fx, dtype=torch.float32)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.relu(net(ft)-net(xt)+alpha*torch.sum(xt*xt,dim=1,keepdim=True)).mean()
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0 or epoch == epochs-1:
                print(f'Neural epoch {epoch}: loss={loss.item():.8g}')
        output_weights = Path('out/quadratic_neural_training_weights.pt')
        output_weights.parent.mkdir(exist_ok=True)
        torch.save(net.state_dict(), output_weights)
        print('Saved trained weights:', output_weights)
    net.eval()
    def evaluate(a):
        result=[]
        with torch.no_grad():
            for i in range(0,len(a),4096):
                result.append(net(torch.tensor(a[i:i+4096],dtype=torch.float32)).numpy().ravel())
        return np.concatenate(result).astype(float)
    return evaluate(x), evaluate(fx)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--module', required=True, help='initial-training script with SAC model accessible')
    ap.add_argument('--weights', default='', help='initial Neural .pt weights; omit to retrain')
    ap.add_argument('--grid', type=int, default=201)
    ap.add_argument('--maxiter', type=int, default=100)
    ap.add_argument('--alpha', type=float, default=.05)
    ap.add_argument('--range', dest='x_range', type=float, default=2.)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='out')
    args=ap.parse_args()
    if args.grid < 3: raise ValueError('Grid must be >=3')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    f, p_dare = load_initial_controller(args.module)
    axis=np.linspace(-args.x_range,args.x_range,args.grid)
    xx, yy=np.meshgrid(axis,axis)
    x=np.column_stack((xx.ravel(),yy.ravel()))
    fx=np.asarray(f(x),dtype=float)
    if fx.shape!=x.shape: raise ValueError('Closed-loop map shape mismatch')
    norm2=np.sum(x*x,axis=1)
    nonzero=norm2>1e-20  # initial-training TOL=1e-10 on radius
    boundary=np.zeros((args.grid,args.grid),dtype=bool)
    boundary[0,:]=boundary[-1,:]=boundary[:,0]=boundary[:,-1]=True
    boundary=boundary.ravel()
    cell_area=(2*args.x_range/(args.grid-1))**2
    def qm(p): return quadratic_metrics(p,x,fx,norm2,boundary,nonzero,cell_area,args.alpha)
    def obj(theta): return -qm(make_p(theta))[1]
    result=differential_evolution(obj,[(-np.pi/2,np.pi/2),(-3.,3.)],seed=args.seed,maxiter=args.maxiter,polish=True,tol=1e-6)
    p_opt=make_p(result.x)
    d=qm(p_dare)
    o=qm(p_opt)
    neural_values.controller=f
    v_nn, vn_nn=neural_values(x,fx,args.weights,args.seed,8192,3000,1e-3,args.alpha,args.x_range)
    n=sublevel(v_nn,vn_nn-v_nn+args.alpha*norm2,boundary,nonzero,cell_area)
    names=['Quadratic (DARE)','Quadratic (Optimized P)','Neural (Initial)']
    metrics=[d,o,n]
    print('\n=== FAIR GRID COMPARISON (same initial-training rule, not a certificate) ===')
    for name,m in zip(names,metrics):
        print(f'{name:25s}: c={m[0]:.8g}, grid_area={m[1]:.8g}, violations_inside={m[2]}')
    print('Optimized P (det=1):\n',p_opt)
    print('Neural/Optimized P area ratio:',n[1]/o[1] if o[1] else float('nan'))
    with (out/'quadratic_neural_comparison_metrics.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['method','level_c','grid_area_estimate','grid_margin_violations'])
        for name,m in zip(names,metrics): writer.writerow([name,m[0],m[1],m[2]])
    fig,ax=plt.subplots(figsize=(8,7))
    for v,m,name,color in [(d[4],d,names[0],'tab:blue'),(o[4],o,names[1],'tab:green'),(v_nn,n,names[2],'tab:orange')]:
        ax.contour(xx,yy,v.reshape(args.grid,args.grid),levels=[m[0]],colors=[color],linewidths=2)
        ax.plot([],[],color=color,label=f'{name} (area={m[1]:.4f})')
    ax.set(xlim=(-args.x_range,args.x_range),ylim=(-args.x_range,args.x_range),xlabel='State x1',ylabel='State x2',title='Quadratic and Neural sublevel comparison (grid-based)')
    ax.set_aspect('equal'); ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(out/'quadratic_neural_sublevel_comparison.png',dpi=200)
    plt.close(fig)
    print('Saved:',out/'quadratic_neural_comparison_metrics.csv', 'and',out/'quadratic_neural_sublevel_comparison.png')

if __name__=='__main__': main()
