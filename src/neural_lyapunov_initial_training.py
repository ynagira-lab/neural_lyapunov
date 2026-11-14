import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.linalg import solve_discrete_are
from stable_baselines3 import SAC


# ============================================================
# 0. Settings
# ============================================================

SEED = 0

np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "residual_sac_nonlinear_100k"

X_RANGE = 2.0

TRAIN_SAMPLES = 8192
EPOCHS = 3000

ALPHA = 0.05
LEARNING_RATE = 1e-3

GRID_N = 201
TOL = 1e-10

device = torch.device("cpu")


# ============================================================
# 1. System
# ============================================================

A = np.array([
    [1.0, 0.1],
    [0.0, 1.0]
])

B = np.array([
    [0.0],
    [0.1]
])

Q = np.diag([
    1.0,
    0.1
])

R = np.array([
    [0.01]
])


# ============================================================
# 2. LQR
# ============================================================

P = solve_discrete_are(
    A,
    B,
    Q,
    R
)

K = np.linalg.solve(
    R + B.T @ P @ B,
    B.T @ P @ A
)


print("========================================")
print("LQR")
print("========================================")

print("P =")
print(P)

print("\nK =")
print(K)


# ============================================================
# 3. Nonlinear plant
# ============================================================

def nonlinear_term_np(x):

    p = x[..., 0]

    out = np.zeros_like(x)

    out[..., 1] = (
        0.02 * p**3
    )

    return out


# ============================================================
# 4. Load Residual SAC
# ============================================================

print("\n========================================")
print("Loading Residual SAC")
print("========================================")

residual = SAC.load(
    MODEL_PATH
)

print(
    "Loaded:",
    MODEL_PATH
)


# ============================================================
# 5. Residual policy
# ============================================================

def residual_policy_np(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    single = (
        x.ndim == 1
    )

    if single:

        action, _ = residual.predict(
            x,
            deterministic=True
        )

        return float(
            action[0]
        )

    action, _ = residual.predict(
        x,
        deterministic=True
    )

    return action[:, 0].astype(
        np.float64
    )


# ============================================================
# 6. Zero-center the policy
# ============================================================

origin = np.array(
    [0.0, 0.0],
    dtype=np.float32
)

DU_ORIGIN = residual_policy_np(
    origin
)


def centered_residual_policy_np(x):

    return (
        residual_policy_np(x)
        - DU_ORIGIN
    )


print("\n========================================")
print("Policy centering")
print("========================================")

print(
    "Original residual at origin =",
    DU_ORIGIN
)

print(
    "Centered residual at origin =",
    centered_residual_policy_np(
        origin
    )
)


# ============================================================
# 7. Closed-loop map
#
# x+ = A x + B(-Kx + du) + d(x)
# ============================================================

def closed_loop_np(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    single = (
        x.ndim == 1
    )

    if single:

        du = centered_residual_policy_np(
            x
        )

        u_lqr = float(
            -(K @ x)[0]
        )

        u = (
            u_lqr
            + du
        )

        return (
            A @ x
            + B[:, 0] * u
            + nonlinear_term_np(x)
        )

    du = centered_residual_policy_np(
        x
    )

    u_lqr = (
        -(x @ K.T)
    )[:, 0]

    u = (
        u_lqr
        + du
    )

    x_next = (
        x @ A.T
        + u[:, None] * B[:, 0][None, :]
        + nonlinear_term_np(x)
    )

    return x_next


# ============================================================
# 8. Origin check
# ============================================================

x_origin_next = closed_loop_np(
    np.array([
        0.0,
        0.0
    ])
)

print("\n========================================")
print("Closed-loop origin check")
print("========================================")

print(
    "x_next from origin =",
    x_origin_next
)


# ============================================================
# 9. Neural Lyapunov network
#
# V(x) =
# ||x||^2
# +
# (g(x)-g(0))^2
# ============================================================

class LyapunovNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def g(self, x):

        return self.net(x)

    def forward(self, x):

        g_x = self.g(x)

        zero = torch.zeros(
            (1, 2),
            dtype=x.dtype,
            device=x.device
        )

        g_zero = self.g(
            zero
        )

        norm_sq = torch.sum(
            x**2,
            dim=1,
            keepdim=True
        )

        return (
            norm_sq
            + (g_x - g_zero)**2
        )


V_net = LyapunovNet().to(
    device
)

optimizer = optim.Adam(
    V_net.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 10. Training data
# ============================================================

X_train_np = np.random.uniform(
    low=-X_RANGE,
    high=X_RANGE,
    size=(
        TRAIN_SAMPLES,
        2
    )
)

X_next_np = closed_loop_np(
    X_train_np
)

X_train = torch.tensor(
    X_train_np,
    dtype=torch.float32,
    device=device
)

X_next = torch.tensor(
    X_next_np,
    dtype=torch.float32,
    device=device
)


# ============================================================
# 11. Train Neural Lyapunov
# ============================================================

loss_history = []


print("\n========================================")
print("Training Neural Lyapunov")
print("========================================")


for epoch in range(EPOCHS):

    optimizer.zero_grad()

    V_x = V_net(
        X_train
    )

    V_next = V_net(
        X_next
    )

    norm_sq = torch.sum(
        X_train**2,
        dim=1,
        keepdim=True
    )

    margin = (
        V_next
        - V_x
        + ALPHA * norm_sq
    )

    loss = torch.relu(
        margin
    ).mean()

    loss.backward()

    optimizer.step()

    loss_history.append(
        float(
            loss.item()
        )
    )

    if (
        epoch % 250 == 0
        or epoch == EPOCHS - 1
    ):

        print(
            f"epoch={epoch:4d}",
            f"loss={loss.item():.8e}"
        )


# ============================================================
# 12. Grid
# ============================================================

p_values = np.linspace(
    -X_RANGE,
    X_RANGE,
    GRID_N
)

v_values = np.linspace(
    -X_RANGE,
    X_RANGE,
    GRID_N
)

P_GRID, V_GRID = np.meshgrid(
    p_values,
    v_values
)

X_grid_np = np.column_stack([
    P_GRID.ravel(),
    V_GRID.ravel()
])

X_grid_next_np = closed_loop_np(
    X_grid_np
)

RADIUS = np.sqrt(
    P_GRID**2
    + V_GRID**2
)

NONZERO = (
    RADIUS > TOL
)


# ============================================================
# 13. Quadratic Lyapunov
# ============================================================

def quadratic_v_batch(x):

    return np.einsum(
        "bi,ij,bj->b",
        x,
        P,
        x
    )


V_P_now = quadratic_v_batch(
    X_grid_np
)

V_P_next = quadratic_v_batch(
    X_grid_next_np
)

DELTA_P = (
    V_P_next
    - V_P_now
).reshape(
    GRID_N,
    GRID_N
)


# ============================================================
# 14. Neural Lyapunov on grid
# ============================================================

V_net.eval()

with torch.no_grad():

    X_grid_t = torch.tensor(
        X_grid_np,
        dtype=torch.float32,
        device=device
    )

    X_grid_next_t = torch.tensor(
        X_grid_next_np,
        dtype=torch.float32,
        device=device
    )

    V_NN_now = (
        V_net(
            X_grid_t
        )
        .cpu()
        .numpy()
        .reshape(-1)
    )

    V_NN_next = (
        V_net(
            X_grid_next_t
        )
        .cpu()
        .numpy()
        .reshape(-1)
    )


DELTA_NN = (
    V_NN_next
    - V_NN_now
).reshape(
    GRID_N,
    GRID_N
)


# ============================================================
# 15. Violations
# ============================================================

VIOL_P = (
    (DELTA_P >= 0.0)
    & NONZERO
)

VIOL_NN = (
    (DELTA_NN >= 0.0)
    & NONZERO
)


# ============================================================
# 16. Approximate numerically checked radius
# ============================================================

def approximate_verified_radius(
    radius,
    delta_v,
    step=0.025
):

    radius_flat = radius.ravel()
    delta_flat = delta_v.ravel()

    max_r = np.max(
        radius_flat
    )

    candidate_r = np.arange(
        step,
        max_r + step,
        step
    )

    verified_r = 0.0

    for r in candidate_r:

        mask = (
            (radius_flat <= r)
            &
            (radius_flat > TOL)
        )

        if not np.any(mask):
            continue

        if np.all(
            delta_flat[mask] < 0.0
        ):

            verified_r = r

        else:

            break

    return verified_r


r_P = approximate_verified_radius(
    RADIUS,
    DELTA_P
)

r_NN = approximate_verified_radius(
    RADIUS,
    DELTA_NN
)


# ============================================================
# 17. Numerical results
# ============================================================

print("\n========================================")
print("Grid verification")
print("========================================")

print(
    "Grid points =",
    GRID_N * GRID_N
)

print()

print(
    "Quadratic max delta V =",
    np.max(
        DELTA_P[NONZERO]
    )
)

print(
    "Neural max delta V =",
    np.max(
        DELTA_NN[NONZERO]
    )
)

print()

print(
    "Quadratic violations =",
    np.sum(
        VIOL_P
    )
)

print(
    "Neural violations =",
    np.sum(
        VIOL_NN
    )
)

print()

print(
    "Quadratic numerically checked radius =",
    r_P
)

print(
    "Neural numerically checked radius =",
    r_NN
)


# ============================================================
# 18. Closest Neural violation
# ============================================================

if np.any(
    VIOL_NN
):

    violation_radius = RADIUS[
        VIOL_NN
    ]

    idx = np.argmin(
        violation_radius
    )

    p_viol = P_GRID[
        VIOL_NN
    ][idx]

    v_viol = V_GRID[
        VIOL_NN
    ][idx]

    dv_viol = DELTA_NN[
        VIOL_NN
    ][idx]

    print("\n----------------------------------------")
    print("Closest Neural violation")
    print("----------------------------------------")

    print(
        "radius =",
        violation_radius[idx]
    )

    print(
        "x =",
        np.array([
            p_viol,
            v_viol
        ])
    )

    print(
        "delta V =",
        dv_viol
    )

else:

    print("\nNo Neural Lyapunov violations on grid.")



# ============================================================
# 18b. Numerical sublevel-set comparison (grid only)
# ============================================================
from pathlib import Path
import csv

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.save(V_net.state_dict(), OUT_DIR / "neural_lyapunov_initial_weights.pt")

# Compare the same strict Lyapunov decrease margin used in training.
# Restrict sets to be strictly inside the sampled square.  A sublevel
# set enclosed by the square and satisfying the decrease condition on
# every point would be invariant, but the grid test is NOT a proof.
MARGIN_P = DELTA_P + ALPHA * RADIUS**2
MARGIN_NN = DELTA_NN + ALPHA * RADIUS**2


def numerical_sublevel(v_grid, margin_grid, name):
    v_grid = v_grid.reshape(GRID_N, GRID_N)
    margin_grid = margin_grid.reshape(GRID_N, GRID_N)
    boundary = np.zeros_like(v_grid, dtype=bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True
    # Limit the level below the first boundary point or violating point.
    # Origin excluded from strict decrease check, since margin(0)=0.
    bad = (margin_grid > 1e-8) & NONZERO
    upper = float(np.min(v_grid[boundary]))
    if np.any(bad):
        upper = min(upper, float(np.min(v_grid[bad])))
    c = max(0.0, 0.99 * upper)
    inside = v_grid <= c
    violations = int(np.count_nonzero(inside & bad))
    cell_area = (2 * X_RANGE / (GRID_N - 1))**2
    area = float(np.count_nonzero(inside) * cell_area)
    print(f"{name}: c={c:.8g}, area~={area:.8g}, grid violations={violations}, "
          f"boundary min V={np.min(v_grid[boundary]):.8g}")
    return c, area, violations, inside

print("\n========================================")
print("Numerical sublevel comparison (not formal verification)")
print("========================================")
c_p, area_p, bad_p, mask_p = numerical_sublevel(
    V_P_now, MARGIN_P, "Quadratic (DARE P)")
c_nn, area_nn, bad_nn, mask_nn = numerical_sublevel(
    V_NN_now, MARGIN_NN, "Neural")

with (OUT_DIR / "neural_initial_sublevel_metrics.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "level_c", "grid_area_estimate", "grid_margin_violations"])
    writer.writerow(["quadratic_dare", c_p, area_p, bad_p])
    writer.writerow(["neural", c_nn, area_nn, bad_nn])


print("Saved initial weights:", OUT_DIR / "neural_lyapunov_initial_weights.pt")
print("Saved numerical metrics:", OUT_DIR / "neural_initial_sublevel_metrics.csv")
