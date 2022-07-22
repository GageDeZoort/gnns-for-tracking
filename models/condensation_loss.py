import torch
from torch.nn.functional import mse_loss


def V_attractive(x, x_alpha, q_alpha, device="cpu"):
    norm_sq = torch.norm(x - x_alpha, dim=1) ** 2
    return norm_sq * q_alpha


def V_repulsive(x, x_alpha, q_alpha, device="cpu"):
    diffs = 1 - torch.norm(x - x_alpha, dim=1)
    hinges = torch.maximum(torch.zeros(len(x)).to(device), diffs)
    return hinges * q_alpha


def condensation_loss(beta, x, particle_id, device="cpu", q_min=1):
    loss = torch.tensor(0.0, dtype=torch.float).to(device)
    q = torch.arctanh(beta) ** 2 + q_min
    for pid in torch.unique(particle_id):
        p = pid.item()
        if p == 0:
            continue
        M = (particle_id == p).squeeze(-1)
        q_pid = q[M]
        x_pid = x[M]
        M = M.long()
        alpha = torch.argmax(q_pid)
        q_alpha = q_pid[alpha]
        x_alpha = x_pid[alpha]
        va = V_attractive(x, x_alpha, q_alpha, device=device)
        vr = V_repulsive(x, x_alpha, q_alpha, device=device)
        loss += torch.mean(q * (M * va + (1 - M) * vr))
    return loss


def background_loss(beta, x, particle_id, device="cpu", q_min=1, sb=10):
    loss = torch.tensor(0.0, dtype=torch.float).to(device)
    unique_pids = torch.unique(particle_id)
    beta_alphas = torch.zeros(len(unique_pids)).to(device)
    for i, pid in enumerate(unique_pids):
        p = pid.item()
        if p == 0:
            continue
        M = (particle_id == p).squeeze(-1)
        beta_pid = beta[M]
        alpha = torch.argmax(beta_pid)
        beta_alpha = beta_pid[alpha]
        beta_alphas[i] = beta_alpha

    n = (particle_id == 0).long()
    nb = torch.sum(n)
    if nb == 0:
        return torch.tensor(0, dtype=float)
    return torch.mean(1 - beta_alphas) + sb * torch.sum(n * beta) / nb


def L(p, t):
    return torch.sum(mse_loss(p, t, reduction="none"), dim=1)


def object_loss(pred, beta, truth, particle_id, high="efficiency", device="cpu"):
    n = (particle_id == 0).long()
    xi = (1 - n) * (torch.arctanh(beta)) ** 2
    mse = L(pred, truth)
    if high == "purity":
        return 1 / torch.sum(xi) * torch.mean(xi * mse)
    loss = torch.tensor(0.0, dtype=torch.float).to(device)
    K = torch.tensor(0.0, dtype=torch.float).to(device)
    for pid in torch.unique(particle_id):
        p = pid.item()
        if p == 0:
            continue
        M = (particle_id == p).squeeze(-1)
        xi_p = M * p
        weight = 1.0 / (torch.sum(xi_p))
        loss += weight * torch.sum(mse * xi_p)
        K += 1.0
    return loss / K
