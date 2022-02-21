import torch
import torch.nn as nn


def mse_loss(x_, x, reduction='sum'):
    mse = nn.MSELoss(reduction=reduction)(x_, x)
    return mse


def nll_loss(x_, x, reduction='none', loss_type='mse'):
    if loss_type == 'mse':
        return nn.MSELoss(reduction=reduction)(x_, x)
    elif loss_type == 'bce':
        x = torch.sigmoid(x)
        x_ = torch.sigmoid(x_)
        return nn.BCELoss(reduction=reduction)(x_, x)
    elif loss_type == 'bce_with_logits':
        x = torch.sigmoid(x)
        x_ = torch.sigmoid(x_)
        return nn.BCEWithLogitsLoss(reduction=reduction)(x_, x)
    else:
        raise NotImplemented


def kl_div(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if logvar2 is None:
        logvar2 = torch.zeros_like(mu1)

    return 0.5 * (
        logvar2 - logvar1 + (
            torch.exp(logvar1) + (mu1 - mu2).pow(2)
        ) / torch.exp(logvar2) - 1)


def kl_div_stn(mu, logvar):
    return 0.5 * (
        mu.pow(2) + torch.exp(logvar) - logvar - 1
    )


def recon_loss(x_, x):
    B, T = x.shape[0], x.shape[-1]
    nll_raw_0 = mse_loss(x_[:, :, 0], x[:, :, 0], 'none')
    nll_raw = mse_loss(x_[:, :, 1:], x[:, :, 1:], 'none')

    nll_m_0 = nll_raw_0.sum() / B
    nll_m = nll_raw.sum() / B

    total = nll_m_0 + nll_m
    return total


def domain_recon_loss(x_, x, D_, D, mu_c, logvar_c, kl_annealing_factor=1, loss_type='mse'):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = nll_loss(x_, x, 'none', loss_type)
    nll_m = nll_raw.sum() / B

    # K = D.shape[1]
    nll_raw_D = nll_loss(D_, D, 'none', loss_type)
    nll_m_D = nll_raw_D.sum() / B

    kl_raw_c = kl_div_stn(mu_c, logvar_c)
    kl_m_c = kl_raw_c.sum() / B

    total = kl_annealing_factor * kl_m_c + nll_m + nll_m_D

    return kl_m_c, nll_m, nll_m_D, total


def domain_loss(x_, x, D_, D, mu_c, logvar_c, mu_c_full, logvar_c_full, kl_annealing_factor=1, loss_type='mse', r1=1, r2=0, l=1):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = nll_loss(x_, x, 'none', loss_type)
    nll_m = nll_raw.sum() / B

    # K = D.shape[1]
    nll_raw_D = nll_loss(D_, D, 'none', loss_type) if D_ is not None else torch.zeros_like(D)
    nll_m_D = nll_raw_D.sum() / B

    kl_raw_c = kl_div(mu_c, logvar_c, mu_c_full, logvar_c_full)
    kl_m_c = kl_raw_c.sum() / B

    kl_raw_0 = kl_div_stn(mu_c, logvar_c)
    kl_m_0 = kl_raw_0.sum() / B

    total = kl_annealing_factor * (r1 * kl_m_c + r2 * kl_m_0) + (nll_m + nll_m_D) / l

    return kl_m_c, nll_m + nll_m_D, kl_m_0, total


def domain_loss_1(x_, x, D_, D, mu_c, logvar_c, mu_c_full, logvar_c_full, kl_annealing_factor=1, loss_type='mse', r1=1, r2=0, l=1):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = nll_loss(x_, x, 'none', loss_type)
    nll_m = nll_raw.sum() / B

    # K = D.shape[1]
    nll_raw_D = nll_loss(D_, D, 'none', loss_type) if D_ is not None else torch.zeros_like(D)
    nll_m_D = nll_raw_D.sum() / B

    # kl_raw_c = kl_div(mu_c, logvar_c, mu_c_full, logvar_c_full)
    kl_raw_c = kl_div(mu_c_full, logvar_c_full, mu_c, logvar_c)
    kl_m_c = kl_raw_c.sum() / B

    kl_raw_0 = kl_div_stn(mu_c, logvar_c)
    kl_m_0 = kl_raw_0.sum() / B

    total = kl_annealing_factor * (r1 * kl_m_c + r2 * kl_m_0) + (nll_m + nll_m_D) / l

    return kl_m_c, nll_m + nll_m_D, kl_m_0, total


def domain_loss_avg_D(x_, x, D_, D, mu_c, logvar_c, mu_c_full, logvar_c_full, kl_annealing_factor=1, loss_type='mse', r1=1, r2=0, l=1):
    B, T = x.shape[0], x.shape[-1]
    nll_raw = nll_loss(x_, x, 'none', loss_type)
    nll_m = nll_raw.sum() / B

    K = D.shape[1]
    nll_raw_D = nll_loss(D_, D, 'none', loss_type) if D_ is not None else torch.zeros_like(D)
    nll_m_D = nll_raw_D.sum() / B

    kl_raw_c = kl_div(mu_c, logvar_c, mu_c_full, logvar_c_full)
    kl_m_c = kl_raw_c.sum() / B

    kl_raw_0 = kl_div_stn(mu_c, logvar_c)
    kl_m_0 = kl_raw_0.sum() / B

    total = kl_annealing_factor * (r1 * kl_m_c + r2 * kl_m_0) + (nll_m + nll_m_D) / (K + 1) * l

    return kl_m_c, nll_m + nll_m_D, kl_m_0, total


def dmm_loss(x, x_q, x_p, mu1, var1, mu2, var2, kl_annealing_factor=1, r1=1, r2=0):
    B, T = x.shape[0], x.shape[-1]
    nll_raw_q = mse_loss(x_q, x[:, :, :T], 'none')
    nll_raw_p = mse_loss(x_p, x[:, :, :T], 'none')
    nll_m_q = nll_raw_q.sum() / B
    nll_m_p = nll_raw_p.sum() / B

    if mu1 is not None:
        kl_raw = kl_div(mu1, var1, mu2, var2)
        kl_raw, kl_raw_D = kl_raw[:, :, :, :-1], kl_raw[:, :, :, -1]
        kl_m = kl_raw.sum() / B
        kl_m_D = kl_raw_D.sum() / B
    else:
        kl_m = torch.zeros_like(x).sum() / B
        kl_m_D = torch.zeros_like(x).sum() / B

    loss = (kl_m + kl_m_D) * kl_annealing_factor + r1 * nll_m_q + r2 * nll_m_p

    return kl_m, nll_m_q, nll_m_p, loss
