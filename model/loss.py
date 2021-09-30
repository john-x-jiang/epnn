import torch
import torch.nn as nn


def mse_loss(x_, x, reduction='sum'):
    mse = nn.MSELoss(reduction=reduction)(x_, x)
    return mse


def kl_div(mu1, var1, mu2=None, var2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if var2 is None:
        var2 = torch.zeros_like(mu1)

    return 0.5 * (
        var2.log() - var1.log() + (
            var1 + (mu1 - mu2).pow(2)
        ) / var2 - 1)
    # return 0.5 * (
    #     var2 - var1 + (
    #         torch.exp(var1) + (mu1 - mu2).pow(2)
    #     ) / torch.exp(var2) - 1)


def dmm_loss(x, x_q, x_p, mu1, var1, mu2, var2, kl_annealing_factor=1, r1=1, r2=0):
    kl_raw = kl_div(mu1, var1, mu2, var2)
    kl_raw, kl_raw_D = kl_raw[:, :, :, :-1], kl_raw[:, :, :, -1]
    B, T = x.shape[0], x.shape[-1]
    nll_raw_q = mse_loss(x_q, x[:, :, :T], 'none')
    nll_raw_p = mse_loss(x_p, x[:, :, :T], 'none')

    kl_m = kl_raw.sum() / B
    kl_m_D = kl_raw_D.sum() / B
    nll_m_q = nll_raw_q.sum() / B
    nll_m_p = nll_raw_p.sum() / B

    loss = (kl_m + kl_m_D) * kl_annealing_factor + r1 * nll_m_q + r2 * nll_m_p

    return kl_m, nll_m_q, nll_m_p, loss
