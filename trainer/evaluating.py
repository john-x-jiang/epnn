import os
import sys
import random
import numpy as np

import scipy.io as sio
import scipy.stats as stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_driver(model, data_loaders, metrics, hparams, exp_dir, data_tag):
    eval_config = hparams.evaluating
    loss_func = hparams.loss

    evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tag, eval_config, loss_func=loss_func)


def evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tag, eval_config, loss_func=None):
    torso_len = eval_config['torso_len']
    signal_source = eval_config['signal_source']
    omit = eval_config['omit']
    window = eval_config.get('window')
    k_shot = eval_config.get('k_shot')
    changable = eval_config.get('changable')
    sparse = eval_config.get('sparse')
    model.eval()
    n_steps = 0
    mses = {}
    tccs = {}
    sccs = {}
    dccs = {}

    q_recons = {}
    all_xs = {}
    all_labels = {}

    with torch.no_grad():
        data_names = list(data_loaders.keys())
        for data_name in data_names:
            data_loader = data_loaders[data_name]
            len_epoch = len(data_loader)
            for idx, data in enumerate(data_loader):
                signal, label = data.x, data.y
                signal = signal.to(device)
                label = label.to(device)

                if window is not None:
                    signal = signal[:, :, :window]

                x = signal[:, :-torso_len, omit:]
                y = signal[:, -torso_len:, omit:]

                if signal_source == 'heart':
                    source = x
                elif signal_source == 'torso':
                    source = y

                if k_shot is None:
                    if sparse is not None:
                        mask = data.mask[0]
                        in_source = source.clone()
                        in_source[:, mask != 3, :] = 0
                        
                        epi = np.where(mask == 3)[0]
                        selected_idx = np.arange(0, epi.shape[0], np.abs(sparse))
                        if sparse > 0:
                            selected_idx = np.delete(np.arange(0, epi.shape[0]), selected_idx)
                        
                        selected_epi = epi[selected_idx]
                        in_source[:, selected_epi, :] = 0
                    else:
                        in_source = source
                    physics_vars, statistic_vars = model(source, data_name, label)
                else:
                    D = data.D
                    D_label = data.D_label
                    D = D.to(device)
                    D_label = D_label.to(device)

                    if window is not None:
                        D = D[:, :, :window]

                    N, M, T = signal.shape
                    D = D.view(N, -1, M ,T)
                    D_x = D[:, :, :-torso_len, omit:]
                    D_y = D[:, :, -torso_len:, omit:]

                    if signal_source == 'heart':
                        D_source = D_x
                    elif signal_source == 'torso':
                        D_source = D_y
                    
                    if changable:
                        K = D.shape[1]
                        sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                        D_source = D_source[:, :sub_K, :]
                        D_label = D_label[:, :sub_K, :]
                    
                    if sparse is not None:
                        mask = data.mask[0]
                        in_source = source.clone()
                        in_D_source = D_source.clone()
                        in_source[:, mask != 3, :] = 0
                        in_D_source[:, :, mask != 3, :] = 0
                        
                        epi = np.where(mask == 3)[0]
                        selected_idx = np.arange(0, epi.shape[0], np.abs(sparse))
                        if sparse > 0:
                            selected_idx = np.delete(np.arange(0, epi.shape[0]), selected_idx)
                        
                        selected_epi = epi[selected_idx]
                        in_source[:, selected_epi, :] = 0
                        in_D_source[:, :, selected_epi, :] = 0
                    else:
                        in_source = source
                        in_D_source = D_source

                    physics_vars, statistic_vars = model(in_source, data_name, label, in_D_source, D_label)
                
                if loss_func == 'dmm_loss':
                    x_q, x_p = physics_vars
                    x_ = x_p
                elif loss_func == 'recon_loss' or loss_func == 'mse_loss':
                    x_, _ = physics_vars

                elif loss_func == 'domain_recon_loss' \
                    or loss_func == 'domain_loss' \
                    or loss_func == 'domain_loss_avg_D' \
                    or loss_func == 'domain_loss_1':
                    x_, _ = physics_vars
                else:
                    raise NotImplemented

                if idx == 0:
                    q_recons[data_name] = tensor2np(x_)
                    all_xs[data_name] = tensor2np(x)
                    all_labels[data_name] = tensor2np(label)
                else:
                    q_recons[data_name] = np.concatenate((q_recons[data_name], tensor2np(x_)), axis=0)
                    all_xs[data_name] = np.concatenate((all_xs[data_name], tensor2np(x)), axis=0)
                    all_labels[data_name] = np.concatenate((all_labels[data_name], tensor2np(label)), axis=0)

                for met in metrics:
                    if met.__name__ == 'mse':
                        mse = met(x_, x)
                        mse = tensor2np(mse)
                        if idx == 0:
                            mses[data_name] = mse
                        else:
                            mses[data_name] = np.concatenate((mses[data_name], mse), axis=0)
                    if met.__name__ == 'tcc':
                        if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                            x = tensor2np(x)
                            x_ = tensor2np(x_)
                        tcc = met(x_, x)
                        if idx == 0:
                            tccs[data_name] = tcc
                        else:
                            tccs[data_name] = np.concatenate((tccs[data_name], tcc), axis=0)
                    if met.__name__ == 'scc':
                        if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                            x = tensor2np(x)
                            x_ = tensor2np(x_)
                        scc = met(x_, x)
                        if idx == 0:
                            sccs[data_name] = scc
                        else:
                            sccs[data_name] = np.concatenate((sccs[data_name], scc), axis=0)
                    if met.__name__ == 'dcc':
                        if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                            x = tensor2np(x)
                            x_ = tensor2np(x_)
                        dcc = met(x_, x)
                        if idx == 0:
                            dccs[data_name] = dcc
                        else:
                            dccs[data_name] = np.concatenate((dccs[data_name], dcc), axis=0)
    
    for met in metrics:
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'tcc':
            print_results(exp_dir, 'tcc', tccs)
        if met.__name__ == 'scc':
            print_results(exp_dir, 'scc', sccs)
        if met.__name__ == 'dcc':
            print_results(exp_dir, 'dcc', dccs)
    
    save_result(exp_dir, q_recons, all_xs, all_labels, data_tag)


def personalize_driver(model, eval_data_loaders, pred_data_loaders, metrics, hparams, exp_dir, eval_tag, pred_tag):
    eval_config = hparams.evaluating
    loss_func = hparams.loss

    personalize_epoch(model, eval_data_loaders, pred_data_loaders, metrics, exp_dir, hparams, eval_tag, pred_tag, eval_config, loss_func=loss_func)


def personalize_epoch(model, eval_data_loaders, pred_data_loaders, metrics, exp_dir, hparams, eval_tag, pred_tag, eval_config, loss_func=None):
    torso_len = eval_config['torso_len']
    signal_source = eval_config['signal_source']
    omit = eval_config['omit']
    window = eval_config.get('window')
    k_shot = eval_config.get('k_shot')
    changable = eval_config.get('changable')
    sparse = eval_config.get('sparse')
    model.eval()
    n_steps = 0
    mses = {}
    tccs = {}
    sccs = {}
    dccs = {}

    q_recons = {}
    all_xs = {}
    all_labels = {}

    with torch.no_grad():
        data_names = list(pred_data_loaders.keys())
        for data_name in data_names:
            data_loader = pred_data_loaders[data_name]
            len_epoch = len(data_loader)

            data_iterator = iter(eval_data_loaders[data_name])
            for idx, data in enumerate(data_loader):

                try:
                    eval_data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(eval_data_loaders[data_name])
                    eval_data = next(data_iterator)

                signal, label = data.x, data.y
                signal = signal.to(device)
                label = label.to(device)

                if window is not None:
                    signal = signal[:, :, :window]

                x = signal[:, :-torso_len, omit:]
                y = signal[:, -torso_len:, omit:]

                eval_signal, eval_label = eval_data.x, eval_data.y
                eval_signal = eval_signal.to(device)
                eval_label = eval_label.to(device)

                if window is not None:
                    eval_signal = eval_signal[:, :, :window]

                eval_x = eval_signal[:, :-torso_len, omit:]
                eval_y = eval_signal[:, -torso_len:, omit:]

                if signal_source == 'heart':
                    source = x
                    eval_source = eval_x
                elif signal_source == 'torso':
                    source = y
                    eval_source = eval_y

                if k_shot is None:
                    if sparse is not None:
                        mask = data.mask[0]
                        in_source = source.clone()
                        in_source[:, mask != 3, :] = 0
                        
                        epi = np.where(mask == 3)[0]
                        selected_idx = np.arange(0, epi.shape[0], np.abs(sparse))
                        if sparse > 0:
                            selected_idx = np.delete(np.arange(0, epi.shape[0]), selected_idx)
                        
                        selected_epi = epi[selected_idx]
                        in_source[:, selected_epi, :] = 0
                    else:
                        in_source = source
                    physics_vars, statistic_vars = model.personalization(eval_source, data_name, label, eval_label)
                else:
                    D = eval_data.D
                    D_label = eval_data.D_label
                    D = D.to(device)
                    D_label = D_label.to(device)

                    if window is not None:
                        D = D[:, :, :window]

                    N, M, T = signal.shape
                    D = D.view(N, -1, M ,T)
                    D_x = D[:, :, :-torso_len, omit:]
                    D_y = D[:, :, -torso_len:, omit:]

                    if signal_source == 'heart':
                        D_source = D_x
                    elif signal_source == 'torso':
                        D_source = D_y
                    
                    if changable:
                        K = D.shape[1]
                        sub_K = np.random.randint(low=1, high=K+1, size=1)[0]
                        D_source = D_source[:, :sub_K, :]
                        D_label = D_label[:, :sub_K, :]
                    
                    if sparse is not None:
                        mask = data.mask[0]
                        in_source = source.clone()
                        in_D_source = D_source.clone()
                        in_D_source[:, :, mask != 3, :] = 0
                        
                        epi = np.where(mask == 3)[0]
                        selected_idx = np.arange(0, epi.shape[0], np.abs(sparse))
                        if sparse > 0:
                            selected_idx = np.delete(np.arange(0, epi.shape[0]), selected_idx)
                        
                        selected_epi = epi[selected_idx]
                        in_D_source[:, :, selected_epi, :] = 0
                    else:
                        in_D_source = D_source

                    physics_vars, statistic_vars = model.personalization(eval_source, data_name, label, eval_label, in_D_source, D_label)
                
                if loss_func == 'dmm_loss':
                    x_q, x_p = physics_vars
                    x_ = x_p
                elif loss_func == 'recon_loss' or loss_func == 'mse_loss':
                    x_, _ = physics_vars

                elif loss_func == 'domain_recon_loss' \
                    or loss_func == 'domain_loss' \
                    or loss_func == 'domain_loss_avg_D' \
                    or loss_func == 'domain_loss_1':
                    x_, _ = physics_vars
                else:
                    raise NotImplemented

                if idx == 0:
                    q_recons[data_name] = tensor2np(x_)
                    all_xs[data_name] = tensor2np(x)
                    all_labels[data_name] = tensor2np(label)
                else:
                    q_recons[data_name] = np.concatenate((q_recons[data_name], tensor2np(x_)), axis=0)
                    all_xs[data_name] = np.concatenate((all_xs[data_name], tensor2np(x)), axis=0)
                    all_labels[data_name] = np.concatenate((all_labels[data_name], tensor2np(label)), axis=0)

                for met in metrics:
                    if met.__name__ == 'mse':
                        mse = met(x_, x)
                        mse = tensor2np(mse)
                        if idx == 0:
                            mses[data_name] = mse
                        else:
                            mses[data_name] = np.concatenate((mses[data_name], mse), axis=0)
                    if met.__name__ == 'tcc':
                        if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                            x = tensor2np(x)
                            x_ = tensor2np(x_)
                        tcc = met(x_, x)
                        if idx == 0:
                            tccs[data_name] = tcc
                        else:
                            tccs[data_name] = np.concatenate((tccs[data_name], tcc), axis=0)
                    if met.__name__ == 'scc':
                        if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                            x = tensor2np(x)
                            x_ = tensor2np(x_)
                        scc = met(x_, x)
                        if idx == 0:
                            sccs[data_name] = scc
                        else:
                            sccs[data_name] = np.concatenate((sccs[data_name], scc), axis=0)
                    if met.__name__ == 'dcc':
                        if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                            x = tensor2np(x)
                            x_ = tensor2np(x_)
                        dcc = met(x_, x)
                        if idx == 0:
                            dccs[data_name] = dcc
                        else:
                            dccs[data_name] = np.concatenate((dccs[data_name], dcc), axis=0)

    for met in metrics:
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'tcc':
            print_results(exp_dir, 'tcc', tccs)
        if met.__name__ == 'scc':
            print_results(exp_dir, 'scc', sccs)
        if met.__name__ == 'dcc':
            print_results(exp_dir, 'dcc', dccs)
    
    save_result(exp_dir, q_recons, all_xs, all_labels, pred_tag, pred=True)


def print_results(exp_dir, met_name, mets):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    data_names = list(mets.keys())
    for data_name in data_names:
        print('{}: {} for full seq = {:05.5f}'.format(data_name, met_name, mets[data_name].mean()))
        with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
            f.write('{}: {} for full seq = {}\n'.format(data_name, met_name, mets[data_name].mean()))


def save_result(exp_dir, recons, all_xs, all_labels, data_tag, pred=False):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    if not pred:
        data_names = list(recons.keys())
        for data_name in data_names:
            sio.savemat(
                os.path.join(exp_dir, 'data/{}_{}.mat'.format(data_name, data_tag)), 
                {'recons': recons[data_name], 'inps': all_xs[data_name], 'label': all_labels[data_name]}
            )
    else:
        data_names = list(recons.keys())
        for data_name in data_names:
            sio.savemat(
                os.path.join(exp_dir, 'data/{}_{}_pred.mat'.format(data_name, data_tag)), 
                {'recons': recons[data_name], 'inps': all_xs[data_name], 'label': all_labels[data_name]}
            )


def tensor2np(t):
    return t.cpu().detach().numpy()
