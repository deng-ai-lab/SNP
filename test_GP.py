import os
import sys
from functools import partial

import numpy as np
import torch as th
from torch.utils.data import DataLoader

import score_likelihood
from utils import ExponentialMovingAverage

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(base_dir))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(base_dir))))
from SDE import VPSDE
from ScoreNet import ScoreNet_temb2
from test_snp import  get_score_fn, EulerMaruyamaPredictor, LangevinCorrector, \
    mean_from_nth_dim, generate_fn_guide
import datasets.gp_data as D


def load_ckpt(model, path, device, is_ema=True):
    if is_ema:
        ema = ExponentialMovingAverage(model.parameters(), 0.999)
        ema.load_state_dict(th.load('./checkpoint_dnp_gp/{}'.format(path), map_location=device))
        ema.copy_to(model.parameters())
    else:
        model.load_state_dict(
            th.load('./checkpoint_dnp_gp/{}'.format(path, map_location=device))['model'])
    model.eval()


def main():
    # np.random.seed(42)
    snp_paths = ['ema_282_snp_RBF.pth.tar',
                 'ema_2000_snp_periodic.pth.tar',
                 'ema_1500_snp_Matern.pth.tar']

    vk_path = ['ema_422_vk.pth.tar']
    vh_path = ['ema_819_vh.pth.tar']

    th.set_grad_enabled(False)
    device = 'cuda:5'
    is_log = True
    is_ema = True
    mode = 'random'
    data = 'single'
    m_ratio = 0.9
    batchsize = 256

    is_mse = False
    is_bpd = True
    is_visual = False
    run_times = 5

    if is_visual:
        is_mse = False
        is_bpd = False
        is_log = False
        batchsize = 1
        batchnum = 32
        target_x_collector, target_y_collector, pred_y_collector, seen_idx_collector = [], [], [], []
    assert is_bpd or is_mse or is_visual

    if data == 'single':
        (_, datasets, __,) = D.get_datasets_single_gp(data_root='./data', n_samples=50000)
        paths = snp_paths
    elif data == 'varying_kernel':
        (_, datasets, __,) = D.get_datasets_varying_kernel_gp(data_root='./data', n_samples=50000)
        paths = vk_path
    elif data == 'variable_hyp':
        (_, datasets, __,) = D.get_datasets_variable_hyp_gp(data_root='./data', n_samples=50000)
        paths = vh_path
    else:
        raise NotImplementedError
    collate_fn = D.collate_fn_gp
    num_total_points = 128

    model = ScoreNet_temb2(dim_input=2, dim_output=1, num_inds=16, dim_hidden=64, num_heads=2).to(device)

    sde = VPSDE(N=1000)
    score_func = get_score_fn(sde, model, False)
    Euler_solver = EulerMaruyamaPredictor(sde, score_func)
    Langevin_corr = LangevinCorrector(sde, score_func, 0.128, 3)
    likelihood_fn = score_likelihood.get_likelihood_fn(sde, rtol=1e-3, atol=1e-5)


    paint_fn = partial(generate_fn_guide, solver=Euler_solver, grad_step=0.5, eps=1e-3)


    criterion = th.nn.MSELoss(reduction='none')

    if mode == 'random':
        pass
    elif mode == 'forecast':
        mask_idx = np.arange(round((1 - m_ratio) * num_total_points), num_total_points)
    elif mode == 'crop':
        start = round((1 - m_ratio) / 2 * num_total_points)
        end = round((1 + m_ratio) / 2 * num_total_points)
        mask_idx = np.arange(start, end)
    else:
        raise NotImplementedError

    results_replay = []

    # total_runs = 4
    # runs_bpd = []
    # for run in range(total_runs):
    for i, (dataset_name, dataset) in enumerate(datasets.items()):
        total_bpd, total_mse = [], []
        load_ckpt(model, paths[i], device, is_ema)
        model.eval()
        ERAloader = DataLoader(dataset, batch_size=batchsize, collate_fn=collate_fn)
        print(dataset_name, len(ERAloader))
        for j, batch in enumerate(ERAloader):
            if mode == 'random':
                mask_idx = np.random.choice(num_total_points, round(m_ratio * num_total_points), replace=False)

            _, _, target_x, target_y = tuple(tensor.to(device) for tensor in batch)
            if is_bpd:
                bpd, z, nfe = likelihood_fn(model, target_x, target_y, mask_idx)
                # mean on batch
                step_bpd = bpd
                print('Test step bpd:', step_bpd.mean())
                total_bpd.append(step_bpd)

                if is_log:
                    numpy_bpd = [bpd.item() for bpd in th.cat(total_bpd)]
                    np.savetxt("DNPbpd_gp_{}_{}.txt".format(dataset_name, mode), numpy_bpd)

            if is_mse:
                seen_y = target_y.clone()
                # sample from Gaussian as priors for target_y
                seen_y[:, mask_idx] = th.randn_like(seen_y[:, mask_idx], device=device)
                mask = target_y.new_ones(target_y.shape)
                mask[:, mask_idx] = 0

                pred_y = paint_fn(target_x, seen_y, mask)  # Test

                error = mean_from_nth_dim(criterion(pred_y[:, mask_idx], target_y[:, mask_idx]), dim=1)
                print('Test mse:', error.mean())
                total_mse.append(error)

                if is_log:
                    numpy_mse = [mse.item() for mse in th.cat(total_mse)]
                    np.savetxt("DNPMSE_gp_{}_{}.txt".format(dataset_name, mode), numpy_mse)

            if is_visual:
                if not os.path.exists('results/' + dataset_name):
                    os.mkdir('results/' + dataset_name)
                target_x_collector.append(target_x)
                target_y_collector.append(target_y)
                seen_idx_collector.append(np.delete(np.arange(num_total_points), mask_idx))

                seen_y = target_y.clone()
                # sample from Gaussian as priors for target_y
                seen_y[:, mask_idx] = th.randn_like(seen_y[:, mask_idx], device=device)
                mask = target_y.new_ones(target_y.shape)
                mask[:, mask_idx] = 0
                sample_pred_y = []
                for i in range(run_times):
                    pred_y = paint_fn(target_x, seen_y, mask)
                    sample_pred_y.append(pred_y)
                sample_pred_y = th.stack(sample_pred_y)

                pred_y_collector.append(sample_pred_y)
                if j != batchnum:
                    continue
                target_x = th.cat(target_x_collector)
                target_y = th.cat(target_y_collector)
                pred_y = th.cat(pred_y_collector, dim=1)

                mu = pred_y.mean(dim=0, keepdim=False)
                sigma = pred_y.std(dim=0, keepdim=False)
                D.draw_gp(target_x, target_y, seen_idx_collector, mu, sigma,
                          dataset_name, mode, sample_times=run_times, samples=pred_y)
                break

        if not is_visual:
            print('Average test bpd:', th.cat(total_bpd).mean().item())
            print('Test bpd std:', th.cat(total_bpd).std().item())
            # runs_bpd.append((dataset_name, th.cat(total_bpd).mean().item(), th.cat(total_bpd).std().item()))
            print('Average test mse:', th.cat(total_mse).mean().item())
            print('Test mse std:', th.cat(total_mse).std().item())
            results_replay.append((th.cat(total_mse).mean().item(), th.cat(total_mse).std().item(),
                                   th.cat(total_bpd).mean().item(), th.cat(total_bpd).std().item()))

    if not is_visual:
        print('************')
        print('Test Summary with mode:{}, m_ratio:{}'.format(mode, m_ratio))
        for dataset_name, result in zip(datasets.keys(), results_replay):
            print('Average test mse on {}:'.format(dataset_name), result[0])
            print('Test mse std on {}:'.format(dataset_name), result[1])
            print('Average test bpd on {}:'.format(dataset_name), result[2])
            print('Test bpd std on {}:'.format(dataset_name), result[3])
            print('')
        print('************')

    # print(runs_bpd)

if __name__ == '__main__':
    main()
