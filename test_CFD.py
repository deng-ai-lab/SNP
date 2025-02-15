import pickle
import os
from functools import partial

import numpy as np
import tensorflow as tf2
import torch
import torch as th

from SDE import VPSDE
from ScoreNet import ScoreNet_temb2
from test_snp import get_score_fn, EulerMaruyamaPredictor, LangevinCorrector,  \
    mean_from_nth_dim, generate_fn_guide
from train_CFD import load_dataset
from utils import ExponentialMovingAverage
from render_CFD import save_frames, save_video
import score_likelihood

tf = tf2.compat.v1
tf.disable_eager_execution()


def spatial_mask(mesh_num, spatial_ratio=0.8):
    # np.random.seed(42)
    s_mask = np.random.choice(mesh_num, round(spatial_ratio * mesh_num), replace=False)

    return s_mask


# For testing
def traj2batch(mesh_pos, node_type, target_y, timestep=6):
    traj, n, c = node_type.shape
    assert traj == 600
    time_input = th.linspace(-1, 1, 600).to(target_y.device)
    window_start = range(0, traj, timestep)
    batchinput = [th.cat((node_type[0], mesh_pos[0],
                          time_input[None, t:t + timestep].repeat(n, 1)), dim=1)
                  for t in window_start]  # [(n, 1type + 2pos + 6time)].100
    batchoutput = [target_y[t:t + timestep, :, :].transpose(0, 1).reshape(n, -1)
                   for t in window_start]  # [(n, 12vel)].100
    batchinput, batchoutput = th.stack(batchinput, dim=0), th.stack(batchoutput, dim=0)

    return batchinput, batchoutput


def batch2traj(batchoutput, timestep=6):
    # batchoutput: [100, n, 12vel] -> [600, n, 2vel]
    batchsize, n, c = batchoutput.shape
    trajout = batchoutput.transpose(0, 1).reshape(n, batchsize * timestep, -1).transpose(0, 1)

    return trajout


if __name__ == '__main__':
    th.set_grad_enabled(False)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    th.set_num_threads(24)

    device = 'cuda:0'
    checkpoint_epoch = 18
    save_path = './results/cylinder_flow'
    test_traj_num = 100
    is_ema = True
    is_repaint = False
    is_guide = True
    mode = 'random'
    mask_ratio = 0.98

    is_visual = False
    is_log = False
    save_mode = 'frames'
    skip = 2

    timestep = 6

    input_dim = 9
    output_dim = 12

    model = ScoreNet_temb2(dim_input=input_dim + output_dim, dim_output=output_dim, num_inds=64).to(device)
    model.eval()

    sde = VPSDE()
    if is_ema:
        ema = ExponentialMovingAverage(model.parameters(), 0.999)
        ema.load_state_dict(th.load('./checkpoint_dnp_CFD/ema_{}.pth.tar'.format(checkpoint_epoch),
                                    map_location=device))
        ema.copy_to(model.parameters())
    else:
        model.load_state_dict(th.load('./checkpoint_dnp_CFD/checkpoint_{}.pth.tar'.format(checkpoint_epoch),
                                      map_location=device)['model'])

    score_func = get_score_fn(sde, model, False)
    Euler_solver = EulerMaruyamaPredictor(sde, score_func)
    Langevin_corr = LangevinCorrector(sde, score_func, 0.128, 3)
    likelihood_fn = score_likelihood.get_likelihood_fn(sde, rtol=1e-2, atol=1e-3)

    paint_fn = partial(generate_fn_guide, solver=Euler_solver, grad_step=1)

    criterion = th.nn.MSELoss(reduction='none')
    total_mse = []
    trajectories = []

    total_runs = 4
    runs_mse = []

    with tf.Session() as sess:
        for run in range(total_runs):
            total_mse = []
            ds = load_dataset('IrregularData/cylinder_flow', 'test', is_shuffle=True)
            test_sample = tf.data.make_one_shot_iterator(ds).get_next()
            for i in range(test_traj_num):
                input_dict = sess.run(test_sample)
                mesh_pos = torch.as_tensor(input_dict['mesh_pos']).to(device)
                node_type = torch.as_tensor(input_dict['node_type'], dtype=torch.float32).to(device)
                velocity = torch.as_tensor(input_dict['velocity']).to(
                    device)  # torch.Size([600, 1854, 2]) torch.float32

                target_x, target_y = traj2batch(mesh_pos, node_type, velocity, timestep=timestep)

                s_mask = spatial_mask(velocity.shape[1], mask_ratio)

                seen_y = target_y.clone()  # [100, n, 12vel]
                # sample from Gaussian as priors for target_y
                seen_y[:, s_mask] = th.randn_like(seen_y[:, s_mask], device=device)
                mask = target_y.new_ones(target_y.shape)
                mask[:, s_mask] = 0

                pred_y = paint_fn(target_x, seen_y, mask)
                # [100, n, 12vel] -> [600, n, 2vel]
                pred_y, target_y = batch2traj(pred_y, timestep), batch2traj(target_y, timestep)
                error = mean_from_nth_dim(criterion(pred_y[:, s_mask], target_y[:, s_mask]), dim=1)
                print('Test mse:', error.mean())
                total_mse.append(error)
                if is_log:
                    numpy_mse = [mse.item() for mse in th.cat(total_mse)]
                    np.savetxt("DNPMSE_CFD_m{}.txt".format(round(100*mask_ratio)), numpy_mse)

            print('Average test mse:', th.cat(total_mse).mean().item())
            print('Test mse std:', th.cat(total_mse).std().item())
            runs_mse.append(th.cat(total_mse).mean().item())

    print(runs_mse)

            # For visualization
            # seen_y[:, s_mask] = 0
            # seen_y = batch2traj(seen_y, timestep)

            # traj_ops = {
            #     'faces': input_dict['cells'],
            #     'mesh_pos': input_dict['mesh_pos'],
            #     'gt_velocity': input_dict['velocity'],
            #     'pred_velocity': pred_y.detach().cpu().numpy(),
            #     'seen_velocity': seen_y.detach().cpu().numpy(),
            # }
            # trajectories.append(traj_ops)

    # if is_ema:
    #     rollout_path = save_path + '/{}_e{}_m{}_ema.pkl'.format(mode, checkpoint_epoch, round(100*mask_ratio))
    # else:
    #     rollout_path = save_path + '/{}_e{}_m{}.pkl'.format(mode, checkpoint_epoch, round(100*mask_ratio))
    #
    # with open(rollout_path, 'wb') as fp:
    #     pickle.dump(trajectories, fp)
    # if is_visual:
    #     if save_mode == 'video':
    #         save_video(rollout_path, save_path, skip, name='e{}_m{}'.format(checkpoint_epoch, round(100*mask_ratio)))
    #     else:
    #         save_frames(rollout_path, save_path, skip=10, mode='pred')
    #         save_frames(rollout_path, save_path, skip=10, mode='gt')
    #         save_frames(rollout_path, save_path, skip=10, mode='seen')

    # print('Average test log likelihood:', th.stack(total_LL).mean())
    print('Average test mse:', th.cat(total_mse).mean().item())
    print('Test mse std:', th.cat(total_mse).std().item())

