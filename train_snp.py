from torch.cuda import utilization
from torch.nn import utils
from tqdm import tqdm
from SDE import VPSDE

from ScoreNet import ScoreNet_temb2
from tensorboardX import SummaryWriter
import torchvision
import torch as t
from torch.utils.data import DataLoader
import math

import os
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, WhiteKernel
from preprocess import collate_fn, collate_fn_celeba32, collate_fn_gp, collate_fn_celeba64
from utils import ExponentialMovingAverage

from datasets.gp_data import get_datasets_single_gp, get_datasets_varying_kernel_gp, get_datasets_variable_hyp_gp
from datasets.celeba_data import CelebA32, CelebA64, train_dev_split
from datasets.eeg_data import EEGDataset,collate_fn_eeg_forcasting,collate_fn_eeg_interpolation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','celeba32','eeg','single_GP','varying_GP','varying_hp'])
args =parser.parse_args()
dataset = args.dataset



def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.
    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.
    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.
        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=True, continuous=True):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, time):

        labels = time*10
        score = model_fn(x, labels)
        std = sde.marginal_prob(t.zeros_like(x), time)[1]

        score = -score / std[:, None, None]
        return score

    return score_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, eps=1e-4):

    reduce_op = t.mean if reduce_mean else lambda *args, **kwargs: 0.5 * \
        t.sum(*args, **kwargs)

    def loss_fn(model, context_x, context_y, target_x, target_y):


        x = target_x
        y = target_y

        score_fn = get_score_fn(
            sde, model, train=train, continuous=continuous)
        time = t.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps

        z = t.randn_like(y)

        mean, std = sde.marginal_prob(y, time)
        perturbed_data = mean + std[:, None, None] * z
        #score = score_fn(t.cat([x, perturbed_data], dim=1), time)
        score = score_fn(t.cat([x, perturbed_data], dim=2), time)

        losses = t.square(score * std[:, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

        loss = t.mean(losses)
        return loss

    return loss_fn


def adjust_learning_rate(optimizer, step_num, warmup_step=4000, init_lr=0.0001):
    lr = init_lr * warmup_step**0.5 * \
        min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    if dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            './data', train=True, download=True,)

        model = ScoreNet_temb2().cuda()
        model.train()
        dloader = DataLoader(train_dataset, batch_size=128,
                             collate_fn=collate_fn, shuffle=True, num_workers=16)

        optim = t.optim.Adam(model.parameters(), lr=2e-4)

    elif dataset == 'celeba32':
        train_dataset, _ = train_dev_split(
            CelebA32(), dev_size=0.1, is_stratify=False)
        model = ScoreNet_temb2(dim_input=6,dim_output=3).cuda()
        model.train()

        dloader = DataLoader(train_dataset, batch_size=128,
                             collate_fn=collate_fn_celeba32, shuffle=True, num_workers=16)

        optim = t.optim.Adam(model.parameters(), lr=2e-4)


    elif dataset == "single_GP":
        k = 'Matern_Kernel'
        (datasets, _, __,) = get_datasets_single_gp()
        train_dataset = datasets[k]

        model = ScoreNet_temb2(dim_input=2, dim_output=1,
                                num_inds=16, dim_hidden=64, num_heads=2).cuda()
        model.train()

        dloader = DataLoader(train_dataset, batch_size=128,
                             collate_fn=collate_fn_gp, shuffle=True, num_workers=16)

        optim = t.optim.Adam(model.parameters(), lr=2e-4)

    elif dataset == "varying_GP":
        k = 'All_Kernels'
        (datasets, _, __,) = get_datasets_varying_kernel_gp()
        train_dataset = datasets[k]

        model = ScoreNet_temb2(dim_input=2, dim_output=1,
                                num_inds=16, dim_hidden=64, num_heads=2).cuda()
        
        model.train()
        dloader = DataLoader(train_dataset, batch_size=128,
                             collate_fn=collate_fn_gp, shuffle=True, num_workers=16)

        optim = t.optim.Adam(model.parameters(), lr=2e-4)

    elif dataset == "varying_hp_GP":
        k = 'Variable_Matern_Kernel'
        (datasets, _, __,) = get_datasets_variable_hyp_gp()
        train_dataset = datasets[k]
        model = ScoreNet_temb2(dim_input=3, dim_output=1,
                               num_inds=16, dim_hidden=64, num_heads=2).cuda()

        model.train()
        dloader = DataLoader(train_dataset, batch_size=128,
                             collate_fn=collate_fn_gp, shuffle=True, num_workers=16)

        optim = t.optim.Adam(model.parameters(), lr=2e-4)
    

    elif dataset=='eeg':
        train_dataset=EEGDataset(is_train=True)
        model = ScoreNet_temb2(dim_input=8,dim_output=7,num_inds=64,).cuda()
        model.train()

        dloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_eeg_interpolation,num_workers=16)
        optim = t.optim.Adam(model.parameters(), lr=2e-4)

    else:
        raise ValueError('Not implemented!')

    epochs = 2000

    sde = VPSDE()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999,)

    writer = SummaryWriter()
    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(dloader)
        for i, data in enumerate(pbar):
            global_step += 1
            context_x, context_y, target_x, target_y = data
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            target_y = target_y.cuda()

            loss_func = get_sde_loss_fn(sde=sde, train=True,)

            # pass through the latent model
            loss = loss_func(model, context_x, context_y, target_x, target_y)

            # Training step
            optim.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optim.step()
            ema.update(model.parameters())

            # Logging
            writer.add_scalars('training_loss', {
                'loss': loss,
            }, global_step)

        # save model by each epoch
        t.save({'model': model.state_dict(),
                'optimizer': optim.state_dict()},
               os.path.join('./checkpoint'+dataset, 'checkpoint_%d.pth.tar' % (epoch+1)))

        t.save(ema.state_dict(),
               os.path.join('./checkpoint'+dataset, 'ema_%d.pth.tar' % (epoch+1)))


if __name__ == '__main__':
    main()
