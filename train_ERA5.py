import os

import numpy as np
import torch as th
from tensorboardX import SummaryWriter
from tqdm import tqdm

import datasets.era5_data as D
from datasets.era5_data import ERA5Converter
from SDE import VPSDE

from ScoreNet import ScoreNet_temb2
from train_snp import get_sde_loss_fn
from utils import ExponentialMovingAverage


def random_indices(num_indices, max_idx):
    """Generates a set of num_indices random indices (without replacement)
    between 0 and max_idx - 1.

    Args:
        num_indices (int): Number of indices to include.
        max_idx (int): Maximum index.
    """
    # It is wasteful to compute the entire permutation, but it looks like
    # PyTorch does not have other functions to do this
    permutation = th.randperm(max_idx)
    # Select first num_indices indices (this will be random since permutation is
    # random)
    return permutation[:num_indices]


device = 'cuda:3'

epochs = 2000
batchsize = 128
input_dim = 3
output_dim = 1
data_shape = (46, 90)
max_num_points = 2000

data_converter = ERA5Converter(device, data_shape, normalize_features=True)

model = ScoreNet_temb2(dim_input=4, dim_output=1).to(device)
model.train()
ema = ExponentialMovingAverage(model.parameters(), decay=0.999, )
optim = th.optim.Adam(model.parameters(), lr=2e-4)
writer = SummaryWriter('runs/experiment_1')
global_step = 0

sde = VPSDE()

for epoch in range(epochs):
    ERAloader = D.era5('EarthData/era5_temp2m_16x_train', batchsize)
    pbar = tqdm(ERAloader)

    loss_bucket = []
    for i, data in enumerate(pbar):
        batch, _ = data
        batch = batch.to(device)
        # Extract coordinates and features from data
        coordinates, features = data_converter.batch_to_coordinates_and_features(batch)
        # (batch_size, num_lats * num_lons, 3), (batch_size, num_lats * num_lons, 1)

        set_size = coordinates.shape[1]
        subset_indices = random_indices(max_num_points, set_size)
        # Select a subsample of coordinates and features
        coordinates = coordinates[:, subset_indices]
        features = features[:, subset_indices]

        loss_func = get_sde_loss_fn(sde=sde, train=True, )
        # pass through the latent model
        loss = loss_func(model, _, _, coordinates, features)

        # Training step
        optim.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optim.step()
        ema.update(model.parameters())

        # Logging
        writer.add_scalars('training_loss', {
            'loss': loss,
        }, global_step)
        global_step += 1
        loss_bucket.append(loss.item())

    print('epoch{} loss:'.format(epoch + 1), np.array(loss_bucket).mean())
    if epoch % 10 == 0:
        # save model by each 10 epoch
        th.save({'model': model.state_dict(),
                 'optimizer': optim.state_dict()},
                os.path.join('./checkpoint_dnp_ERA', 'checkpoint_%d.pth.tar' % (epoch + 1)))

        th.save(ema.state_dict(),
                os.path.join('./checkpoint_dnp_ERA', 'ema_%d.pth.tar' % (epoch + 1)))
