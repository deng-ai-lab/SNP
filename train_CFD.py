import functools
import json
import os
import random

import numpy as np
import tensorflow as tf2
import torch as th
from tensorboardX import SummaryWriter

from SDE import VPSDE
from ScoreNet import ScoreNet_temb2
from train_snp import get_sde_loss_fn
from utils import ExponentialMovingAverage

tf = tf2.compat.v1
tf.disable_eager_execution()


def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                     for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out


def load_dataset(path, split, is_shuffle=True):
    """Load dataset."""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + '.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    # print(ds.output_shapes)
    # 'mesh_pos': TensorShape([600, None, 2])
    if is_shuffle:
        ds = ds.shuffle(1000)
    ds = ds.repeat(None)
    return ds.prefetch(10)


# For training
def traj2batch2(mesh_pos, node_type, target_y, timestep=6, batchsize=100, batchnum=10):
    traj, n, c = node_type.shape
    allbatch = []
    time_input = th.linspace(-1, 1, 600).to(target_y.device)
    for i in range(batchnum):
        window_start = random.sample(range(600 - timestep), batchsize)
        batchinput = [th.cat((node_type[0], mesh_pos[0],
                              time_input[None, t:t + timestep].repeat(n, 1)), dim=1)
                      for t in window_start]  # [(n, 1type + 2pos + timesteps)].batchsize
        batchoutput = [target_y[t:t + timestep, :, :].transpose(0, 1).reshape(n, -1)
                       for t in window_start]  # [(n, 2*timesteps)].batchsize
        batchinput, batchoutput = th.stack(batchinput, dim=0), th.stack(batchoutput, dim=0)
        allbatch.append((batchinput, batchoutput))

    return allbatch


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    th.set_num_threads(24)
    device = 'cuda:0'

    epochs = 100
    train_samples_num = 1000
    traj = 600

    batchnum = 4
    timestep = 20
    batchsize = 128

    input_dim = timestep + 3
    output_dim = timestep * 2

    model = ScoreNet_temb2(dim_input=input_dim + output_dim, dim_output=output_dim, num_inds=64).to(device)
    model.train()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999, )
    optim = th.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter('runs/experiment_CFD2_t{}'.format(timestep))
    global_step = 0

    sde = VPSDE()

    # resume = 69
    # ema = ExponentialMovingAverage(model.parameters(), 0.999)
    # ema.load_state_dict(th.load('./checkpoint_dnp_CFD2_t{}/ema_{}.pth.tar'.format(timestep, resume),
    #                             map_location=device))
    # ema.copy_to(model.parameters())

    for epoch in range(epochs):
        ds = load_dataset('IrregularData/cylinder_flow', 'train', is_shuffle=True)
        train_sample = tf.data.make_one_shot_iterator(ds).get_next()
        loss_bucket = []
        with tf.Session() as sess:
            # one epoch, 1000 samples
            for i in range(train_samples_num):
                input_dict = sess.run(train_sample)
                mesh_pos = th.as_tensor(input_dict['mesh_pos']).to(device)
                node_type = th.as_tensor(input_dict['node_type'], dtype=th.float32).to(device)
                velocity = th.as_tensor(input_dict['velocity']).to(
                    device)

                allbatch = traj2batch2(mesh_pos, node_type, velocity, timestep=timestep, batchsize=batchsize, batchnum=batchnum)

                for target_x, target_y in allbatch:
                    loss_func = get_sde_loss_fn(sde=sde, train=True, )
                    # pass through the latent model
                    _ = 0
                    loss = loss_func(model, _, _, target_x, target_y)

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

            th.save({'model': model.state_dict(),
                     'optimizer': optim.state_dict()},
                    os.path.join('./checkpoint_dnp_CFD2_t{}'.format(timestep), 'checkpoint_%d.pth.tar' % (epoch + 1)))

            th.save(ema.state_dict(),
                    os.path.join('./checkpoint_dnp_CFD2_t{}'.format(timestep), 'ema_%d.pth.tar' % (epoch + 1)))
