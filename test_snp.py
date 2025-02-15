from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import alpha
from torch.utils.data import DataLoader
from ScoreNet import ScoreNet_temb2 
import abc
from SDE import VPSDE
import torch as t
import math
import torchvision
from utils import ExponentialMovingAverage
from datasets.celeba_data import CelebA32, train_dev_split
from ScoreNet import ScoreNet_temb2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','celeba32'])
args =parser.parse_args()
dataset = 'celeba32'
device = 'cuda:0'


def mean_from_nth_dim(t, dim):
    """Mean all dims from `dim`. E.g. mean_from_nth_dim(torch.rand(2,3,4,5), 2).shape = [2,3]"""
    return t.view(*t.shape[:dim], -1).mean(-1)

def collate_fn_mnist(batch):

    # Puts each data field into a tensor with outer dimension batch size
    assert isinstance(batch[0], tuple)
    trans = torchvision.transforms.ToTensor()
    batch_size = len(batch)

    num_total_points = 784
    #num_context = np.random.randint(10, num_total_points)  # half of total points
    num_context=392
    c_index_batch = list()

    context_x, context_y, target_x, target_y = list(), list(), list(), list()

    for d, _ in batch:
        d = trans(d)
        total_idx = range(784)
        total_idx = list(map(lambda x: (x//28, x % 28), total_idx))
        c_idx = np.random.choice(range(num_context), num_context, replace=False)
        c_idx = c_idx[:num_context]
        c_index_batch.append(t.tensor(c_idx))
        c_idx = list(map(lambda x: (x//28, x % 28), c_idx))
        c_x, c_y, total_x, total_y = list(), list(), list(), list()
        for idx in c_idx:
            c_y.append(d[:, idx[0], idx[1]])
            c_x.append((idx[0] / (27.), idx[1] / (27.)))
        for idx in total_idx:
            total_y.append(d[:, idx[0], idx[1]])
            total_x.append((idx[0] / (27.), idx[1] / (27.)))
        c_x, c_y, total_x, total_y = list(
            map(lambda x: t.FloatTensor(x), (c_x, c_y, total_x, total_y)))
        context_x.append(c_x)
        context_y.append(c_y)
        target_x.append(total_x)
        target_y.append(total_y)

    context_x = t.stack(context_x, dim=0)
    context_y = t.stack(context_y, dim=0).unsqueeze(-1)
    target_x = t.stack(target_x, dim=0)
    target_y = t.stack(target_y, dim=0).unsqueeze(-1)
    c_index_batch = t.stack(c_index_batch, dim=0)

    return context_x, context_y, target_x, target_y, c_index_batch


def collate_fn_celeba32(batch):

    # Puts each data field into a tensor with outer dimension batch size
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)

    num_total_points = 1024

    num_context=512
    c_index_batch = list()

    context_x, context_y, target_x, target_y = list(), list(), list(), list()

    for d, _ in batch:
        total_idx = range(1024)

        total_idx = list(map(lambda x: (x//32, x % 32), total_idx))
        c_idx = np.random.choice(range(num_context,num_total_points), num_context, replace=False)
        c_idx = c_idx[:num_context]

        c_index_batch.append(t.tensor(c_idx))
        c_idx = list(map(lambda x: (x//32, x % 32), c_idx))

        c_x, c_y, total_x, total_y = list(), list(), list(), list()
        for idx in c_idx:
            c_y.append(d[:, idx[0], idx[1]])
            c_x.append((idx[0] / (31.), idx[1] / (31.)))
        for idx in total_idx:
            total_y.append(d[:, idx[0], idx[1]])
            total_x.append((idx[0] / (31.), idx[1] / (31.)))
        c_x, total_x = list(map(lambda x: t.FloatTensor(x), (c_x, total_x,)))
        c_y, total_y = list(map(lambda x: t.stack(x), (c_y, total_y,)))
        context_x.append(c_x)
        context_y.append(c_y)
        target_x.append(total_x)
        target_y.append(total_y)

    context_x = t.stack(context_x, dim=0)
    context_y = t.stack(context_y, dim=0)
    target_x = t.stack(target_x, dim=0)
    target_y = t.stack(target_y, dim=0)
    c_index_batch = t.stack(c_index_batch, dim=0)

    return context_x, context_y, target_x, target_y, c_index_batch


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


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.
        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.
        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.
        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.
        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, condition, x, time):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE):
            timestep = (time * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(time.device)[timestep]
        else:
            alpha = t.ones_like(time)

        for i in range(n_steps):
            grad = score_fn(t.cat([condition,x],dim=2), time)
            noise = t.randn_like(x)
            grad_norm = t.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = t.norm(noise.reshape(
                noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None] * grad
            x = x_mean + t.sqrt(step_size * 2)[:, None, None] * noise

        return x, x_mean


class LangevinCorrectorSNP(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, condition, x, x_gt, time, mask):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE):
            timestep = (time * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(time.device)[timestep]
        else:
            alpha = t.ones_like(time)

        for i in range(n_steps): 
            # Langevin 
            grad = score_fn(t.cat([condition,x],dim=2), time)
            noise = t.randn_like(x)
            grad_norm = t.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = t.norm(noise.reshape(
                noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None] * grad
            x = x_mean + t.sqrt(step_size * 2)[:, None, None] * noise

            # Context guidence
            target_y_mean_t, std_t = self.sde.marginal_prob(
                    x_gt, time)
            z = t.randn_like(x_gt)
            target_y_t = target_y_mean_t+std_t[:, None, None]*z
            x = x*(1-mask[..., None])+target_y_t*mask[..., None]

        return x, x_mean
    

class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, time):
        f, G = self.rsde.discretize(x, time)
        z = t.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None] * z
        return x, x_mean


    

class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, condition, y_t, time):

        dt = -1. / self.rsde.N
        z = t.randn_like(y_t)

        drift, diffusion = self.rsde.sde(condition, y_t, time,)

        y_t_mean = y_t + drift * dt
        y_t = y_t_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        # return y_t.transpose(1,2), y_t_mean.transpose(1,2)

        return y_t, y_t_mean
    
    def updata_fn_guide(self, condition, y_t, time, mask, target_y,r=1):

        y_t.requires_grad_() 
        dt = -1. / self.rsde.N
        z = t.randn_like(y_t)
            
        drift, diffusion, alpha,sigma_2,score = self.rsde.sde(condition, y_t, time,guide=True)
        y_t_mean = y_t.detach() + drift.detach() * dt
        y_t_hat = y_t_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

        y_0_hat=(y_t+sigma_2[:, None, None]*score)/alpha
        norm = t.norm((target_y * mask) - (y_0_hat * mask))
        norm_grad = t.autograd.grad(outputs=norm, inputs=y_t)[0] 
        y_t_hat=y_t_hat  - r*norm_grad

        return y_t_hat, y_t_mean




def generate_fn_guide(target_x, solver,  target_y=None, c_idx=None,):

    eps = 1e-4
    B, N, F = target_x.shape
    # Initial sample
    channel = target_y.shape[-1]
    y_t = solver.sde.prior_sampling([B, N, channel]).to(device)
    timesteps = t.linspace(solver.sde.T, eps, solver.sde.N).to(device)

    # Generate mask {0,1}^B*N
    if c_idx != None:
        mask = t.ones((B, N)).to(target_y.device)
        for j in range(mask.shape[0]):
            mask[j][c_idx[j]] = 0

    for i in range(solver.sde.N):
        print(f'SDE step:{i}', timesteps[i])
        time = timesteps[i]
        vec_t = t.ones(B, device=time.device) * time

        # rsde updata
        y_t, y_t_mean = solver.updata_fn_guide(target_x, y_t, vec_t, mask[..., None], target_y)

        # data consistent
        if c_idx != None:
            target_y_mean_t, std_t = solver.sde.marginal_prob(
                target_y, vec_t)
            z = t.randn_like(target_y)
            target_y_t = target_y_mean_t+std_t[:, None, None]*z
            y_t = y_t*(1-mask[..., None])+target_y_t*mask[..., None]
            y_t_mean = y_t*(1-mask[..., None]) + \
                target_y_mean_t*mask[..., None]

    return y_t_mean
    



if __name__ == '__main__':
    if dataset == 'mnist':

        model = ScoreNet_temb2(dim_input=6, dim_output=3,ln=False).to(device)

        ema = ExponentialMovingAverage(model.parameters(), 0.999)

        ema.load_state_dict(t.load('./checkpoint_dnp6/ema_1000.pth.tar'))
        ema.copy_to(model.parameters())

        test_dataset = torchvision.datasets.MNIST(
            './data', train=False, download=True)
        dloader = DataLoader(test_dataset, batch_size=10,
                            collate_fn=collate_fn_mnist, shuffle=False)

    elif dataset == 'celeba32':
        model = ScoreNet_temb2(dim_input=6,dim_output=3).to(device)

        ema = ExponentialMovingAverage(model.parameters(), 0.999)
        ema.load_state_dict(t.load('./checkpoint_dnp_celeba2/ema_300.pth.tar'))
        ema.copy_to(model.parameters())

        _, test_dataset = train_dev_split(
            CelebA32(), dev_size=0.1, is_stratify=False)

        dloader = DataLoader(test_dataset, batch_size=10,
                            collate_fn=collate_fn_celeba32, shuffle=True)
        
    else:
        raise NotImplementedError('Not Implementation')
        
    model.eval()
    sde = VPSDE(N=1000)
    score_func = get_score_fn(sde, model, False)

    Euler_solver = EulerMaruyamaPredictor(sde, score_func)

    
    Langevin_corr = LangevinCorrector(sde, score_func, 0.128, 3)

    fig = plt.figure()
    for i, d in enumerate(dloader):
        context_x, context_y, target_x, target_y, c_idx = d
        context_x, context_y, target_x, target_y = context_x.to(
            device), context_y.to(device), target_x.to(device), target_y.to(device)

        pred_y = generate_fn_guide(
            target_x, Euler_solver, target_y,c_idx)  # Test

        for n in range(target_x.shape[0]):
            fig.add_subplot(3, 10, n+1)
            plt.axis('off')
            h = int(math.sqrt(pred_y[n].shape[0]))
            channel = pred_y[n].shape[1]
            plt.imshow(
                pred_y[n].view(-1, h, channel).squeeze().detach().cpu().numpy())
            
            fig.add_subplot(3, 10, n+11)
            plt.axis('off')
            plt.imshow(
                target_y[n].view(-1, h, channel).squeeze().detach().cpu().numpy())

            fig.add_subplot(3, 10, n+21)
            plt.axis('off')
            target_y[n][c_idx[0]] = 0
            plt.imshow(
                target_y[n].view(-1, h, channel).squeeze().detach().cpu().numpy())

            print(pred_y[n].view(-1, h,
                channel).squeeze().detach().cpu().numpy().shape)
    
        if i == 0:

            break
    plt.savefig('./original.png', bbox_inches='tight')
