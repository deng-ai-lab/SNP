# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import numpy as np
from scipy import integrate


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps, target_idx):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t)[:, target_idx] * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0][:, target_idx]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(sde, hutchinson_type='Rademacher',
                      rtol=1e-4, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

    def drift_fn(model, condition, x, t):
        from test_snp import get_score_fn
        """The drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(condition, x, t)[0]

    def div_fn(model, condition, x, t, noise, target_idx):
        return get_div_fn(lambda xx, tt: drift_fn(model, condition, xx, tt))(x, t, noise, target_idx)

    def likelihood_fn(model, condition, data, target_idx=None):
        """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
      condition: [b, n, c_x]
      data: [b, n, c_y]
      target_idx: numpy array, masks on dimension 'n'. It contains the target data, others as context.

        Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
        with torch.no_grad():
            assert target_idx is not None
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data[:, target_idx])
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data[:, target_idx], low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = to_flattened_numpy(drift_fn(model, condition, sample, vec_t))
                logp_grad = to_flattened_numpy(div_fn(model, condition, sample, vec_t, epsilon, target_idx))
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = from_flattened_numpy(zp[:-shape[0]], shape)[:, target_idx].to(data.device).type(torch.float32)
            delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
            prior_logp = sde.prior_logp(z)
            # ll = prior_logp + delta_logp
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[2:]) * len(target_idx)
            bpd = bpd / N
            # # A hack to convert log-likelihoods to bits/dim
            # offset = 7. - inverse_scaler(-1.)
            # bpd = bpd + offset
            return bpd, z, nfe

    return likelihood_fn
