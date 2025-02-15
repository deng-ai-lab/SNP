import logging
import torch as th

import numpy as np
import sklearn
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, WhiteKernel
from torch.utils.data import Dataset
from torch.distributions import Normal
import matplotlib.pyplot as plt
import os
import h5py
import copy


def collate_fn_gp(batch):
    # Puts each data field into a tensor with outer dimension batch size
    assert isinstance(batch[0], tuple)

    max_num_context = 128
    num_context = np.random.randint(10, 128)  # extract random number of contexts
    # num_target = np.random.randint(0, max_num_context - num_context)
    num_target = max_num_context - num_context
    num_total_points = num_context + num_target  # this num should be # of target points
    # num_total_points = max_num_context
    context_x, context_y, target_x, target_y = list(), list(), list(), list()

    for x, y in batch:
        total_x = x.float()
        total_y = y.float()

        context_idx = np.random.choice(range(128), num_context, replace=False)

        c_x = total_x[context_idx]
        c_y = total_y[context_idx]

        context_x.append(c_x)
        context_y.append(c_y)
        target_x.append(total_x)
        target_y.append(total_y)

    context_x = th.stack(context_x, dim=0)
    context_y = th.stack(context_y, dim=0)
    target_x = th.stack(target_x, dim=0)
    target_y = th.stack(target_y, dim=0)

    return context_x, context_y, target_x, target_y


def draw_gp(target_x, target_y, seen_idx, mu, sigma, dataset_name, mode, sample_times=5, path='results', samples=None
            , draw_mode='No samples'):
    seen_idx_list = seen_idx if isinstance(seen_idx, list) else None
    kernel_dict = {'Matern_Kernel': Matern(length_scale=0.2, nu=1.5),
                   'Periodic_Kernel': ExpSineSquared(length_scale=1, periodicity=0.5),
                   'RBF_Kernel': RBF(length_scale=0.2)}
    generator = GaussianProcessRegressor(kernel=kernel_dict[dataset_name], alpha=0.005,)

    if draw_mode == 'samples':
        if samples is None:
            new_samples = []
            normal_dist = Normal(loc=mu, scale=sigma)
            for i in range(sample_times):
                new_samples.append(normal_dist.sample().cpu().numpy())
            samples = new_samples
        else:
            samples = samples.cpu().numpy()
    # print(seen_idx[0])
    # print(target_x[:, 20, 0])
    # print(target_y[:, 20, 0])
    mu, sigma = mu.cpu().numpy(), sigma.cpu().numpy()
    target_x, target_y = target_x.cpu().numpy(), target_y.cpu().numpy()
    batchsize = target_x.shape[0]

    for batch in range(batchsize):
        if seen_idx_list is not None:
            seen_idx = seen_idx_list[batch]
        x = target_x[batch]
        y = target_y[batch]
        generator.fit(x[seen_idx], y[seen_idx])

        gt_mu, gt_sigma = generator.predict(x, return_std=True)

        plt.figure(figsize=(12, 2))
        plt.xticks(np.linspace(-1, 1, 6))
        plt.yticks(np.linspace(-6, 6, 3))
        # plt.gca().set_aspect(3)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

        # 绘制观测点（黑色点）
        plt.scatter(x[seen_idx], y[seen_idx], color='black', marker='o', label='Observed points')

        # 绘制真值曲线,紫色虚线
        plt.plot(x, y, linestyle='solid', color='red', label='True values', alpha=0.6)
        plt.plot(x, gt_mu, linestyle='dashdot', color='purple', label='gt mean')
        plt.plot(x, gt_mu - 2 * gt_sigma, linestyle='dashdot', color='purple', label='gt lower bound')
        plt.plot(x, gt_mu + 2 * gt_sigma, linestyle='dashdot', color='purple', label='gt upper bound')

        # 绘制预测值
        plt.plot(x, mu[batch], linestyle='solid', color='blue', alpha=0.8)
        plt.fill_between(x.flatten(), (mu[batch] - 2 * sigma[batch]).flatten(), (mu[batch] + 2 * sigma[batch]).flatten(),
                         color='lightblue', alpha=0.4)

        if draw_mode == 'samples':
            for i in range(sample_times):
                plt.plot(x, samples[i][batch], linestyle='solid', color='blue', alpha=0.4)

        # plt.legend()
        plt.ylabel(dataset_name)
        save_path = os.path.join(path, dataset_name, 'b{}_{}_c{}.png'.format(batch, mode, len(seen_idx)))
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


class NotLoadedError(Exception):
    pass


class DatasetMerger(Dataset):
    """
    Helper which merges an iterable of datasets. Assume that they all have the same attributes
    (redirect to the first one).
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.cumul_len = np.cumsum([len(d) for d in self.datasets])

    def __getitem__(self, index):
        idx_dataset = self.cumul_len.searchsorted(index + 1)  # + 1 because of 0 index
        idx_in_dataset = index
        if idx_dataset > 0:
            idx_in_dataset -= self.cumul_len[idx_dataset - 1]  # - 1 because rm previous
        return self.datasets[idx_dataset][idx_in_dataset]

    def __len__(self):
        return self.cumul_len[-1]

    def __getattr__(self, attr):
        return getattr(self.datasets[0], attr)


def rescale_range(X, old_range, new_range):
    """Rescale X linearly to be in `new_range` rather than `old_range`."""
    old_min = old_range[0]
    new_min = new_range[0]
    old_delta = old_range[1] - old_min
    new_delta = new_range[1] - new_min
    return (((X - old_min) * new_delta) / old_delta) + new_min


def _parse_save_file_chunk(save_file, idx_chunk):
    if save_file is None:
        save_file, save_group = None, None
    elif isinstance(save_file, tuple):
        save_file, save_group = save_file[0], save_file[1] + "/"
    elif isinstance(save_file, str):
        save_file, save_group = save_file, ""
    else:
        raise ValueError("Unsupported type of save_file={}.".format(save_file))

    if idx_chunk is not None:
        chunk_suffix = "_chunk_{}".format(idx_chunk)
    else:
        chunk_suffix = ""

    return save_file, save_group, chunk_suffix


def load_chunk(keys, save_file, idx_chunk):
    items = dict()
    save_file, save_group, chunk_suffix = _parse_save_file_chunk(save_file, idx_chunk)

    if save_file is None or not os.path.exists(save_file):
        raise NotLoadedError()

    try:
        with h5py.File(save_file, "r") as hf:
            for k in keys:
                items[k] = torch.from_numpy(
                    hf["{}{}{}".format(save_group, k, chunk_suffix)][:]
                )
    except KeyError:
        raise NotLoadedError()

    return items


def save_chunk(to_save, save_file, idx_chunk, logger=None):
    save_file, save_group, chunk_suffix = _parse_save_file_chunk(save_file, idx_chunk)

    if save_file is None:
        return  # don't save

    if logger is not None:
        logger.info(
            "Saving group {} chunk {} for future use ...".format(save_group, idx_chunk)
        )

    with h5py.File(save_file, "a") as hf:
        for k, v in to_save.items():
            hf.create_dataset(
                "{}{}{}".format(save_group, k, chunk_suffix), data=v.numpy()
            )


class GPDataset(Dataset):
    """
    Dataset of functions generated by a gaussian process.
    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels or list
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default.
    min_max : tuple of floats, optional
        Min and max point at which to evaluate the function (bounds).
    n_samples : int, optional
        Number of sampled functions contained in dataset.
    n_points : int, optional
        Number of points at which to evaluate f(x) for x in min_max.
    is_vary_kernel_hyp : bool, optional
        Whether to sample each example from a kernel with random hyperparameters,
        that are sampled uniformly in the kernel hyperparameters `*_bounds`.
    save_file : string or tuple of strings, optional
        Where to save and load the dataset. If tuple `(file, group)`, save in
        the hdf5 under the given group. If `None` regenerate samples indefinitely.
        Note that if the saved dataset has been completely used,
        it will generate a new sub-dataset for every epoch and save it for future
        use.
    n_same_samples : int, optional
        Number of samples with same kernel hyperparameters and X. This makes the
        sampling quicker.
    is_reuse_across_epochs : bool, optional
        Whether to reuse the same samples across epochs.  This makes the
        sampling quicker and storing less memory heavy if `save_file` is given.
    kwargs:
        Additional arguments to `GaussianProcessRegressor`.
    """

    def __init__(
            self,
            kernel=(
                    WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 0.5))
                    + RBF(length_scale=0.4, length_scale_bounds=(0.1, 1.0))
            ),
            min_max=(-2, 2),
            n_samples=1000,
            n_points=128,
            is_vary_kernel_hyp=False,
            save_file=None,
            logging_level=logging.INFO,
            n_same_samples=20,
            is_reuse_across_epochs=True,
            **kwargs,
    ):

        self.n_samples = n_samples
        self.n_points = n_points
        self.min_max = min_max
        self.is_vary_kernel_hyp = is_vary_kernel_hyp
        self.logger = logging.getLogger("GPDataset")
        self.logger.setLevel(logging_level)
        self.save_file = save_file
        self.n_same_samples = n_same_samples
        self.is_reuse_across_epochs = is_reuse_across_epochs

        self._idx_precompute = 0  # current index of precomputed data
        self._idx_chunk = 0  # current chunk (i.e. epoch)

        if not is_vary_kernel_hyp:
            # only fit hyperparam when predicting if using various hyperparam
            kwargs["optimizer"] = None

            # we also fix the bounds as these will not be needed
            for hyperparam in kernel.hyperparameters:
                kernel.set_params(**{f"{hyperparam.name}_bounds": "fixed"})

        self.generator = GaussianProcessRegressor(
            kernel=kernel, alpha=0.005, **kwargs  # numerical stability for preds
        )

        self.precompute_chunk_()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.is_reuse_across_epochs:
            return self.data[index], self.targets[index]

        else:
            # doesn't use index because randomly generated in any case => sample
            # in order which enables to know when epoch is finished and regenerate
            # new functions
            self._idx_precompute += 1
            if self._idx_precompute == self.n_samples:
                self.precompute_chunk_()
            return self.data[self._idx_precompute], self.targets[self._idx_precompute]

    def get_samples(
            self,
            n_samples=None,
            test_min_max=None,
            n_points=None,
            save_file=None,
            idx_chunk=None,
    ):
        """Return a batch of samples
        Parameters
        ----------
        n_samples : int, optional
            Number of sampled function (i.e. batch size). Has to be dividable
            by n_diff_kernel_hyp or 1. If `None` uses `self.n_samples`.
        test_min_max : float, optional
            Testing range. If `None` uses training one.
        n_points : int, optional
            Number of points at which to evaluate f(x) for x in min_max. If None
            uses `self.n_points`.
        save_file : string or tuple of strings, optional
            Where to save and load the dataset. If tuple `(file, group)`, save in
            the hdf5 under the given group. If `None` uses does not save.
        idx_chunk : int, optional
            Index of the current chunk. This is used when `save_file` is not None,
            and you want to save a single dataset through multiple calls to
            `get_samples`.
        """
        test_min_max = test_min_max if test_min_max is not None else self.min_max
        n_points = n_points if n_points is not None else self.n_points
        n_samples = n_samples if n_samples is not None else self.n_samples

        try:
            loaded = load_chunk({"data", "targets"}, save_file, idx_chunk)
            data, targets = loaded["data"], loaded["targets"]
        except NotLoadedError:
            X = self._sample_features(test_min_max, n_points, n_samples)
            X, targets = self._sample_targets(X, n_samples)
            data = self._postprocessing_features(X, n_samples)
            save_chunk(
                {"data": data, "targets": targets},
                save_file,
                idx_chunk,
                logger=self.logger,
            )

        return data, targets

    def set_samples_(self, data, targets):
        """Use the samples (output from `get_samples`) as the data."""
        self.is_reuse_across_epochs = True
        self.data = data
        self.targets = targets
        self.n_samples = self.data.size(0)

    def precompute_chunk_(self):
        """Load or precompute and save a chunk (data for an epoch.)"""
        self._idx_precompute = 0
        self.data, self.targets = self.get_samples(
            save_file=self.save_file, idx_chunk=self._idx_chunk
        )
        self._idx_chunk += 1

    def _sample_features(self, min_max, n_points, n_samples):
        """Sample X with non uniform intervals. """
        X = np.random.uniform(min_max[1], min_max[0], size=(n_samples, n_points))
        # sort which is convenient for plotting
        X.sort(axis=-1)
        return X

    def _postprocessing_features(self, X, n_samples):
        """Convert the features to a tensor, rescale them to [-1,1] and expand."""
        X = torch.from_numpy(X).unsqueeze(-1).float()
        X = rescale_range(X, self.min_max, (-1, 1))
        return X

    def _sample_targets(self, X, n_samples):
        targets = X.copy()
        n_samples, n_points = X.shape
        for i in range(0, n_samples, self.n_same_samples):
            if self.is_vary_kernel_hyp:
                self.sample_kernel_()

            for attempt in range(self.n_same_samples):
                # can have numerical issues => retry using a different X
                try:
                    # takes care of boundaries
                    n_same_samples = targets[i: i + self.n_same_samples, :].shape[0]
                    targets[i: i + self.n_same_samples, :] = self.generator.sample_y(
                        X[i + attempt, :, np.newaxis],
                        n_samples=n_same_samples,
                        random_state=None,
                    ).transpose(1, 0)
                    X[i: i + self.n_same_samples, :] = X[i + attempt, :]
                except np.linalg.LinAlgError:
                    continue  # try again
                else:
                    break  # success
            else:
                raise np.linalg.LinAlgError("SVD did not converge 10 times in a row.")

        # shuffle output to not have n_same_samples consecutive
        X, targets = sklearn.utils.shuffle(X, targets)
        targets = torch.from_numpy(targets)
        targets = targets.view(n_samples, n_points, 1).float()
        return X, targets

    def sample_kernel_(self):
        """
        Modify inplace the kernel hyperparameters through uniform sampling in their
        respective bounds.
        """
        K = self.generator.kernel
        for hyperparam in K.hyperparameters:
            K.set_params(
                **{hyperparam.name: np.random.uniform(*hyperparam.bounds.squeeze())}
            )


def sample_gp_dataset_like(dataset, **kwargs):
    """Wrap the output of `get_samples` in a gp dataset."""
    new_dataset = copy.deepcopy(dataset)
    new_dataset.set_samples_(*dataset.get_samples(**kwargs))
    return new_dataset


def get_gp_datasets(
        kernels, data_root, **kwargs
):
    """
    Return a train, test and validation set for all the given kernels (dict).
    """
    save_file = f"{os.path.join(data_root, 'GP_DATA', 'gp_dataset.hdf5')}"
    datasets = dict()

    def get_save_file(name, save_file=save_file):
        if save_file is not None:
            save_file = (save_file, name)
        return save_file

    for name, kernel in kernels.items():
        datasets[name] = GPDataset(
            kernel=kernel, save_file=get_save_file(name), **kwargs
        )

    datasets_test = {
        k: sample_gp_dataset_like(
            dataset, save_file=get_save_file(k), idx_chunk=-1, n_samples=10000
        )
        for k, dataset in datasets.items()
    }

    datasets_valid = {
        k: sample_gp_dataset_like(
            dataset,
            save_file=get_save_file(k),
            idx_chunk=-2,
            n_samples=dataset.n_samples // 10,
        )
        for k, dataset in datasets.items()
    }

    return datasets, datasets_test, datasets_valid


def get_datasets_single_gp(data_root='../data', n_samples=50000):
    """Return train / tets / valid sets for 'Samples from a single GP'."""
    kernels = dict()

    kernels["RBF_Kernel"] = RBF(length_scale=(0.2))

    kernels["Periodic_Kernel"] = ExpSineSquared(length_scale=1, periodicity=0.5)

    kernels["Matern_Kernel"] = Matern(length_scale=0.2, nu=1.5)

    # kernels["Noisy_Matern_Kernel"] = WhiteKernel(noise_level=0.1) + Matern(
    #     length_scale=0.2, nu=1.5
    # )

    return get_gp_datasets(
        kernels,
        data_root=data_root,
        is_vary_kernel_hyp=False,  # use a single hyperparameter per kernel
        n_samples=n_samples,  # number of different context-target sets
        n_points=128,  # size of target U context set for each sample
        is_reuse_across_epochs=False,  # never see the same example twice
    )


def get_datasets_varying_kernel_gp(data_root='../data', n_samples=50000):
    """Return train / tets / valid sets for 'Samples from GPs with varying Kernels'."""

    datasets, test_datasets, valid_datasets = get_datasets_single_gp(data_root, n_samples)
    return (
        dict(All_Kernels=DatasetMerger(datasets.values())),
        dict(All_Kernels=DatasetMerger(test_datasets.values())),
        dict(All_Kernels=DatasetMerger(valid_datasets.values())),
    )


def get_datasets_variable_hyp_gp(data_root='../data', n_samples=50000):
    """Return train / tets / valid sets for 'Samples from GPs with varying Kernel hyperparameters'."""
    kernels = dict()

    kernels["Variable_Matern_Kernel"] = Matern(length_scale_bounds=(0.01, 0.3), nu=1.5)

    return get_gp_datasets(
        kernels,
        data_root=data_root,
        is_vary_kernel_hyp=True,  # use a single hyperparameter per kernel
        n_samples=n_samples,  # number of different context-target sets
        n_points=128,  # size of target U context set for each sample
        is_reuse_across_epochs=False,  # never see the same example twice
    )


if __name__ == '__main__':
    (datasets, _, __,) = get_datasets_single_gp()

    for i, (k, dataset) in enumerate(datasets.items()):
        print(k)
        print(dataset.n_samples)
        print(len(dataset.data))
