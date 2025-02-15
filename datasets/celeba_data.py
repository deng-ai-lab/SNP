import torch
import os
import abc
from torch.distributed import batch_isend_irecv
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import logging
import matplotlib.pyplot as plt
import glob
import subprocess
import hashlib
import zipfile

from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class _DatasetSubset(Dataset):
    """Helper to split train dataset into train and dev dataset.
    Parameters
    ----------
    to_split: Dataset
        Dataset to subset.
    idx_mapping: array-like
        Indices of the subset.
    Notes
    -----
    - Modified from: https: // gist.github.com / Fuchai / 12f2321e6c8fa53058f5eb23aeddb6ab
    - Cannot modify the length and targets with indexing anymore! I.e.
    `d.targets[1]=-1` doesn't work because np.array doesn't allow `arr[i][j]=-1`
    but you can do `d.targets=targets`
    """

    def __init__(self, to_split, idx_mapping):
        self.idx_mapping = idx_mapping
        self.length = len(idx_mapping)
        self.to_split = to_split

    def __getitem__(self, index):
        return self.to_split[self.idx_mapping[index]]

    def __len__(self):
        return self.length

    @property
    def targets(self):
        return self.to_split.targets[self.idx_mapping]

    @targets.setter
    def targets(self, values):
        self.to_split.targets[self.idx_mapping] = values

    @property
    def data(self):
        return self.to_split.data[self.idx_mapping]

    def __getattr__(self, attr):
        return getattr(self.to_split, attr)


def train_dev_split(to_split, dev_size=0.1, seed=123, is_stratify=True):
    """Split a training dataset into a training and validation one.
    Parameters
    ----------
    dev_size: float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the dev split. If int, represents the absolute
        number of dev samples.
    seed: int
        Random seed.
    is_stratify: bool
        Whether to stratify splits based on class label.
    """
    n_all = len(to_split)
    idcs_all = list(range(n_all))
    stratify = to_split.targets if is_stratify else None
    idcs_train, indcs_val = train_test_split(
        idcs_all, stratify=stratify, test_size=dev_size, random_state=seed
    )
    train = _DatasetSubset(to_split, idcs_train)
    valid = _DatasetSubset(to_split, indcs_val)

    return train, valid


def preprocess(root, size=(64, 64), img_format="JPEG", center_crop=None):
    """Preprocess a folder of images.
    Parameters
    ----------
    root : string
        Root directory of all images.
    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.
    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.
    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, "*" + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)



class ExternalDataset(Dataset, abc.ABC):
    """Base Class for external datasets.
    Parameters
    ----------
    root : string
        Root directory of dataset.
    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.dir = os.path.join(root, self.name)
        self.train_data = os.path.join(self.dir, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(self.dir):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass

class CelebA64(ExternalDataset):
    """CelebA Dataset from [1].
    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.
    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    Parameters
    ----------
    root : string
        Root directory of dataset.
    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).
    """
    DIR_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/"))
    urls = {
        "train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
    }
    files = {"train": "img_align_celeba"}
    shape = (3, 64, 64)
    n_classes = 0  # not classification
    name = "celeba64"

    def __init__(self, root=DIR_DATA, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + "/*")

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.dir, "celeba.zip")
        os.makedirs(self.dir)

        try:
            subprocess.check_call(
                ["curl", "-L", type(self).urls["train"], "--output", save_path]
            )
        except FileNotFoundError as e:
            raise Exception(e + " Please instal curl with `apt-get install curl`...")

        hash_code = "00d2c5bc6d35e252742224ab0c1e8fcb"
        assert (
            hashlib.md5(open(save_path, "rb").read()).hexdigest() == hash_code
        ), "{} file is corrupted.  Remove the file and try again.".format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.dir)

        os.remove(save_path)

        self.preprocess()

    def preprocess(self):
        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).shape[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.
        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = plt.imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0
    


class CelebA32(CelebA64):
    shape = (3, 32, 32)
    name = "celeba32"


class CelebA128(CelebA64):
    shape = (3, 128, 128)
    name = "celeba128"


class CelebA(CelebA64):
    shape = (3, 218, 178)
    name = "celeba"

    # use the default ones
    def preprocess(self):
        pass

if __name__ == '__main__':
    dataset= CelebA64()