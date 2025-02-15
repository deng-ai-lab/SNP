from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os

from torch.utils.data import DataLoader
import gzip
import torchvision
from functools import partial


def collate_fn_eeg(batch, task_type):

    # Puts each data field into a tensor with outer dimension batch size

    assert task_type in ('interpolation', 'forcasting')
    batch_size = len(batch)
    num_total_points = 256
    # extract random number of contexts
    num_context = np.random.randint(10, 256)
    max_num_context = num_total_points  # this num should be of target points

    context_x, context_y, target_x, target_y = list(), list(), list(), list()

    for d in batch:
        if task_type == 'interpolation':
            total_idx = np.arange(num_total_points)
            np.random.shuffle(total_idx)
        elif task_type == 'forcasting':
            total_idx = np.arange(num_total_points)

        c_idx = total_idx[:num_context]
        c_x, c_y, total_x, total_y = list(), list(), list(), list()
        for idx in c_idx:
            c_y.append(d[idx])
            c_x.append([idx/256.])
        for idx in total_idx:
            total_y.append(d[idx])
            total_x.append([idx/256.])
        c_x, total_x = list(
            map(lambda x: torch.FloatTensor(x), (c_x,  total_x)))
        c_y, total_y = list(map(lambda x: torch.stack(x), (c_y, total_y)))
        context_x.append(c_x)
        context_y.append(c_y)
        target_x.append(total_x)
        target_y.append(total_y)

    context_x = torch.stack(context_x, dim=0)
    context_y = torch.stack(context_y, dim=0)
    target_x = torch.stack(target_x, dim=0)
    target_y = torch.stack(target_y, dim=0)

    return context_x, context_y, target_x, target_y


collate_fn_eeg_interpolation = partial(
    collate_fn_eeg, task_type='interpolation')
collate_fn_eeg_forcasting = partial(collate_fn_eeg, task_type='forcasting')


class EEGDataset(Dataset):
    num_trial_list = [88,  93,  74, 118, 118, 106, 115,  85, 100, 112, 108,  87, 104,
                      114, 100,  70,  72,  67,  87, 104,  96, 111,  98,  77,  96,  78,
                      92,  85, 104,  57,  79,  66, 119, 117,  86,  68,  58,  80,
                      71, 101,  98, 116,  71, 111,  98, 102, 107,  59,  60, 106,  78,
                      100, 102, 109, 105,  88,  98,  66,  99,  59,  41,  99, 104,  81,
                      83,  99, 115,  93, 112, 112,  89,  73,  92,  61,  92,  67,  99,
                      88,  69,  83,  70,  89,  79,  59, 108, 104,  81,  74,  79, 109,
                      111, 101, 102,  92,  83,  80, 100,  93,  97,  30,  78, 110, 109,
                      96,  72,  61,  93, 106, 110,  94,  90, 115,  79,  85,  99, 101,
                      117, 100,  74,  77, 102]
    electrodes_name = ["FZ", "F1", "F2", "F3", "F4", "F5", "F6"]
    eeg_mean = -1.1425
    eeg_std = 9.1250

    def __init__(self,
                 root='data/egg_full',
                 test_len=20,
                 is_train=True,
                 norm=True
                 ) -> None:
        super().__init__()
        self.root = root
        self.test_len = test_len
        self.is_train = is_train
        subjects_all = os.listdir(root)
        self.norm = norm
        if self.is_train == True:
            self.subjects = subjects_all[:-test_len]
            self.num_trial_list=self.num_trial_list[:-test_len]
        else:
            self.subjects = subjects_all[-test_len:]
            self.num_trial_list=self.num_trial_list[-test_len:]

    def __getitem__(self, index):

        # Random select a subject
        #subject_idx = np.random.randint(0, len(self.subjects))

        subject_idx, trial_idx = self.get_file_by_number(
            self.num_trial_list, index)
        subject_folder = os.path.join(self.root, self.subjects[subject_idx])

        # Random select a trial
        #trial_idx = np.random.randint(0, len(subject_folder))

        trial_file = os.listdir(subject_folder)[trial_idx]

        # Read as dataframe
        with gzip.open(os.path.join(subject_folder, trial_file), 'rb') as f:
            data = f.read()
        data_str = data.decode('utf-8')
        data_lines = data_str.split('\n')[5:]
        data_csv = pd.DataFrame([x.split()
                                for x in data_lines if len(x.split()) > 0])

        # [256*7]
        data_electrodes = [data_csv[data_csv[1] == i].iloc[1:,
                                                           3].to_numpy() for i in self.electrodes_name]
        data_electrodes = np.stack(data_electrodes).transpose(
            1, 0).astype(np.float64)

        if self.norm == True:
            data_electrodes = self._norm(data_electrodes)

        return torch.FloatTensor(data_electrodes)

    def __len__(self):

        if self.is_train == True:
            return int(sum(self.num_trial_list[:-self.test_len]))
        else:
            return int(sum(self.num_trial_list[-self.test_len:]))

    
    def get_file_by_number(self,file_counts, file_number):
        count = 0
        for i, file_count in enumerate(file_counts):
            if file_number <= count + file_count:
                folder_index = i
                file_index = file_number - count - 1
                break
            count += file_count
        else:
            raise ValueError("File number out of range")
    
        return folder_index,file_index

    def _norm(self, value):

        return (value-self.eeg_mean)/self.eeg_std


if __name__ == '__main__':
    datset = EEGDataset()

    data = DataLoader(datset, batch_size=128,
                      collate_fn=collate_fn_eeg_interpolation)

    data = iter(data)
    print(next(data))
