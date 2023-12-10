'''
DWC-syn dataset.
'''
import os
import json
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchaudio import transforms
from torch.utils.data import Dataset
from Register import DATASET_REGISTER_MACHINE
from Utils.density_map import GenerateDensityMap
from sklearn.preprocessing import StandardScaler

@DATASET_REGISTER_MACHINE.register()
class DWC_syn(Dataset):
    def __init__(self, cfg, proportion=None):
        self.is_local = cfg.Is_local
        self.data_cfg = cfg.DWC_syn
        self.proportion = proportion
        self.split_type = cfg.Dataset.split_type
        self.down_sample = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1)
        if self.is_local:
            self.root_path = self.data_cfg.local_data_path
        else:
            self.root_path = self.data_cfg.server_data_path
        assert self.split_type in ['action_plus']
        self.syn_data_path = os.path.join(self.root_path, self.split_type, 'data')
        if self.proportion == None:
            self.sample_num = len(os.listdir(self.syn_data_path))
        else:
            assert self.proportion > 0 and self.proportion < 1
            self.sample_num = int(len(os.listdir(self.syn_data_path)) * self.proportion)
        self.verbose()

    def __len__(self):
        return self.sample_num

    def verbose(self):
        print('DWC Syn samples: {}'.format(self.sample_num))
        print('DWC Syn data split type: ', self.split_type)

    def __getitem__(self, item):
        item += 1
        sensor_path = os.path.join(self.syn_data_path , str(item), str(item) + '_sensor.pt')
        density_path = os.path.join(self.syn_data_path , str(item), str(item) + '_density.pt')
        exemplar_idx = os.path.join(self.syn_data_path, str(item), str(item) + '_exemplar_idx.pt')
        sensor_data = torch.load(sensor_path).numpy()
        density_map = torch.load(density_path)
        exemplar_idx = torch.load(exemplar_idx)
        if self.data_cfg.normalize:
            scaler_query = StandardScaler()
            scaler_query.fit(sensor_data)
            sensor_data = scaler_query.transform(sensor_data)
            # L C
        if self.data_cfg.padding:
            sensor_data = torch.FloatTensor(sensor_data)
            if sensor_data.shape[0] > self.data_cfg.padding_length:
                sensor_data = self.down_sample(sensor_data.transpose(-1, -2).unsqueeze(0)).transpose(-1, -2).squeeze(0)
            sensor_data_pad = np.zeros((self.data_cfg.padding_length, 6))
            sensor_pad_mask = np.zeros((self.data_cfg.padding_length))
            if not sensor_data.shape[0] > self.data_cfg.padding_length:
                sensor_data_pad[:sensor_data.shape[0], :] = sensor_data
                sensor_pad_mask[:sensor_data.shape[0]] = np.ones((sensor_data.shape[0]))
            else:
                sensor_data_pad = sensor_data[:self.data_cfg.padding_length, :]
                sensor_pad_mask = np.ones((self.data_cfg.padding_length))
        else:
            sensor_data = torch.FloatTensor(sensor_data)
        # Create Density Map
        count = np.rint(density_map.sum().item())
        sample = {}
        sample['count'] = count
        sample['idx'] = item
        sample['exemplar_idx'] = exemplar_idx
        # Pad correlation map
        if self.data_cfg.padding:
            sensor_data = torch.FloatTensor(sensor_data_pad)
            sensor_mask = torch.FloatTensor(sensor_pad_mask)
            assert sensor_data.shape[0] == self.data_cfg.padding_length
            assert sensor_mask.shape[0] == self.data_cfg.padding_length
        else:
            sensor_data = torch.FloatTensor(sensor_data)
            sensor_mask = torch.ones((sensor_data.shape[0]))
        sample['sensor'] = sensor_data
        sample['density'] = density_map
        sample['mask'] = sensor_mask
        return sample
