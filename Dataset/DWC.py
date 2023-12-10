'''
DWC dataset.
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
class DWC(Dataset):
    def __init__(self, cfg, data_split: str, audio_anno = False):
        self.audio_anno = audio_anno
        # If run on local machine
        self.is_local = cfg.Is_local
        # config fo data set
        self.data_cfg = cfg.DWC
        self.split_type = cfg.Dataset.split_type
        # For AAAI removed the random split
        assert self.split_type in ['subject', 'action', 'action_plus', 'extreme']
        # For baseline or our method, baseline no audio
        self.is_baseline = cfg.Is_baseline
        self.audio_sampling_rate = self.data_cfg.audio_sampling_rate
        self.down_sample = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1)
        if self.is_local:
            self.root_path = self.data_cfg.local_data_path
        else:
            self.root_path = self.data_cfg.server_data_path

        if self.split_type == 'subject':
            self.split_path = os.path.join(self.root_path, 'subject_split_v1.json')
        elif self.split_type == 'action':
            self.split_path = os.path.join(self.root_path, 'action_split_v1.json')
        elif self.split_type == 'action_plus':
            self.split_path = os.path.join(self.root_path, 'action_plus_split_v1.json')
        else:
            self.split_path = os.path.join(self.root_path, 'extreme_samples_split_v1.json')
        self.annotation_path = os.path.join(self.root_path, 'metadata_v1.json')
        self.audio_anno_path = os.path.join(self.root_path, 'audio_index_dict.json')
        with open(self.split_path, "r") as infile:
            self.split = json.load(infile)
        with open(self.audio_anno_path, "r") as infile:
            self.audio_anno = json.load(infile)
        with open(self.annotation_path, "r") as infile:
            self.annotation = json.load(infile)
        if self.split_type != 'extreme':
            assert data_split in ['train', 'val', 'test']
        else:
            assert data_split in ['train', 'val']
        self.data_split = data_split
        self.data_list = self.split[data_split]
        self.verbose()

    def verbose(self):
        print('DWC samples {} : {}'.format(self.data_split, len(self.data_list)))
        print('DWC data split type: ', self.split_type)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        idx = self.data_list[item]
        data_sample = self.annotation[str(idx)]
        action_category = data_sample['action_category']
        subject_name = str(data_sample['subject_id'])
        count = data_sample['count']
        path_index = data_sample['path_index']
        count = int(count)
        int_path_index = int(path_index)
        str_path_index = str(path_index)
        audio_anno = False
        if int_path_index in self.audio_anno.keys():
            audio_anno = self.audio_anno[int_path_index]
        if str_path_index in self.audio_anno.keys():
            audio_anno = self.audio_anno[str_path_index]
        sensor_path = os.path.join(self.root_path, subject_name, str(path_index), 'sensor.csv')
        audio_path = os.path.join(self.root_path, subject_name, str(path_index), 'interpolated_audio.wav')
        # Read sensor data
        sensor_data = pd.read_csv(sensor_path, header=0)
        sensor_data = sensor_data.drop('timestamp', axis=1)
        sensor_data = sensor_data.to_numpy()
        num_nan = np.count_nonzero(np.isnan(sensor_data))
        assert num_nan == 0
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
            sensor_data_pad[:sensor_data.shape[0], :] = sensor_data
            sensor_pad_mask[:sensor_data.shape[0]] = np.ones((sensor_data.shape[0]))
        else:
            sensor_data = torch.FloatTensor(sensor_data)
        # Create Density Map
        density_map = GenerateDensityMap(sensor_data.numpy(), count)
        density_map = torch.FloatTensor(density_map)
        density_map = density_map.unsqueeze(0).unsqueeze(0)
        density_map = F.interpolate(density_map, size=(self.data_cfg.density_map_length), mode='linear')
        density_map = density_map.squeeze()
        density_map = density_map * count / (density_map.sum()).item()

        density_sum = np.rint(density_map.sum().item())
        assert density_sum == count
        # Read audio data
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.audio_sampling_rate:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=self.audio_sampling_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()
        sample = {}
        sample['count'] = count
        if not self.is_baseline:
            sample['audio'] = waveform
        sample['idx'] = int(path_index)
        sample['save_path'] = os.path.join(self.root_path, subject_name, str(path_index))
        # Pad correlation map, corre feat is not be used
        corr_feat = torch.zeros((self.data_cfg.padding_length, 12))
        if self.data_cfg.padding:
            sensor_data = torch.FloatTensor(sensor_data_pad)
            sensor_mask = torch.FloatTensor(sensor_pad_mask)
            pad_length = self.data_cfg.padding_length - corr_feat.shape[0]
            corr_feat = F.pad(corr_feat, (0, 0, 0, pad_length))
            assert corr_feat.shape[0] == self.data_cfg.padding_length
            assert sensor_data.shape[0] == self.data_cfg.padding_length
            assert sensor_mask.shape[0] == self.data_cfg.padding_length
        else:
            sensor_data = torch.FloatTensor(sensor_data)
            sensor_mask = torch.ones((sensor_data.shape[0]))
        sample['sensor'] = sensor_data
        sample['density'] = density_map
        sample['mask'] = sensor_mask
        sample['action_category'] = action_category
        if self.audio_anno:
            sample['audio_anno'] = audio_anno
        else:
            sample['audio_anno'] = False
        '''
        count: count label
        audio: audio waveform
        idx: sample index
        sensor: sensor data
        density: not be used
        mask: sensor mask for baselines
        action_category: action category
        '''
        return sample
