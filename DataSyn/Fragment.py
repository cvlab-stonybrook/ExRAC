import os
import torch
import datetime
import warnings
import torchaudio
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchaudio import transforms
from Utils.soft_dtw_cuda import SoftDTW
from numba import NumbaPerformanceWarning
from Utils.ExemplarDetect import max_product_cy
from Models.Feature.ExEnc import fusion_block, MultiScaleModule
from Models.Feature.bc_resnet_model import BcResNetModel, N_CLASS

# Filter out UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

class FragmentExtracter(nn.Module):
    def __init__(self,
                 cfg,
                 device: str,
                 save_path: str,
                 ):
        super(FragmentExtracter, self).__init__()
        # save path for action framents
        self.save_path = save_path
        self.cfg = cfg.ExEnc
        # hyper-para for audio sampling
        self.SAMPLE_RATE = 16000
        self.EPS = 1e-9
        self.device = device
        self.proj_dim = self.cfg.proj_dim
        self.series_size = self.cfg.series_size
        # Init S-DTW
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        self.in_conv = nn.Sequential(
            nn.Conv1d(kernel_size=5,
                      in_channels=6,
                      out_channels=6,
                      stride=1,
                      padding=2),
            nn.GELU(),
        )
        self.window_projection = nn.Sequential(
            nn.Conv1d(kernel_size=10,
                      in_channels=6,
                      out_channels=60,
                      stride=10),
        )
        self.mask_projection = nn.MaxPool1d(kernel_size=11, stride=10, padding=5)
        self.fusion = fusion_block()
        self.MultiScale = MultiScaleModule(self.proj_dim, self.series_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        self.init_audio_model()

    def init_audio_model(self):
        self.audio_model = BcResNetModel(
            n_class=N_CLASS,
            scale=2,
            dropout=0.1,
            use_subspectral=True,
        )
        if os.path.exists('../Models/Feature/best_model.pth'):
            self.audio_model.load_state_dict(torch.load('../Models/Feature/best_model.pth'))
        else:
            self.audio_model.load_state_dict(torch.load('./best_model.pth'))
        self.audio_model.to(self.device)

    def pairwise_minus_l2_distance(self, x, y):
        x = x.unsqueeze(3).detach()
        y = y.unsqueeze(2)
        l2_dist = torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + 1e-8)
        l2_dist = nn.InstanceNorm2d(l2_dist.size(1))(l2_dist)
        return -l2_dist

    def forward(self, input):
        # Get the action fragment in the inference
        result = {}
        result['qual'] = {}
        sensor = input['sensor']
        result['qual']['sensor'] = sensor[0, :, :].cpu().squeeze().numpy()
        audio = input['audio']
        idx = input['idx']
        mask = input['mask']
        # Perfrom detection on audio
        with torch.no_grad():
            audio = audio.unfold(1, self.SAMPLE_RATE, int(self.SAMPLE_RATE * 0.1))
            to_mel = transforms.MelSpectrogram(sample_rate=self.SAMPLE_RATE, n_fft=1024, f_max=16000, n_mels=40)
            log_mel = (to_mel(audio) + self.EPS).log2()
            log_mel = log_mel.to(self.device)
            B, L, H, W = log_mel.shape
            log_mel = rearrange(log_mel, 'b l h w -> (b l) h w')
            log_mel = log_mel.unsqueeze(1)
            log_mel = log_mel[:150, :, :, :]
            window_score = self.audio_model(log_mel)
            probs = torch.nn.Softmax(dim=-1)(window_score).cpu().detach().numpy()
            one_index = 21
            two_index = 29
            three_index = 27
            prob_array = []
            # Build the array for max product searching
            max_one_prob = 0
            max_two_prob = 0
            max_three_prob = 0
            for win_idx in range(probs.shape[0]):
                prob = probs[win_idx, :]
                one_prob = prob[one_index]
                two_prob = prob[two_index]
                three_prob = prob[three_index]
                max_one_prob = max(max_one_prob, one_prob)
                max_two_prob = max(max_two_prob, two_prob)
                max_three_prob = max(max_three_prob, three_prob)
                prob_array.append([one_prob, two_prob, three_prob])
            final_max_product = -1
            final_indices = (0, 0, 0)
            if log_mel.shape[0] > 100:
                for win_1_start in range(0, log_mel.shape[0] - 100, 1):
                    win_2 = prob_array[win_1_start:win_1_start + 100]
                    max_product, indices = max_product_cy(win_2)
                    if max_product > final_max_product:
                        indices[0] = indices[0] + win_1_start
                        indices[1] = indices[1] + win_1_start
                        indices[2] = indices[2] + win_1_start
                        final_max_product = max_product
                        final_indices = indices
            else:
                max_product, final_indices = max_product_cy(prob_array)
            audio_exemplar_indices = final_indices
        assert final_indices[0] < final_indices[1] < final_indices[2]
        key_list = []
        for audio_idx in audio_exemplar_indices:
            sensor_idx = audio_idx * 10
            key_list.append(sensor_idx)
        assert len(key_list) == 3



        one_prob = prob_array[final_indices[0]][0]
        two_prob = prob_array[final_indices[1]][1]
        three_prob = prob_array[final_indices[2]][2]

        save_path = self.save_path
        sensor_np = sensor.squeeze().detach().cpu().numpy()
        if one_prob > 0.75 and two_prob > 0.75 and key_list[1] - key_list[0] < 500:
            input['fragment_num'] += 1
            sensor_np_fragment = sensor_np[key_list[0]:key_list[1], :]
            assert sensor_np_fragment.shape[0] > 0
            np.save(os.path.join(save_path, str(input['fragment_num']) + '.npy'), sensor_np_fragment)

        if two_prob > 0.75 and three_prob > 0.75 and key_list[2] - key_list[1] < 500:
            input['fragment_num'] += 1
            sensor_np_fragment = sensor_np[key_list[1]:key_list[2], :]
            assert sensor_np_fragment.shape[0] > 0
            np.save(os.path.join(save_path, str(input['fragment_num']) + '.npy'), sensor_np_fragment)

        exemplar_list = []
        for key in key_list:
            for key_size in [10, 20, 40]:
                key = max(40, key)
                key_sensor = sensor[:, max(key - key_size, 0): key + key_size + 1, :]
                exemplar_list.append(key_sensor)
        assert len(exemplar_list) == 9
        result['fragment_num'] = input['fragment_num']
        return result

    def get_simi(self, query, exemplar):
        cor_stride = 1
        DTW_batch_size = 0
        kernel_size = exemplar.shape[1]
        padd_length = kernel_size - 1
        if padd_length % 2 == 0:
            left = padd_length // 2
            right = padd_length // 2
        else:
            left = padd_length // 2 + 1
            right = padd_length // 2
        pad_query = F.pad(query, (0, 0, left, right), "constant", 0)
        corr_feat = F.conv1d(pad_query.transpose(-1, -2), exemplar.transpose(-1, -2).flip(-1), stride=cor_stride)
        pad_query = pad_query.unfold(1, kernel_size, cor_stride)
        pad_query = pad_query.squeeze().permute(0, 2, 1)
        if DTW_batch_size != 0:
            DTW_it = int(np.ceil(pad_query.shape[0] / DTW_batch_size).astype(int))
            exemplar = exemplar.repeat(DTW_batch_size, 1, 1)
            dtw_feat = None
            for i in range(DTW_it):
                query_window = pad_query[i * DTW_batch_size:(i + 1) * DTW_batch_size, :, :]
                if len(query_window.shape) == 2:
                    dis = self.sdtw(query_window.unsqueeze(0), exemplar[0, :, :].unsqueeze(0))
                else:
                    dis = self.sdtw(query_window, exemplar[:query_window.shape[0], :, :])
                if dtw_feat is None:
                    dtw_feat = dis
                else:
                    dtw_feat = torch.cat((dtw_feat, dis), dim=0)
        else:
            exemplar = exemplar.repeat(pad_query.shape[0], 1, 1)
            dtw_feat = self.sdtw(pad_query, exemplar)
        act = torch.nn.ReLU()
        dtw_feat = dtw_feat.squeeze()
        dtw_feat = torch.max(dtw_feat) - dtw_feat
        dtw_feat = (dtw_feat - dtw_feat.mean(dim=0)) / dtw_feat.std(dim=0, unbiased=False)
        dtw_feat = act(dtw_feat)
        corr_feat_norm = corr_feat.squeeze()
        corr_feat_norm = (corr_feat_norm - corr_feat_norm.mean(dim=0)) / corr_feat_norm.std(dim=0, unbiased=False)
        corr_feat_norm = act(corr_feat_norm)
        simi = dtw_feat * corr_feat_norm
        simi = simi.unsqueeze(0).unsqueeze(-1)
        return simi

import tqdm
from Dataset.DWC import DWC
from Utils.config import load_cfg
# Subject
cfg = load_cfg('../Config/vanilla.yaml')
cfg.Dataset.split_type = 'extreme'
enc = FragmentExtracter(cfg, 'cuda:0', 'F:\\DWC_v1\\syn_data\\{}\\fragments'.format(cfg.Dataset.split_type))
os.makedirs('F:\\DWC_v1\\syn_data\\{}\\fragments'.format(cfg.Dataset.split_type), exist_ok=True)
enc.to('cuda:0')
dataset = DWC(cfg, 'train')
fragment_num = 0
# Save Fragments

for idx in tqdm.tqdm(range(dataset.__len__())):
    #idx = 1
    sample = dataset.__getitem__(idx)
    sample['sensor'] = sample['sensor'].unsqueeze(0)
    sample['audio'] = sample['audio'].unsqueeze(0)
    sample['sensor'] = sample['sensor'].to('cuda:0')
    sample['fragment_num'] = fragment_num
    result = enc(sample)
    fragment_num = result['fragment_num']
print('Fragment Num:', fragment_num)
