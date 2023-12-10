import os
import torch
import random
import torchaudio
import numpy as np
import torch.nn as nn
from einops import rearrange
from torchaudio import transforms
from Register import FEATURE_REGISTER_MACHINE
from Models.Feature.bc_resnet_model import BcResNetModel, N_CLASS
import warnings
import torch.nn.functional as F
from Utils.soft_dtw_cuda import SoftDTW
from numba import NumbaPerformanceWarning
from Models.Feature.ExemplarDetect import max_product_cy

# Filter out UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

@FEATURE_REGISTER_MACHINE.register()
class ExEnc(nn.Module):
    def __init__(self,
                 cfg,
                 device: str,
                 ):
        super(ExEnc, self).__init__()
        self.cfg = cfg.ExEnc
        # hyper-para for audio sampling
        self.SAMPLE_RATE = 16000
        self.EPS = 1e-9
        self.device = device
        self.proj_dim = self.cfg.proj_dim
        self.series_size = self.cfg.series_size
        # Init S-DTW
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        self.window_projection = nn.Sequential(
            nn.Conv1d(kernel_size=10,
                      in_channels=6,
                      out_channels=64,
                      stride=10),
        )
        self.raw_sensor = nn.Sequential(
            nn.Conv1d(64, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(96, 128, 3, padding=1),
            nn.GELU(),
        )
        self.down_sample = nn.MaxPool1d(kernel_size=11, stride=10, padding=5)
        self.fusion = fusion_block()
        self.MultiScale = MultiScaleModule(self.proj_dim, self.series_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        # After init para, init audio model
        self.init_audio_model()
        # Save exempalr detection result in cache, since the audio exemplar detection model is fixed
        self.memory = {}
        self.memory['Audio_exemplar_idx'] = {}


    def init_audio_model(self):
        self.audio_model = BcResNetModel(
            n_class=N_CLASS,
            scale=2,
            dropout=0.1,
            use_subspectral=True,
        )
        assert os.path.exists('./Checkpoints/AudioPretrain.pth'), 'Audio pretrain model not found!'
        self.audio_model.load_state_dict(torch.load('./Checkpoints/AudioPretrain.pth'))
        self.audio_model.to(self.device)

    def pairwise_minus_l2_distance(self, x, y):
        x = x.unsqueeze(3).detach()
        y = y.unsqueeze(2)
        l2_dist = torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + 1e-8)
        l2_dist = nn.InstanceNorm2d(l2_dist.size(1))(l2_dist)
        return -l2_dist

    def forward(self, input):
        result = {}
        result['qual'] = {}
        sensor = input['sensor']
        result['qual']['sensor'] = sensor[0, :, :].cpu().squeeze().numpy()
        idx = input['idx']
        mask = input['mask']
        mask = mask.to(self.device)  # B L
        mask = mask.unsqueeze(-1)  # B L C(C = 1)

        '''
        Since the audio part is fixed, introducing cache mechanism to speed up the training process.
        '''
        if 'audio' in input.keys():
            if input['idx'].item() not in self.memory['Audio_exemplar_idx'].keys():
                audio = input['audio']
                with torch.no_grad():
                    audio = audio.unfold(1, self.SAMPLE_RATE, int(self.SAMPLE_RATE * 0.1))
                    to_mel = transforms.MelSpectrogram(sample_rate=self.SAMPLE_RATE, n_fft=1024, f_max=16000, n_mels=40)
                    log_mel = (to_mel(audio) + self.EPS).log2()
                    log_mel = log_mel.to(self.device)
                    B, L, H, W = log_mel.shape
                    log_mel = rearrange(log_mel, 'b l h w -> (b l) h w')
                    log_mel = log_mel.unsqueeze(1)
                    log_mel = log_mel[:200, :, :, :]
                    window_score = self.audio_model(log_mel)
                    probs = torch.nn.Softmax(dim=-1)(window_score).cpu().detach().numpy()
                    one_index = 21
                    two_index = 29
                    three_index = 27
                    prob_array = []
                    # Build the array for max product searching
                    for win_idx in range(probs.shape[0]):
                        prob = probs[win_idx, :]
                        one_prob = prob[one_index]
                        two_prob = prob[two_index]
                        three_prob = prob[three_index]
                        prob_array.append([one_prob, two_prob, three_prob])
                    # Perform max product searching
                    final_max_product = -1
                    final_indices = (0, 0, 0)
                    if log_mel.shape[0] > 90:
                        for win_1_start in range(0, log_mel.shape[0] - 90, 1):
                            win_2 = prob_array[win_1_start:win_1_start + 90]
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
                one_prob = prob_array[final_indices[0]][0]
                two_prob = prob_array[final_indices[1]][1]
                three_prob = prob_array[final_indices[2]][2]
                if one_prob < two_prob and one_prob < three_prob:
                    audio_exemplar_indices = [final_indices[1], final_indices[2]]
                elif two_prob < one_prob and two_prob < three_prob:
                    audio_exemplar_indices = [final_indices[0], final_indices[2]]
                elif three_prob < one_prob and three_prob < two_prob:
                    audio_exemplar_indices = [final_indices[0], final_indices[1]]
                self.memory['Audio_exemplar_idx'][input['idx'].item()] = audio_exemplar_indices
                key_list = []
                for audio_idx in audio_exemplar_indices:
                    sensor_idx = audio_idx * 10
                    key_list.append(sensor_idx)
            else:
                # Cache hit
                audio_exemplar_indices = self.memory['Audio_exemplar_idx'][input['idx'].item()]
                key_list = []
                for audio_idx in audio_exemplar_indices:
                    sensor_idx = audio_idx * 10
                    key_list.append(sensor_idx)
        else:
            # Pretraining examples
            key_list = input['exemplar_idx'].cpu().numpy().tolist()[0]
            key_list = random.sample(key_list, 2)

        assert len(key_list) == 2
        exemplar_list = []
        for key in key_list:
            key = int(key)
            for key_size in [10, 20, 40]:
                key = max(40, key)
                key_sensor = sensor[:, max(key - key_size, 0): key + key_size + 1, :]
                exemplar_list.append(key_sensor)
        assert len(exemplar_list) == 6
        mask = self.down_sample(mask.transpose(-2, -1)).transpose(-2, -1)
        unfold_sensor = sensor.unfold(1, 10, 10)
        unfold_sensor = rearrange(unfold_sensor, 'b l c w -> b l (c w)')
        sensor = self.window_projection(sensor.transpose(-2, -1)).transpose(-2, -1)
        result['qual']['embedding_before'] = unfold_sensor.detach().cpu()
        result['qual']['embedding_after'] = sensor.detach().cpu()
        dis_pre_loss = distance_preserving_loss(unfold_sensor.squeeze(0), sensor.squeeze(0), mask.squeeze())
        result['dis_pre_loss'] = dis_pre_loss
        new_exemplar_list = [self.window_projection(ex.transpose(-1, -2)).transpose(-1, -2) for ex in exemplar_list]
        corr_feat_list = []
        for exemplar_idx, exemplar in enumerate(new_exemplar_list):
            simi_feat = self.get_simi(sensor, exemplar)
            corr_feat_list.append(simi_feat)
        corr_feat_stack = torch.stack(corr_feat_list, dim=1).squeeze(-1)
        result['qual']['corr_feat'] = corr_feat_stack.squeeze().transpose(-1, -2).detach().cpu().numpy()
        corre_feat = self.fusion(corr_feat_stack.transpose(-1, -2), sensor, mask) # B L C 1 x 1400 x 128
        corre_feat_1, corre_feat_2, corre_feat_3 = self.MultiScale(corre_feat.transpose(-1, -2))
        sensor = self.raw_sensor(sensor.transpose(-1, -2)).transpose(-1, -2)
        sensor_1, sensor_2, sensor_3 = self.MultiScale(sensor.transpose(-1, -2))
        result['corre_feat'] = corre_feat_1.transpose(-1, -2)
        result['sensor'] = sensor_1.transpose(-1, -2)
        result['corre_feat_2'] = corre_feat_2.transpose(-1, -2)
        result['sensor_2'] = sensor_2.transpose(-1, -2)
        result['corre_feat_3'] = corre_feat_3.transpose(-1, -2)
        result['sensor_3'] = sensor_3.transpose(-1, -2)
        result['mask'] = mask.squeeze(-1) # 1 x 2800

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


class fusion_block(nn.Module):
    def __init__(self):
        super(fusion_block, self).__init__()
        self.f1 = fusion_module(in_channel=64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 96, 3, padding=1),
            nn.GELU(),
        )
        self.f2 = fusion_module(in_channel=96)
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 128, 3, padding=1),
            nn.GELU(),
        )
        self.f3 = fusion_module(in_channel=128)

        self.conv1_corre = nn.Sequential(
            nn.Conv1d(6, 12, 3, padding=1),
            nn.GELU(),
        )
        self.conv2_corre = nn.Sequential(
            nn.Conv1d(12, 24, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, corre_feat, sensor_feat, mask=None):
        # B L C
        # Fusion 1
        fcorre_feat = corre_feat.mean(-1)
        fcorre_feat = fcorre_feat.unsqueeze(-1)
        fcorre_feat = fcorre_feat.repeat(1, 1, sensor_feat.shape[-1])
        assert fcorre_feat.shape == sensor_feat.shape
        sensor_feat = rearrange(sensor_feat, 'b l c -> b c l')
        fcorre_feat = rearrange(fcorre_feat, 'b l c -> b c l')
        sensor_feat = self.f1(fcorre_feat, sensor_feat)

        # Conv 1
        sensor_feat = rearrange(sensor_feat, 'b l c -> b c l')
        sensor_feat = self.conv1(sensor_feat)
        sensor_feat = rearrange(sensor_feat, 'b c l -> b l c')
        corre_feat = rearrange(corre_feat, 'b l c -> b c l')
        corre_feat = self.conv1_corre(corre_feat)
        corre_feat = rearrange(corre_feat, 'b c l -> b l c')

        # Fusion 2
        fcorre_feat = corre_feat.mean(-1)
        fcorre_feat = fcorre_feat.unsqueeze(-1)
        fcorre_feat = fcorre_feat.repeat(1, 1, sensor_feat.shape[-1])
        assert fcorre_feat.shape == sensor_feat.shape
        sensor_feat = rearrange(sensor_feat, 'b l c -> b c l')
        fcorre_feat = rearrange(fcorre_feat, 'b l c -> b c l')
        sensor_feat = self.f2(fcorre_feat, sensor_feat)
        corre_feat = rearrange(corre_feat, 'b l c -> b c l')
        corre_feat = self.conv2_corre(corre_feat)
        corre_feat = rearrange(corre_feat, 'b c l -> b l c')

        # Conv 2
        sensor_feat = rearrange(sensor_feat, 'b l c -> b c l')
        sensor_feat = self.conv2(sensor_feat)
        sensor_feat = rearrange(sensor_feat, 'b c l -> b l c')

        # Module 3
        fcorre_feat = corre_feat.mean(-1)
        fcorre_feat = fcorre_feat.unsqueeze(-1)
        fcorre_feat = fcorre_feat.repeat(1, 1, sensor_feat.shape[-1])
        assert fcorre_feat.shape == sensor_feat.shape
        sensor_feat = rearrange(sensor_feat, 'b l c -> b c l')
        fcorre_feat = rearrange(fcorre_feat, 'b l c -> b c l')
        sensor_feat = self.f3(fcorre_feat, sensor_feat)

        # Mutiply with mask
        mask_repeat = mask.repeat(1, 1, sensor_feat.shape[-1])
        assert sensor_feat.shape == mask_repeat.shape
        sensor_feat = sensor_feat * mask_repeat

        return sensor_feat


class fusion_module(nn.Module):
    def __init__(self, in_channel):
        super(fusion_module, self).__init__()
        # assert in_channel % 3 == 0
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.GroupNorm(16, in_channel)
        # self.norm = nn.BatchNorm1d(num_features=in_channel)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, corr_feat, sensor_feat):
        fusion_feat = sensor_feat * corr_feat
        fusion_feat = self.conv(fusion_feat)
        sensor_feat = sensor_feat + fusion_feat
        sensor_feat = self.norm(sensor_feat)
        sensor_feat = self.act(sensor_feat)
        sensor_feat = self.dropout(sensor_feat)
        sensor_feat = rearrange(sensor_feat, 'b c l -> b l c')
        return sensor_feat


class MultiScaleModule(nn.Module):
    def __init__(self, proj_dim, series_size):
        super(MultiScaleModule, self).__init__()
        self.proj1 = nn.Conv1d(in_channels=128,
                               out_channels=proj_dim,
                               kernel_size=2,
                               stride=2,
                               bias=False,)

        self.proj2 = nn.Conv1d(in_channels=128,
                               out_channels=proj_dim,
                               kernel_size=4,
                               stride=4,
                               bias=False, )

        self.proj3 = nn.Conv1d(in_channels=128,
                               out_channels=proj_dim,
                               kernel_size=8,
                               stride=8,
                               bias=False)

        self.norm = nn.GroupNorm(64, proj_dim)
        self.act = nn.GELU()

    def forward(self, feat):
        # feat b c l
        feat_1 = self.proj1(feat)
        feat_1 = self.norm(feat_1)
        feat_1 = self.act(feat_1)

        feat_2 = self.proj2(feat)
        feat_2 = self.norm(feat_2)
        feat_2 = self.act(feat_2)

        feat_3 = self.proj3(feat)
        feat_3 = self.norm(feat_3)
        feat_3 = self.act(feat_3)
        return feat_1, feat_2, feat_3

'''
For DPL Loss
'''

def gaussian_similarity_matrix(X, sigma):
    X_sqnorms = torch.sum(X ** 2, dim=1, keepdim=True)
    distances_sq = -2 * torch.matmul(X, X.t()) + X_sqnorms + X_sqnorms.t()
    similarities = torch.exp(-distances_sq / (2 * sigma ** 2))
    return similarities


def adjacency_matrix(S_low, threshold=None, k=None):
    if threshold is not None:
        W = (S_low > threshold).float() * S_low
    elif k is not None:
        W = torch.zeros_like(S_low)
        _, topk_indices = S_low.topk(k + 1, dim=1)
        for i, neighbors in enumerate(topk_indices):
            W[i, neighbors[1:]] = S_low[i, neighbors[1:]]
    else:
        raise ValueError("Either threshold or k must be provided")
    return W


def laplacian_matrix(W):
    D = torch.diag(W.sum(dim=1))
    L = D - W
    return L


def distance_preserving_loss(low_dim_features, high_dim_features, mask, sigma_low=1.0, threshold=0, k=150):
    S_low = gaussian_similarity_matrix(low_dim_features, sigma_low)
    W = adjacency_matrix(S_low, threshold, k)
    mask = mask.view(-1, 1)
    W = W * mask * mask.t()
    L = laplacian_matrix(W)
    Y = high_dim_features
    L_Laplacian = torch.trace(Y.t() @ L @ Y)
    return L_Laplacian
