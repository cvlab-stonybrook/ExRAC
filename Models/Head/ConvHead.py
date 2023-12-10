import torch
import torch.nn as nn
from Register import COUNTING_HEAD_REGISTER_MACHINE

@COUNTING_HEAD_REGISTER_MACHINE.register()
class ConvHead(nn.Module):
    def __init__(self, cfg, device):
        super(ConvHead, self).__init__()
        '''
        ConvHead
        In: query feature batch * token_num * token_dim
        fusion feature -> density map
        '''
        self.cfg = cfg.ConvHead
        self.Enc_cfg = cfg[cfg.Feature]

        self.conv_cor = nn.Sequential(
            nn.Conv1d(self.Enc_cfg.proj_dim, 128, 7, stride=1, padding=3),
            # nn.BatchNorm1d(num_features=512),
            # nn.GroupNorm(128, 512),
            nn.GELU(),
            # nn.Conv1d(512, 256, 3, stride=1, padding=1),
            # nn.BatchNorm1d(num_features=256),
            # nn.GroupNorm(64, 256),
            # nn.ReLU(),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            # nn.BatchNorm1d(num_features=64),
            # nn.GroupNorm(16, 64),
            nn.GELU(),
        )

        self.conv_raw = nn.Sequential(
            nn.Conv1d(self.Enc_cfg.proj_dim, 128, 7, stride=1, padding=3),
            # nn.BatchNorm1d(num_features=512),
            # nn.GroupNorm(128, 512),
            nn.GELU(),
            # nn.Conv1d(512, 256, 3, stride=1, padding=1),
            # nn.BatchNorm1d(num_features=256),
            # nn.GroupNorm(64, 256),
            # nn.ReLU(),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            # nn.BatchNorm1d(num_features=64),
            # nn.GroupNorm(16, 64),
            nn.GELU(),
        )

        self.down_sample = nn.MaxPool1d(3, stride=2, padding=1)
        self.down_sample_2 = nn.MaxPool1d(5, stride=4, padding=2)
        self.mask_down_sample = nn.MaxPool1d(9, stride=8, padding=4)
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(64 * 6, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv1d(64 * 6, 64 * 2, 3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Conv1d(64, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            # elif isinstance(m, nn.Conv1d):
            #    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input):
        sensor_feat = input['sensor']
        corre_feat = input['corre_feat']
        corre_feat_2 = input['corre_feat_2']
        sensor_feat_2 = input['sensor_2']
        corre_feat_3 = input['corre_feat_3']
        sensor_feat_3 = input['sensor_3']
        mask = input['mask']

        sensor_feat = sensor_feat.permute(0, 2, 1)
        corre_feat = corre_feat.permute(0, 2, 1)
        corre_feat_2 = corre_feat_2.permute(0, 2, 1)
        sensor_feat_2 = sensor_feat_2.permute(0, 2, 1)
        corre_feat_3 = corre_feat_3.permute(0, 2, 1)
        sensor_feat_3 = sensor_feat_3.permute(0, 2, 1)

        sensor_feat = self.conv_raw(sensor_feat)
        corre_feat = self.conv_cor(corre_feat)
        sensor_feat_2 = self.conv_raw(sensor_feat_2)
        corre_feat_2 = self.conv_cor(corre_feat_2)
        sensor_feat_3 = self.conv_raw(sensor_feat_3)
        corre_feat_3 = self.conv_cor(corre_feat_3)

        mask = mask.unsqueeze(1)
        mask = self.mask_down_sample(mask)

        sensor_feat = self.down_sample_2(sensor_feat)
        corre_feat = self.down_sample_2(corre_feat)
        sensor_feat_2 = self.down_sample(sensor_feat_2)
        corre_feat_2 = self.down_sample(corre_feat_2)

        corre_feat = torch.cat([corre_feat, corre_feat_2, corre_feat_3, sensor_feat, sensor_feat_2, sensor_feat_3],
                               dim=1)
        # sensor_feat = torch.cat([sensor_feat, sensor_feat_2, sensor_feat_3], dim=1)

        # sensor_density = self.conv_fusion(sensor_feat)
        density = self.conv_fusion(corre_feat)
        # print('mmmmmmmmmmmmmmmmm', mask.shape)
        # print('ddddddddddddddddddd', density.shape)
        density = density * mask
        # sensor_density_2 = self.conv_fusion(sensor_feat_2)
        # corre_density_2 = self.conv_fusion(corre_feat_2)
        # sensor_density_3 = self.conv_fusion(sensor_feat_3)
        # corre_density_3 = self.conv_fusion(corre_feat_3)

        # density = torch.cat([sensor_density, corre_density, sensor_density_2, corre_density_2, corre_density_3, sensor_density_3], dim=1)
        # density = torch.cat([corre_density, corre_density_2, corre_density_3], dim=1)
        if density.shape[1] != 1:
            density = density.mean(1)
        else:
            density = density.squeeze(1)
        # print(density.shape)
        input['density'] = density
        input['qual']['density'] = density.detach().cpu().squeeze().numpy()
        return input