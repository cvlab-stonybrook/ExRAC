'''
General model builder.
By yifeng(yifehuang@cs.stonybrook.edu)
'''
import Models
import torch.nn as nn
from Register import FEATURE_REGISTER_MACHINE
from Register import COUNTING_HEAD_REGISTER_MACHINE


def build_model(cfg, device):
    class AudioSensorModel(nn.Module):
        def __init__(self, cfg, device):
            super().__init__()
            self.feature = FEATURE_REGISTER_MACHINE.get(cfg.Feature)(cfg, device)
            self.head = COUNTING_HEAD_REGISTER_MACHINE.get(cfg.Head)(cfg, device)

        def forward(self, input):
            output = self.feature(input)
            return self.head(output)

    return AudioSensorModel(cfg, device).to(device)
