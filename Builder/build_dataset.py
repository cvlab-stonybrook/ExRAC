import copy
import random
from Register import DATASET_REGISTER_MACHINE
from torch.utils.data import DataLoader, Sampler, ConcatDataset
import Dataset

def build_dataset(cfg, pretrain_proportion=None, audio_anno = False) -> (DataLoader, DataLoader, DataLoader):
    assert cfg.Dataset.name in ['DWC'], NotImplementedError
    train, val, test = DATASET_REGISTER_MACHINE.get(cfg.Dataset.name)(cfg, 'train', audio_anno), \
                       DATASET_REGISTER_MACHINE.get(cfg.Dataset.name)(cfg, 'val', audio_anno), \
                       DATASET_REGISTER_MACHINE.get(cfg.Dataset.name)(cfg, 'test', audio_anno)
    # Prepare Pretrain Loader
    if cfg.Pretrain:
        pretrain = DATASET_REGISTER_MACHINE.get('DWC_syn')(cfg, pretrain_proportion)
        pretrain_loader = DataLoader(pretrain, batch_size=cfg.Train.batch_size, shuffle=True)
    else:
        pretrain = None
        pretrain_loader = None
    train_loader = DataLoader(train, batch_size=cfg.Train.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=cfg.Train.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=cfg.Train.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, pretrain_loader
