import os
import torch
import argparse
import numpy as np
import random
from tqdm import tqdm
from Utils.config import load_cfg
from Builder import build_dataset, build_model
import torchaudio
# Fix Seed
seed = 3407
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)

# If want to use other GPU for evaluation
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = 'cpu'

# Checkpoint Path and Exp Config
check_point_path = './Checkpoints/ExRAC.pth'
cfg_path = './Config/vanilla.yaml'
cfg = load_cfg(cfg_path)
cfg.Pretrain = False
cfg.Dataset.split_type = 'action_plus'
train_loader, val_loader, test_loader, _, = build_dataset(cfg)
model = build_model(cfg, device)
model.load_state_dict(torch.load(check_point_path))
model.eval()
with torch.no_grad():
    sample_num = 0
    model.eval()
    # Absolute Error
    METRIC_E_list = []
    # Square Error
    METRIC_SE_list = []
    for idx, data in tqdm(enumerate(val_loader)):
        data['count'] = data['count'].to(device)
        data['sensor'] = data['sensor'].to(device)
        data['density'] = data['density'].to(device)
        result = model(data)
        pred_density = result['density']
        sample_num += pred_density.shape[0]
        if pred_density.shape[0] == 1:
            pred_count = pred_density.sum()
            error = np.absolute((pred_count - data['count']).detach().cpu().numpy()).item()
            square_error = (pred_count - data['count']).pow(2).item()
        else:
            density_loss = 0.
            pred_count = pred_density.sum(dim=-1)
            loss = (pred_count - data['count']).pow(2).mean()
            error = np.absolute((pred_count - data['count']).detach().cpu().numpy()).sum()
            count_loss = (pred_count - data['count']).pow(2).sum().item()
            square_error = count_loss
        # Update metric list
        METRIC_E_list.append(error)
        METRIC_SE_list.append(square_error)
    mae = sum(METRIC_E_list) / sample_num
    rmse = np.sqrt(sum(METRIC_SE_list) / sample_num)
    print('Validation Result: MAE: %.4f, RMSE: %.4f' % (mae, rmse))

with torch.no_grad():
    sample_num = 0
    model.eval()
    # Absolute Error
    METRIC_E_list = []
    # Square Error
    METRIC_SE_list = []
    for idx, data in tqdm(enumerate(test_loader)):
        data['count'] = data['count'].to(device)
        data['sensor'] = data['sensor'].to(device)
        data['density'] = data['density'].to(device)
        result = model(data)
        pred_density = result['density']
        sample_num += pred_density.shape[0]
        if pred_density.shape[0] == 1:
            pred_count = pred_density.sum()
            error = np.absolute((pred_count - data['count']).detach().cpu().numpy()).item()
            square_error = (pred_count - data['count']).pow(2).item()
        else:
            density_loss = 0.
            pred_count = pred_density.sum(dim=-1)
            loss = (pred_count - data['count']).pow(2).mean()
            error = np.absolute((pred_count - data['count']).detach().cpu().numpy()).sum()
            count_loss = (pred_count - data['count']).pow(2).sum().item()
            square_error = count_loss
        # Update metric list
        METRIC_E_list.append(error)
        METRIC_SE_list.append(square_error)
    mae = sum(METRIC_E_list) / sample_num
    rmse = np.sqrt(sum(METRIC_SE_list) / sample_num)
    print('Test Result: MAE: %.4f, RMSE: %.4f' % (mae, rmse))
