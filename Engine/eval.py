import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def eval_one_epoch(cfg,
                   dataloader: DataLoader,
                    model: nn.Module,
                    split: str,
                    Result: dict,
                    criterion,
                    epoch: int,
                    device: str,):
    assert split in ['val', 'test', 'partb_eval', 'partb_val']
    sample_num = 0
    model.eval()
    METRIC_E_list = []
    METRIC_SE_list = []
    METRIC_CL_list = []
    METRIC_DL_list = []
    for idx, data in tqdm(enumerate(dataloader)):
        data['count'] = data['count'].to(device)
        # data['audio'] = data['audio'].to(device)
        data['sensor'] = data['sensor'].to(device)
        data['density'] = data['density'].to(device)
        # print(data['idx'])
        # Pred
        result = model(data)
        pred_density = result['density']
        sample_num += pred_density.shape[0]
        if pred_density.shape[0] == 1:
            pred_count = pred_density.sum()
            error = np.absolute((pred_count - data['count']).detach().cpu().numpy()).item()
            square_error = (pred_count - data['count']).pow(2).item()
            if data['density'].shape[-1] != pred_density.shape[-1]:
                data['density'] = torch.nn.functional.interpolate(data['density'].unsqueeze(1), size=(pred_density.shape[-1]), mode='linear').squeeze(1)
                data['density'] = data['density'] * data['count'] / data['density'].sum()
            all_losses = criterion(pred_density, data['density'])
            count_loss = all_losses['count_loss']
            density_loss = all_losses['density_loss']
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
        METRIC_CL_list.append(count_loss)
        METRIC_DL_list.append(density_loss)


    mae = sum(METRIC_E_list) / sample_num
    # OBO
    obo = 0.
    for ae in METRIC_E_list:
        if ae <= 1:
            obo += 1
    obo = obo / sample_num
    rmse = np.sqrt(sum(METRIC_SE_list) / sample_num)
    count_loss = sum(METRIC_CL_list) / sample_num
    density_loss = sum(METRIC_DL_list) / sample_num
    print('Evaluation on {}, sample num {}'.format(split, sample_num))
    Result[epoch + 1][split]['MAE'] = float(mae)
    Result[epoch + 1][split]['RMSE'] = float(rmse)
    Result[epoch + 1][split]['OBO'] = float(obo)
    Result[epoch + 1][split]['CountLoss'] = float(count_loss)
    Result[epoch + 1][split]['DensityLoss'] = float(density_loss)
    return Result
