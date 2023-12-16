import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(cfg,
                    dataloader: DataLoader,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    criterion,
                    Result: dict,
                    epoch: int,
                    device: str,
                    pesudo_density_criterion = None,):
    sample_num = 0
    model.train()
    if cfg.Feature == 'ExEnc':
        model.feature.audio_model.eval()
    METRIC_E_list = []
    METRIC_SE_list = []
    METRIC_CL_list = []
    METRIC_DL_list = []
    METRIC_PL_list = []
    METRIC_L_list = []
    METRIC_DPL_list = []
    acc_loss = 0.
    acc_grad_batch = cfg.Optimizer.acc_grad_batch
    acc_grad_index = 0
    for idx, data in tqdm(enumerate(dataloader)):
        data['count'] = data['count'].to(device)
        data['sensor'] = data['sensor'].to(device)
        data['density'] = data['density'].to(device)
        if pesudo_density_criterion is not None:
            data['pretrain'] = True
        # print(data['sensor'].shape)
        # Pred
        result = model(data)
        pred_density = result['density']
        sample_num += pred_density.shape[0]
        pesudo_density_loss = 0.
        if 'pesudo_density' in result.keys():
            pesudo_density = result['pesudo_density']
            if result['pretrain']:
                gt_density = data['density']
                if gt_density.shape[-1] != pesudo_density.shape[-1]:
                    gt_density = torch.nn.functional.interpolate(gt_density.unsqueeze(1),
                                                                      size=(pesudo_density.shape[-1]),
                                                                      mode='linear').squeeze(1)
                    gt_density = gt_density * data['count'] / gt_density.sum()
                gt_density = gt_density.float()
                pesudo_density_loss = pesudo_density_criterion(pesudo_density, gt_density)['loss']
            else:
                pesudo_density_loss = 0.
                data['density'] = pesudo_density
        # Loss
        if pred_density.shape[0] == 1:
            pred_count = pred_density.sum()
            error = np.absolute((pred_count - data['count']).detach().cpu().numpy()).item()
            square_error = (pred_count - data['count']).pow(2).item()
            if data['density'].shape[-1] != pred_density.shape[-1]:
                data['density'] = torch.nn.functional.interpolate(data['density'].unsqueeze(1), size=(pred_density.shape[-1]), mode='linear').squeeze(1)
                data['density'] = data['density'] * data['count'] / data['density'].sum()
            data['density'] = data['density'].float()
            all_losses = criterion(pred_density, data['density'])
            count_loss = all_losses['count_loss']
            density_loss = all_losses['density_loss']
            loss = all_losses['loss']
        else:
            # Only baselines batch size can greater than 1 when padding.
            density_loss= 0.
            pred_count = pred_density.sum(dim=-1)
            loss = (pred_count - data['count']).pow(2).mean()
            error = np.absolute((pred_count - data['count']).detach().cpu().numpy()).sum()
            count_loss = (pred_count - data['count']).pow(2).sum().item()
            square_error = count_loss
        # pesudo_density_loss = pesudo_density_loss.float()
        loss += 100000 * pesudo_density_loss
        if 'dis_pre_loss' in result.keys():
            dis_pre_loss = 0.05 * result['dis_pre_loss']
        else:
            dis_pre_loss = 0.
        loss += dis_pre_loss
        if cfg.Train.batch_size == 1 and acc_grad_batch > 1:
            acc_grad_index += 1
            acc_loss += loss
            if acc_grad_index % acc_grad_batch == 0 or acc_grad_index == sample_num:
                optimizer.zero_grad()
                acc_loss.backward()
                optimizer.step()
                acc_loss = 0.
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Update metric list
        METRIC_E_list.append(error)
        METRIC_SE_list.append(square_error)
        METRIC_CL_list.append(count_loss)
        METRIC_DL_list.append(density_loss)
        if not isinstance(pesudo_density_loss, float):
            METRIC_PL_list.append(pesudo_density_loss.item())
        else:
            METRIC_PL_list.append(pesudo_density_loss)
        if not isinstance(dis_pre_loss, float):
            METRIC_DPL_list.append(dis_pre_loss.item())
        else:
            METRIC_DPL_list.append(dis_pre_loss)

    mae = sum(METRIC_E_list) / sample_num
    #OBO
    obo = 0.
    for ae in METRIC_E_list:
        if ae <= 1:
            obo += 1
    obo = obo / sample_num
    rmse = np.sqrt(sum(METRIC_SE_list) / sample_num)
    count_loss = sum(METRIC_CL_list) / sample_num
    density_loss = sum(METRIC_DL_list) / sample_num
    pesudo_density_loss = sum(METRIC_PL_list) / sample_num
    distance_loss = sum(METRIC_DPL_list) / sample_num
    Result[epoch + 1]['train']['MAE'] = float(mae)
    Result[epoch + 1]['train']['RMSE'] = float(rmse)
    Result[epoch + 1]['train']['OBO'] = float(obo)
    Result[epoch + 1]['train']['CountLoss'] = float(count_loss)
    Result[epoch + 1]['train']['DensityLoss'] = float(density_loss)
    Result[epoch + 1]['train']['PesudoLoss'] = float(pesudo_density_loss)
    Result[epoch + 1]['train']['distance_loss'] = float(distance_loss)
    torch.cuda.empty_cache()
    return Result
