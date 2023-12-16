import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=False, default='vanilla')
parser.add_argument('--name', required=False, default='ExRAC')
parser.add_argument('--dataset', type=str, default=None, required=False)
parser.add_argument('--is_local', action='store_true', default=False, required=False)
parser.add_argument('--pretrain', action='store_true', default=False, required=False)
parser.add_argument('--pretrain_epochs', type=int, default=0, required=False)
parser.add_argument('--save_model', action='store_true', default=False, required=False)
parser.add_argument('--gpu_id', type=int, default=0, required=False)
parser.add_argument('--seed', type=int, default=3407, required=False)
parser.add_argument('--feature', type=str, required=False, default=None)
parser.add_argument('--head', type=str, required=False, default=None)
parser.add_argument('--split_type', type=str, required=False, default=None)
parser.add_argument('--case_analyze', action='store_true', default=False, required=False)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
config = args.config
custom_name = args.name
is_local = args.is_local
pretrain = args.pretrain
exp_setting_path = './Config/' + config +'.yaml'
import torch
import random
import datetime
import numpy as np
from omegaconf import OmegaConf
from Engine import train_one_epoch, eval_one_epoch
from Utils.config import save_cfg, load_cfg, show_cfg
from Builder import build_dataset, build_model, build_optimizer, build_criterion
###############################################################################################################################


# Experiemtn Log
if is_local:
    log_root = './log'
else:
    log_root = '../log'
current_time = datetime.datetime.now()
experiment_date = current_time.strftime("%m-%d-%Y %H-%M")

# Load experiment config and update the config with args
cfg = load_cfg(exp_setting_path)
cfg.Pretrain = pretrain
cfg.Is_local = is_local
if args.dataset is not None:
    cfg.Dataset = args.dataset
if args.feature is not None:
    cfg.Feature = args.feature
if args.head is not None:
    cfg.Head = args.head
if args.split_type is not None:
    cfg.Dataset.split_type = args.split_type
if args.seed is None:
    seed = cfg.Seed
else:
    seed = args.seed
assert cfg.Train.batch_size == 1, 'Batch size shoule 1, you can set the accumulated gradient in config to simulate a larger batch size.'
# Set the experiment name
# The experiment name is formulated as (model)feature_head_(dataset)dataset_zero_host(optimizer)optimizer_lr_criterion_(training)epoch_Batchsize_(seed)_custom_time
exp_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                            custom_name,
                            experiment_date,
                            cfg.Feature,
                            cfg.Head,
                            cfg.Dataset.name,
                            cfg.Optimizer.name,
                            cfg.Optimizer.feature_lr,
                            cfg.Optimizer.head_lr,
                            cfg.Criterion,
                            cfg.Train.epoch,
                            cfg.Train.batch_size,
                            cfg.Seed,
                            )

log_save_dir = os.path.join(log_root, exp_name)
if not os.path.exists(log_save_dir):
    os.makedirs(log_save_dir)
cfg.log_save_dir = log_save_dir
config_save_dir = os.path.join(log_save_dir, 'config.yaml')
best_model_save_dir = os.path.join(log_save_dir, exp_name + '_BEST.pth')
last_model_save_dir = os.path.join(log_save_dir, exp_name + '_LAST.pth')


show_cfg(cfg)
print(exp_name)

# For reproducibility, fix the seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)
print('Dataset: ', cfg.Dataset)
print('Counting Head Model: ', cfg.Head)
print('Feature Model: ', cfg.Feature)
print('Random Seed: ', seed)
print('Split Type: ', cfg.Dataset.split_type)
print('Pretrain Epochs: ', args.pretrain_epochs)
save_cfg(cfg, config_save_dir)
# Single GPU
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = 'cpu'
# Result Dict
Result = {}
# Setting
Epoch = cfg.Train.epoch
train_loader, val_loader, test_loader, pretrain_loader, = build_dataset(cfg)
model = build_model(cfg, device)
optimizer, scheduler = build_optimizer(cfg, model)
criterion = build_criterion(cfg)

best_MAE = 1e6
best_RMSE = 1e6
best_OBO = 0
test_MAE = 1e6
test_RMSE = 1e6
test_OBO = 0
test_Partb_MAE = 1e6
test_Partb_RMSE = 1e6
test_Partb_OBO = 0
if pretrain:
    assert pretrain_loader is not None, 'Pretrain loader is None, please check the config file.'
    pretrainEpoch = args.pretrain_epochs
    from Builder.build_optimizer import combine_loss, density_loss, count_loss
    pretrain_criterion = count_loss()
    pesudo_criterion = density_loss()
    for pretrain_epoch in range(pretrainEpoch):
        pretrain_Result = {}
        pretrain_Result[pretrain_epoch + 1] = {}
        pretrain_Result[pretrain_epoch + 1]['train'] = {}
        pretrain_Result[pretrain_epoch + 1]['test'] = {}
        pretrain_Result[pretrain_epoch + 1]['val'] = {}
        pretrain_Result[pretrain_epoch + 1]['best val'] = {}
        model.train()
        print('#################################################################')
        print('Pretrain Epoch:', pretrain_epoch)
        # Train one epoch
        pretrain_Result = train_one_epoch(cfg=cfg,
                                 dataloader=pretrain_loader,
                                 model=model,
                                 optimizer=optimizer,
                                 criterion=pretrain_criterion,
                                 Result=pretrain_Result,
                                 epoch=pretrain_epoch,
                                 device=device,
                                 pesudo_density_criterion = pesudo_criterion,)

        # Verbose training
        train_mae = pretrain_Result[pretrain_epoch + 1]['train']['MAE']
        train_rmse = pretrain_Result[pretrain_epoch + 1]['train']['RMSE']
        train_obo = pretrain_Result[pretrain_epoch + 1]['train']['OBO']
        train_count_loss = pretrain_Result[pretrain_epoch + 1]['train']['CountLoss']
        train_density_loss = pretrain_Result[pretrain_epoch + 1]['train']['DensityLoss']
        pesudo_loss = pretrain_Result[pretrain_epoch + 1]['train']['PesudoLoss']
        distance_loss = pretrain_Result[pretrain_epoch + 1]['train']['distance_loss']
        print('PreTrain MAE: ', train_mae)
        print('PreTrain RMSE: ', train_rmse)
        print('PreTrain OBO: ', train_obo)
        print('PreTrain count Loss: ', train_count_loss)
        print('PreTrain density Loss: ', train_density_loss)
        print('PreTrain pesudo Loss: ', pesudo_loss)
        print('PreTrain distance Loss: ', distance_loss)

for epoch in range(Epoch):
    Result[epoch + 1] = {}
    Result[epoch + 1]['train'] = {}
    Result[epoch + 1]['test'] = {}
    Result[epoch + 1]['val'] = {}
    Result[epoch + 1]['partb_eval'] = {}
    Result[epoch + 1]['best val'] = {}
    model.train()
    print('#################################################################')
    print('Epoch:', epoch)
    # Train one epoch
    Result = train_one_epoch(cfg = cfg,
                    dataloader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    Result=Result,
                    epoch=epoch,
                    device=device,)

    # Verbose training
    train_mae = Result[epoch + 1]['train']['MAE']
    train_rmse = Result[epoch + 1]['train']['RMSE']
    train_obo = Result[epoch + 1]['train']['OBO']
    train_count_loss = Result[epoch + 1]['train']['CountLoss']
    train_density_loss = Result[epoch + 1]['train']['DensityLoss']
    distance_loss = Result[epoch + 1]['train']['distance_loss']
    print('Train MAE: ', train_mae)
    print('Train RMSE: ', train_rmse)
    print('Train OBO: ', train_obo)
    print('Train count Loss: ', train_count_loss)
    print('Train density Loss: ', train_density_loss)
    print('Train distance Loss: ', distance_loss)

    print('Evaluation: ')
    with torch.no_grad():
        Result = eval_one_epoch(
                            cfg = cfg,
                            dataloader=val_loader,
                            model=model,
                            split='val',
                            criterion=criterion,
                            Result=Result,
                            epoch=epoch,
                            device=device,)

    # Verbose val
    val_mae = Result[epoch + 1]['val']['MAE']
    val_rmse = Result[epoch + 1]['val']['RMSE']
    val_obo = Result[epoch + 1]['val']['OBO']
    val_count_loss = Result[epoch + 1]['val']['CountLoss']
    val_density_loss = Result[epoch + 1]['val']['DensityLoss']

    print('Val MAE: ', val_mae)
    print('Val RMSE: ', val_rmse)
    print('Val OBO: ', val_obo)
    print('Val count Loss: ', val_count_loss)
    print('Val density Loss: ', val_density_loss)

    if val_mae < best_MAE:
        best_MAE = val_mae
        best_RMSE = val_rmse
        best_OBO = val_obo
        # This shoould be the test
        # However currently we don't have enough data to do it
        # So we simply pass
        with torch.no_grad():
            Result = eval_one_epoch(
                cfg = cfg,
                dataloader=test_loader,
                model=model,
                split='test',
                criterion=criterion,
                Result=Result,
                epoch=epoch,
                device=device, )

        test_mae = Result[epoch + 1]['test']['MAE']
        test_rmse = Result[epoch + 1]['test']['RMSE']
        test_obo = Result[epoch + 1]['test']['OBO']
        test_count_loss = Result[epoch + 1]['test']['CountLoss']
        test_density_loss = Result[epoch + 1]['test']['DensityLoss']

        print('Test MAE: ', test_mae)
        print('Test RMSE: ', test_rmse)
        print('Test OBO: ', test_obo)
        print('Test count Loss: ', test_count_loss)
        print('Test density Loss: ', test_density_loss)

        test_MAE = test_mae
        test_RMSE = test_rmse
        test_OBO = test_obo
        if args.save_model:
            torch.save(model.state_dict(), best_model_save_dir)

    if args.save_model:
        torch.save(model.state_dict(), last_model_save_dir)
    Result[epoch + 1]['best val']['MAE'] = best_MAE
    Result[epoch + 1]['best val']['RMSE'] = best_RMSE
    Result[epoch + 1]['best val']['OBO'] = best_OBO
    scheduler.step()
print('Final Best Val MAE:', best_MAE)
print('Final Best Val RMSE:', best_RMSE)
print('Final Best Val OBO:', best_OBO)
print('Final Test MAE:', test_MAE)
print('Final Test RMSE:', test_RMSE)
print('Final Test OBO:', test_OBO)
print('Final PartB Test MAE:', test_Partb_MAE)
print('Final PartB Test RMSE:', test_Partb_RMSE)
print('Final PartB Test OBO:', test_Partb_OBO)
Result['Final Result'] = {}
Result['Final Result']['best val'] = {}
Result['Final Result']['best val']['MAE'] = best_MAE
Result['Final Result']['best val']['RMSE'] = best_RMSE
Result['Final Result']['best val']['OBO'] = best_OBO
Result['Final Result']['test'] = {}
Result['Final Result']['test']['MAE'] = test_MAE
Result['Final Result']['test']['RMSE'] = test_RMSE
Result['Final Result']['test']['OBO'] = test_OBO
Result['Final Result']['test_partB'] = {}
Result['Final Result']['test_partB']['MAE'] = test_Partb_MAE
Result['Final Result']['test_partB']['RMSE'] = test_Partb_RMSE
Result['Final Result']['test_partB']['OBO'] = test_Partb_OBO
result_conf = OmegaConf.create(Result)
Result_save_path = os.path.join(log_save_dir, 'Result.yaml')
save_cfg(result_conf, Result_save_path)
