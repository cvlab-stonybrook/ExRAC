import torch
import torch.nn as nn
import torch.optim as optim

def build_criterion(cfg):
    assert cfg.Criterion in ['density', 'count', 'combine']
    if cfg.Criterion == 'density':
        return density_loss()
    elif cfg.Criterion == 'count':
        return count_loss()
    else:
        return combine_loss()


def build_optimizer(cfg, model: nn.Module) -> \
        (optim.Optimizer, optim.lr_scheduler.ExponentialLR):
    assert cfg.Optimizer.name in ['adamw', 'adam']
    if cfg.Optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW\
        (
            [{'params': model.feature.parameters(), 'lr': cfg.Optimizer.feature_lr},
             {'params': model.head.parameters(), 'lr': cfg.Optimizer.head_lr},],
            weight_decay = cfg.Optimizer.weight_decay
        )
    elif cfg.Optimizer.name == 'adam':
        optimizer = torch.optim.Adam \
                (
                [{'params': model.feature.parameters(), 'lr': cfg.Optimizer.feature_lr},
                 {'params': model.head.parameters(), 'lr': cfg.Optimizer.head_lr}, ],
                weight_decay=cfg.Optimizer.weight_decay
            )

    return optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.Optimizer.scheduler_gamma)


class my_loss(nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()
        self.count_loss = 0.
        self.density_loss = 0.
        self.loss = 0.
        self.all_losses = {
            'count_loss': self.count_loss,
            'density_loss': self.density_loss,
            'loss': self.loss,
        }

class density_loss(my_loss):
    def __init__(self):
        super(density_loss, self).__init__()
        self.criterion = torch.nn.MSELoss()
    def forward(self, pred_density: torch.FloatTensor, gt_density: torch.FloatTensor) -> dict:
        loss = self.criterion(pred_density, gt_density)
        self.all_losses.update({
            'density_loss': loss.detach().cpu().numpy().item(),
            'loss': loss,
        })
        return self.all_losses


class count_loss(my_loss):
    def __init__(self):
        super(count_loss, self).__init__()
    def forward(self, pred_density: torch.FloatTensor, gt_density: torch.FloatTensor) -> dict:
        pred_count = pred_density.sum()
        gt_count = gt_density.sum()
        #print(pred_count, gt_count)
        loss = (pred_count - gt_count) ** 2
        #print(loss)
        self.all_losses.update({
            'count_loss': loss.detach().cpu().numpy().item(),
            'loss': loss,
        })
        return self.all_losses


class combine_loss(my_loss):
    def __init__(self, alpha=1000):
        super(combine_loss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.alpha = alpha
    def forward(self, pred_density: torch.FloatTensor, gt_density: torch.FloatTensor) -> dict:
        density_loss = self.criterion(pred_density, gt_density)
        pred_count = pred_density.sum()
        gt_count = gt_density.sum()
        count_loss = (pred_count - gt_count) ** 2
        loss = count_loss + self.alpha * density_loss
        self.all_losses.update({
            'count_loss': count_loss.detach().cpu().numpy().item(),
            'density_loss': self.alpha * density_loss.detach().cpu().numpy().item(),
            'loss': loss,
        })
        return self.all_losses
