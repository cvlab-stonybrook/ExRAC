from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def load_cfg(file_path: str):
    cfg = OmegaConf.load(file_path)
    return cfg


def merge_from_file(cfg: DictConfig, file_path: str):
    cfg2 = OmegaConf.load(file_path)
    cfg_new = OmegaConf.merge(cfg, cfg2)
    return cfg_new


def show_cfg(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


def save_cfg(cfg: DictConfig, save_path: str):
    OmegaConf.save(config=cfg, f=save_path)


def dict_to_cfg(dict_data: dict):
    result_conf = OmegaConf.create(dict_data)
    return result_conf
