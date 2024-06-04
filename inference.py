import hydra
from omegaconf import DictConfig
from utils.misc_utils import check_input
from src.infer import infer
import torch

@hydra.main(
    config_path='configs/',
    config_name='train_lf.yaml'
)

def main(config: DictConfig):
    check_input(config)
    infer(config)

if __name__ == '__main__':
    main()