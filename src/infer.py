from omegaconf import DictConfig
import os.path as osp
from pytorch_lightning import Trainer
from src.datamodules.module import CNetworkDataModule
from src.models.network import create_model

def infer(config: DictConfig):
    data_module = CNetworkDataModule(
        lf_name=config.lf_name,
        path=config.data_path,
        read_mode=config.read_mode,
        anchor_mode=config.anchor_mode,
        angular_resolution=config.angular_resolution
    )
    checkpoint_dir = osp.join(config.result_dir, config.lf_name,
                              f'checkpoints_c{config.channel_number[0]}_a{config.angular_number[0]}')

    checkpoint_path = osp.join(checkpoint_dir, 'model.ckpt')
    trainer = Trainer(
        devices=config.num_gpus,
        num_nodes=config.num_nodes,
        accelerator=config.accelerator)
    model = create_model(config)
    # Launch model training
    trainer.test(model, data_module, ckpt_path=checkpoint_path)

    return