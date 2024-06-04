from typing import Any, Dict, List
from pytorch_lightning import LightningModule
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from utils.fb import calculate_FB_bases
import os

class CNNModel(nn.Module):
    def __init__(
        self,
        channel_number: List,
        angular_list: List,
        dilation_list: List,
        angular_resolution: int,
        img_height: int,
        img_width: int,
        noise_std: float,
        upsampling: str,
        num_base: int,
        noise_dir: str,
    ):
        super().__init__()

        self._angular_channel = angular_list
        self._channel_number = channel_number
        self._angular_resolution = angular_resolution
        self._img_height = img_height
        self._img_width = img_width
        self._noise_std = noise_std
        self._upsampling = upsampling
        self._dilation_list = dilation_list
        self._num_base = num_base

        self.row_kernels = nn.ParameterDict()
        self.col_kernels = nn.ParameterDict()
        self.row_biases = nn.ParameterDict()
        self.col_biases = nn.ParameterDict()
        self.spatial_kernels = nn.ParameterList()
        self.batchnorm_list = []

        # Initialization of decoding kernels
        self.decod_kernel = nn.Parameter(torch.Tensor(3, self._channel_number[-1],1,1),requires_grad=True)
        nn.init.kaiming_uniform(self.decod_kernel, a=math.sqrt(5))
        # Initialization of activation functions
        self.act_fn = nn.GELU()
        self.decod_fn = nn.Sigmoid()
        # Initialization of upsampling function
        self.up_sampler = nn.Upsample(scale_factor=2, mode=self._upsampling)
        # Initialization of the input noise
        totalupsample = 2**(len(self._channel_number) - 1)
        width = int(self._img_width/totalupsample)
        height = int(self._img_height/totalupsample)
        self.input_path = noise_dir + '/' + 'input_' + str(self._channel_number[0]) + '_' + str(height) + '_' + str(width) + '.npy'
        self.input = torch.FloatTensor(np.load(self.input_path)).to('cuda')
        # Initialization of the Bessel basis
        self.base_np, _, _ = calculate_FB_bases(1)
        if self._num_base == -1:
            self._num_base = self.base_np.shape[1]
        else:
            self.base_np = self.base_np[:, :self._num_base]
        self.base_np = self.base_np.reshape(3, 3, self._num_base).transpose(2, 0, 1)
        # TODO: Setting the basis as trainable variables
        self.bases = nn.Parameter(torch.tensor(np.array(self.base_np)).float(), requires_grad=True)
        # Initialization of other kernels
        self.construct_kernels()

    def construct_kernels(self):
        self._spatial_channel = list(map(lambda x, y: x - 2 * y, self._channel_number, self._angular_channel))

        for i in range(len(self._channel_number)):
            # Initialize filters' shape
            input_channel = self._channel_number[i] if i == 0 else self._channel_number[i-1]
            spa_channel = self._spatial_channel[i]
            ang_channel = self._angular_channel[i]
            ##########################################################
            shared_weight = nn.Parameter(torch.Tensor(spa_channel, input_channel, self._num_base),requires_grad=True)
            stdv = 1. / math.sqrt(input_channel*self._num_base)
            shared_weight.data.normal_(0, stdv)
            self.spatial_kernels.append(shared_weight)
            ##########################################################
            self.batchnorm_list.append(nn.BatchNorm2d(self._channel_number[i]).to('cuda'))

            # Individual angular kernels for different views
            for row in range(1, self._angular_resolution + 1):
                indiv_weight = nn.Parameter(torch.Tensor(ang_channel, input_channel, self._num_base),requires_grad=True)
                stdv = 1. / math.sqrt(input_channel * self._num_base)
                indiv_weight.data.normal_(0, stdv)
                self.row_kernels[str(i)+'-'+str(row)] = indiv_weight
                # self.row_kernels[str(i) + '-' + str(row)].grad = None
                # Individual bias
                bias = nn.Parameter(torch.Tensor(self._channel_number[i]), requires_grad=True)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(shared_weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
                self.row_biases[str(i)+'-'+str(row)] = bias
                # self.row_biases[str(i) + '-' + str(row)].grad = None
            for col in range(1, self._angular_resolution + 1):
                indiv_weight = nn.Parameter(torch.Tensor(ang_channel, input_channel, self._num_base),requires_grad=True)
                stdv = 1. / math.sqrt(input_channel * self._num_base)
                indiv_weight.data.normal_(0, stdv)
                self.col_kernels[str(i)+'-'+str(col)] = indiv_weight
                # self.col_kernels[str(i) + '-' + str(col)].grad = None
                # Individual bias
                bias = nn.Parameter(torch.Tensor(self._channel_number[i]), requires_grad=True)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(shared_weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
                self.col_biases[str(i)+'-'+str(col)] = bias
                # self.col_biases[str(i) + '-' + str(col)].grad = None

    def convolution2d(self, index, row, col, x, d):
        spa_filter = self.spatial_kernels[index]
        row_filter = self.row_kernels[str(index)+'-'+str(row)]
        col_filter = self.col_kernels[str(index)+'-'+str(col)]
        row_bias = self.row_biases[str(index)+'-'+str(row)].to('cuda')
        col_bias = self.col_biases[str(index)+'-'+str(col)].to('cuda')
        conv_filter = torch.einsum('abc,cdf->abdf',torch.concat([spa_filter, row_filter, col_filter], 0), self.bases).to('cuda')
        output = F.conv2d(x, conv_filter, row_bias+col_bias,padding='same',dilation=d)
        return output

    def forward(self, row, col):
        x = self.input
        # 2d Convolution in terms of light field row and column indices
        for i in range(len(self._channel_number)):
            x = self.convolution2d(i, row, col, x, self._dilation_list[i])
            if i != len(self._channel_number) - 1:
                x = self.up_sampler(x)
            x = self.batchnorm_list[i](x)
            x = self.act_fn(x)

        # Final decoding operation
        x = F.conv2d(x, self.decod_kernel.to('cuda'), None, padding='same')
        output = self.decod_fn(x)

        return output



class BaseModel(LightningModule):
    def __init__(
        self, 
        result_dir,
        optimizer,
        lr_scheduler,
        run_type,
        lf_name,
        channnel_number,
        angular_number,
    ):
        super().__init__()
        
        self._result_dir = result_dir
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._run_type = run_type
        self._lf_name = lf_name

        # Create test folder to save rendered images
        self._test_dir = osp.join(
            result_dir, self._lf_name, f'test_c{channnel_number}_a{angular_number}')
        if not osp.exists(self._test_dir):
            os.makedirs(self._test_dir)

    def test_step(self):
        raise NotImplementedError()
    
    def configure_optimizers(self) -> Any:
        if self._optimizer is None:
            optimizer = lambda model: torch.optim.Adam(
                self.model.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-14,
                weight_decay=1e-10
            )
        else:
            optimizer = self._optimizer(self.model)
        if self._lr_scheduler is None:
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1
            )
        else:
            lr_scheduler = self._lr_scheduler(optimizer)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'train/loss'
        }

    
    def _test_save_step(self, img_pred, u, v):
        # Save predicted RGB images for test
        pred_path = osp.join(
            self._test_dir, f"lf_{u:1}_{v:1}.png"
        )
        img = img_pred[0].permute(1,2,0).detach().cpu().numpy()
        img = np.round(img*255)/255.
        plt.imsave(pred_path,img)
        return


class CNetwork(BaseModel):
    def __init__(
            self,
            channel_number: List,
            result_dir: str,
            optimizer: nn.Module,
            lr_scheduler: nn.Module,
            criterion: nn.Module,
            run_type: str,
            angular_list: List,
            dilation_list: List,
            angular_resolution: int,
            img_height: int,
            img_width: int,
            lf_name: str,
            noise_std: float,
            upsampling: str,
            num_base: int,
            num_centroid: int,
            training_epochs: int,
            noise_dir: str,
            data_path: str,
    ):
        super().__init__(
            result_dir=result_dir,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            run_type=run_type,
            lf_name=lf_name,
            channnel_number=channel_number[0],
            angular_number=angular_list[0],
        )
        self.model = CNNModel(
            channel_number=channel_number,
            angular_list=angular_list,
            angular_resolution=angular_resolution,
            img_height=img_height,
            img_width=img_width,
            noise_std=noise_std,
            upsampling=upsampling,
            dilation_list=dilation_list,
            num_base=num_base,
            noise_dir=noise_dir,
        ).to('cuda')

        self._lf_name = lf_name
        self._data_path = data_path
        self._criterion = criterion
        self._num_centroid = num_centroid
        self._angular_resolution = angular_resolution
        self._training_epochs = training_epochs
        self._total_epochs = training_epochs + len(channel_number) + 1



    def test_step(self, batch: Dict[str, Any], batch_idx: torch.Tensor):
        [row, col] = batch['rand_coord']
        rand_pred = self.model(row, col)
        self._test_save_step(rand_pred, row, col)


def create_model(config) -> LightningModule:
    criterion = torch.nn.MSELoss()

    # Initialize model
    optimizer = lambda model: torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=(0.9, 0.99), eps=1e-14,
        weight_decay=1e-9
    )

    lr_scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda iter: 0.15 ** (iter / config.epoch_decay))

    model = CNetwork(
        channel_number=config.channel_number,
        result_dir=config.result_dir,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        run_type=config.run_type,
        angular_list=config.angular_number,
        lf_name=config.lf_name,
        img_height=config.img_height,
        img_width=config.img_width,
        angular_resolution=config.angular_resolution,
        noise_std=config.noise_std,
        upsampling=config.upsampling,
        dilation_list=config.dilation_list,
        num_base=config.num_base,
        num_centroid=config.num_centroid,
        training_epochs=config.training_epochs,
        noise_dir=config.input_dir,
        data_path=config.data_path,
    )
    return model