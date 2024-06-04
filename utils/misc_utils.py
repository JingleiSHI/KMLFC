import os.path
from torch.autograd import Variable
import torch
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def check_input(config):
    # Basic parameters
    _channel_number = config.channel_number
    _img_width = config.img_width
    _img_height = config.img_height
    totalupsample = 2 ** (len(_channel_number) - 1)
    width = int(_img_width / totalupsample)
    height = int(_img_height / totalupsample)
    ###################################################
    noise_folder = config.input_dir
    if not os.path.exists(noise_folder):
        os.makedirs(noise_folder)
    input_path = noise_folder + '/' + 'input_' + str(_channel_number[0]) + '_' + str(height) + '_' + str(width) + '.npy'
    if not os.path.exists(input_path):
        shape = [1, _channel_number[0], height, width]
        noise = Variable(torch.zeros(shape))
        seed_everything(config.seed)
        noise.data.uniform_()
        noise.data *= 1. / 10
        np.save(input_path, noise.cpu().data.numpy())