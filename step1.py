# Import libraries basic
import os
import numpy as np
import hydra
from omegaconf import OmegaConf
from sslearning.data.data_loader import check_file_list
from torchvision import transforms
from torchsummary import summary
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import signal
import time
import sys
from sslearning.pytorchtools import EarlyStopping

import warnings

# Import functions from current directory
from sslearning.models.accNet import SSLNET, Resnet
from sslearning.data.datautils import (
    RandomSwitchAxisTimeSeries,
    RotationAxisTimeSeries,
)
from sslearning.data.data_loader import (
    SSL_dataset,
    subject_collate,
    worker_init_fn,
)

#set random seed
def set_seed(my_seed=0):
    random_seed = my_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed_all(random_seed)


# generate dummy data

# what is h,w=?

# run ssl model with dummy data