import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.dataloader.data_utils import *
from Models.models.Network import DeepEMD
from Models.models.baseline_models import Prototype, Matching
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time


PRETRAIN_DIR='/content/DeepEMD/outputs/deepemd_pretrain_model'
DATA_DIR='/content/DeepEMD/datasets'

model_dispatcher = {
    "DeepEMD": DeepEMD,
    "Matching": Matching,
    "Prototype": Prototype
}

def main(args):
    pass


if __name__ == "__main__":
    pass

