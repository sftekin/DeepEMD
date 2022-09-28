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
    #transform str parameter into list
    if args.feature_pyramid is not None:
        args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
    args.patch_list = [int(x) for x in args.patch_list.split(',')]

    set_seed(args.seed)
    num_gpu = set_gpu(args)
    Dataset=set_up_datasets(args)

    # model
    args.pretrain_dir = osp.join(args.pretrain_dir,'%s/resnet12/max_acc.pth'%(args.dataset))
    model = model_dispatcher[args.model_name](args)
    model = load_model(model, args.pretrain_dir)
    model = nn.DataParallel(model, list(range(num_gpu)))
    model = model.cuda()
    model.eval()

    print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # My additional arguments
    parser.add_argument('-model_name', type=str, default="Prototype", choices=['DeepEMD', 'Prototype', 'Matching'])

    #about dataset and training
    parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'cub','tieredimagenet','fc100','tieredimagenet_yao','cifar_fs'])
    parser.add_argument('-data_dir', type=str, default=DATA_DIR,help='dir of datasets')
    parser.add_argument('-set',type=str,default='val',choices=['test','val'],help='the set used for validation')# set used for validation
    #about training
    parser.add_argument('-bs', type=int, default=1,help='batch size of tasks')
    parser.add_argument('-max_epoch', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-temperature', type=float, default=12.5)
    parser.add_argument('-step_size', type=int, default=10)
    parser.add_argument('-gamma', type=float, default=0.5)
    parser.add_argument('-val_frequency',type=int,default=50)
    parser.add_argument('-random_val_task',action='store_true',help='random samples tasks for validation at each epoch')
    parser.add_argument('-save_all',action='store_true',help='save models on each epoch')
    #about task
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('-shot', type=int, default=1)
    parser.add_argument('-query', type=int, default=15,help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=1000, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=5000, help='number of testing episodes after training')
    # about model
    parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
    parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
    parser.add_argument('-norm', type=str, default='center', choices=['center'], help='feature normalization')
    parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
    #deepemd fcn only
    parser.add_argument('-feature_pyramid', type=str, default=None, help='you can set it like: 2,3')
    #deepemd sampling only
    parser.add_argument('-num_patch',type=int,default=9)
    #deepemd grid only patch_list
    parser.add_argument('-patch_list',type=str,default='2,3',help='the size of grids at every image-pyramid level')
    parser.add_argument('-patch_ratio',type=float,default=2,help='scale the patch to incorporate context around the patch')
    # slvoer about
    parser.add_argument('-solver', type=str, default='opencv', choices=['opencv', 'qpth'])
    parser.add_argument('-form', type=str, default='L2', choices=['QP', 'L2'])
    parser.add_argument('-l2_strength', type=float, default=0.000001)
    # SFC
    parser.add_argument('-sfc_lr', type=float, default=0.1, help='learning rate of SFC')
    parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('-sfc_update_step', type=float, default=100, help='number of updating step of SFC')
    parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')

    # OTHERS
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')
    parser.add_argument('-seed', type=int, default=1)

    args = parser.parse_args()

    main(args)
