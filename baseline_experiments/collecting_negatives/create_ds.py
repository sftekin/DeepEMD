import argparse
from imp import load_compiled
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
import pickle as pkl
import os
from Models.dataloader.data_utils import *
import pandas as pd
import numpy as np
from plotting import plot_comparison


DATA_DIR='/content/DeepEMD/datasets'

model_dispatcher = {
    "DeepEMD": DeepEMD,
    "Matching": Matching,
    "Prototype": Prototype
}

model_dir_dispatcher = {
    "DeepEMD": "/content/DeepEMD/outputs/deepemd_trained_model/miniimagenet/fcn/max_acc.pth",
    "Matching": "/content/drive/MyDrive/repo_dumps/DeepEMD/checkpoints/miniimagenet/Matching/1shot-5way/max_acc.pth",
    "Prototype": "/content/drive/MyDrive/repo_dumps/DeepEMD/checkpoints/miniimagenet/Prototype/1shot-5way/max_acc.pth"
}

def main(args):
    if args.feature_pyramid is not None:
        args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
    args.patch_list = [int(x) for x in args.patch_list.split(',')]

    pprint(vars(args))
    set_seed(args.seed)
    num_gpu = set_gpu(args)
    Dataset=set_up_datasets(args)

    # model
    model = model_dispatcher[args.model_name](args)
    model_dir = model_dir_dispatcher[args.model_name]
    model = load_model(model, model_dir)
    model = nn.DataParallel(model, list(range(num_gpu)))
    model = model.cuda()
    model.eval()

    # test dataset
    test_set = Dataset(args.set, args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=2, pin_memory=True)
    tqdm_gen = tqdm.tqdm(loader)

    # label of query images
    ave_acc = Averager()
    test_acc_record = np.zeros((args.test_episode,))
    label = torch.arange(args.way).repeat(args.query)
    label = label.type(torch.cuda.LongTensor)

    # set the constants for the negative
    negative_data = []
    negative_label = []

    # set the rules for the negatives
    if args.rule == "compare":
        comparison_model = Matching(args)
        comparison_model = load_model(comparison_model, model_dir_dispatcher['Matching'])
        comparison_model = nn.DataParallel(comparison_model, list(range(num_gpu)))
        comparison_model = comparison_model.cuda()
        comparison_model.eval()

    closeness_set = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm_gen, 1):

            path_batch, data_batch, label_batch = batch
            data, label_batch = data_batch.cuda(), label_batch.cuda()
            # data, _ = [_.cuda() for _ in batch]

            acc, logits = model_step(data, label, model, args, num_gpu)
            pred = torch.argmax(logits, dim=1)

            if args.rule == "threshold":
                if acc <= float(args.threshold) * 100:
                    negative_data += path_batch
                    negative_label += label_batch.cpu().numpy().tolist()
            elif args.rule == "compare":
                acc_2, logits_2 = model_step(data, label, comparison_model, args, num_gpu)
                pred2 = torch.argmax(logits_2, dim=1)
                if acc < acc_2:
                    negative_data += path_batch
                    negative_label += label_batch.cpu().numpy().tolist()
                    # plot the bests of compare
                    mistakes_emd = pred != label
                    truths_model = pred2 == label
                    query_ind = np.where(np.logical_and(mistakes_emd.cpu(), truths_model.cpu()))[0]
                    plot_comparison(i, path_batch, logits, logits_2, query_ind)
            elif args.rule == "goods":
                pos_logits = logits[pred == label]
                top_k = torch.topk(pos_logits, k=2, dim=1)[0]
                closeness = top_k[:, 0] - top_k[:, 1]
                closeness_set += closeness.cpu().numpy().tolist()
            else:
                support_path = path_batch[:5]
                query_path = path_batch[5:]
                query_path = np.array(query_path)[pred == label]
                label_ind = np.array(label_batch[5:])[pred == label]
                negative_data.append(support_path + query_path)
                negative_label.append(label_batch[:5] + label_ind)

            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            m, pm = compute_confidence_interval(test_acc_record[:i])
            tqdm_gen.set_description('batch {}: This episode:{:.2f}  average: {:.4f}+{:.4f}'.format(i, acc, m, pm))

        dir_path = os.path.dirname(os.path.abspath(__file__))
        negative_path = os.path.join(dir_path, f"{args.rule}", f"{args.model_name}")
        ensure_path(negative_path)

        batch_inds = []
        batch_count = 0
        neg_dir = {"data_path": [], "label": []}
        for i in range(len(negative_data)):
            if isinstance(negative_data[i], list):
                batch_count += 1
                batch_inds += [batch_count for _ in range(len(negative_data[i]))]
                neg_dir["data_path"] += negative_data[i]
                neg_dir["label"] += negative_label[i]
            else:
                batch_count = batch_count + 1 if i % 80 == 0 else batch_count
                batch_inds.append(batch_count)

        negatives_df = pd.DataFrame({"data_path": negative_data, "label": negative_label}, index=batch_inds)
        negatives_df.to_csv(os.path.join(negative_path, "negatives.csv"))


        m, pm = compute_confidence_interval(test_acc_record)
        result_list = ['test Acc {:.4f}'.format(ave_acc.item())]
        result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print(result_list[0])
        print(result_list[1])


def model_step(data, label, model, args, num_gpu):
    k = args.way * args.shot
    model.module.mode = 'encoder'
    data = model(data)
    data_shot, data_query = data[:k], data[k:]  # shot: 5,3,84,84  query:75,3,84,84
    model.module.mode = 'meta'
    if args.shot > 1:
        data_shot = model.module.get_sfc(data_shot)
    # logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
    logits = model((data_shot, data_query))
    acc = count_acc(logits, label) * 100
    return acc, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # set the rule
    parser.add_argument('-model_name', type=str, default="DeepEMD", choices=['DeepEMD', 'Prototype', 'Matching'])
    parser.add_argument("-rule", type=str, default="goods", choices=["threshold", "compare", "fail", "goods"])
    parser.add_argument("-threshold", type=float, default=0.63)
    # about task
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('-shot', type=int, default=1)
    parser.add_argument('-query', type=int, default=15)  # number of query image per class
    parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'cub','tieredimagenet','fc100','tieredimagenet_yao','cifar_fs'])
    parser.add_argument('-set', type=str, default='test', choices=['train','val', 'test'])
    # about model
    parser.add_argument('-temperature', type=float, default=12.5)
    parser.add_argument('-metric', type=str, default='cosine', choices=[ 'cosine' ])
    parser.add_argument('-norm', type=str, default='center', choices=[ 'center'])
    parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
    #deepemd fcn only
    parser.add_argument('-feature_pyramid', type=str, default=None)
    #deepemd sampling only
    parser.add_argument('-num_patch',type=int,default=9)
    #deepemd grid only patch_list
    parser.add_argument('-patch_list',type=str,default='2,3')
    parser.add_argument('-patch_ratio',type=float,default=2)
    # solver
    parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
    # SFC
    parser.add_argument('-sfc_lr', type=float, default=100)
    parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('-sfc_update_step', type=float, default=100)
    parser.add_argument('-sfc_bs', type=int, default=4)
    # others
    parser.add_argument('-test_episode', type=int, default=5000)
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-data_dir', type=str, default=DATA_DIR)
    parser.add_argument('-seed', type=int, default=1)

    args = parser.parse_args()
    main(args)
