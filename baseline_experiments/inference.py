"""
Use this script to collect samples from the models (baseline models).
    for set in ["train", "validation", "test"]:
    - Create a support set (5 class, 5 examples)
    - Create a query set (5 examples)
    - Feed each query feed it to the model (5 times)
    - obtain logits for each sample (25 logits)
    - Get top10.
In your plots show the class labels and logits.
"""
import os
import os.path as osp
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
import torch
from torchvision import transforms
from PIL import Image

from Models.models.Network import DeepEMD
from Models.models.baseline_models import Prototype, Matching
from Models.utils import ensure_path, load_model
from plotting import plot_support_set, plot_query_set

DATA_DIR = 'datasets/miniimagenet'
SAVE_DIR = "outputs"

model_dispatcher = {
    "DeepEMD": DeepEMD,
    "Matching": Matching,
    "Prototype": Prototype
}

model_dir_dispatcher = {
    "DeepEMD": "outputs/deepemd_trained_model/miniimagenet/fcn/max_acc.pth",
    "Matching": "outputs/checkpoints/miniimagenet/Matching/1shot-5way/max_acc.pth",
    "Prototype": "outputs/checkpoints/miniimagenet/Prototype/1shot-5way/max_acc.pth"
}


def main(args):
    model_name = args.model_name
    set_names = ["train", "val", "test"]
    num_class, num_samples = 5, 5
    save_path = osp.join(SAVE_DIR, "inference")

    # load model
    model = model_dispatcher[args.model_name](args)
    model_dir = model_dir_dispatcher[args.model_name]
    model = load_model(model, model_dir, mode="cpu")
    model = model.cpu()
    model.eval()

    for set_n in set_names:
        set_path = osp.join(DATA_DIR, "split", f"{set_n}.csv")
        ds_df = pd.read_csv(set_path)

        # select labels
        label_count = len(ds_df["label"].unique())
        rand_label_ints = np.random.randint(label_count, size=num_class)
        label_list = ds_df["label"].unique()[rand_label_ints]
        data_filtered = pd.concat([ds_df.loc[ds_df["label"] == label] for label in label_list])

        # create support set and query set
        support_set = {}
        query_set = {}
        for label in label_list:
            samples = data_filtered.loc[data_filtered["label"] == label, ["filename"]]
            rand_sample_ints = np.random.randint(len(samples), size=num_samples + 1)
            support_ids, query_id = rand_sample_ints[:num_class], rand_sample_ints[-1]
            support_set[label] = samples["filename"].iloc[support_ids].values.tolist()
            query_set[label] = samples["filename"].iloc[query_id]

        # save the meta data
        ensure_path(save_path)
        meta_data_path = osp.join(save_path, f"{set_n}_meta.pkl")
        meta_data = [support_set, query_set]
        with open(meta_data_path, "wb") as f:
            pkl.dump(meta_data, f)

        # create support and query data
        support_data, query_data = [], []
        support_paths, query_paths = [], []
        for label, filenames in support_set.items():
            im_paths = [osp.join(DATA_DIR, "images", file_n) for file_n in filenames]
            query_path = osp.join(DATA_DIR, "images", query_set[label])
            support_data.append(load_image(im_paths))
            query_data.append(load_image([query_path]))
            support_paths.append(im_paths)
            query_paths.append(query_path)
        support_data = torch.stack(support_data, dim=0)
        query_data = torch.stack(query_data, dim=0)
        support_paths = np.array(support_paths).T
        query_paths = np.array(query_paths)

        # plot query and support data
        label_names = list(support_set.keys())
        plot_support_set(support_paths, label_names, set_name=set_n)
        plot_query_set(query_paths, label_names, set_name=set_n)

        print()
        # perform inference




def load_image(batch_path):
    image_size = 84
    transform = transforms.Compose([
        transforms.Resize([92, 92]),
        transforms.CenterCrop(image_size),

        transforms.ToTensor(),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    images = []
    for path in batch_path:
        image = transform(Image.open(path).convert("RGB"))
        images.append(image)
    images = torch.stack(images)
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # My additional arguments
    parser.add_argument('-model_name', type=str, default="Matching", choices=['DeepEMD', 'Prototype', 'Matching'])

    # about task
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('-shot', type=int, default=1)
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')

    # about model
    parser.add_argument('-temperature', type=float, default=12.5)
    parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
    parser.add_argument('-norm', type=str, default='center', choices=['center'])
    parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
    parser.add_argument('-feature_pyramid', type=str, default=None)
    parser.add_argument('-num_patch',type=int,default=9)
    parser.add_argument('-patch_list',type=str,default='2,3')
    parser.add_argument('-patch_ratio',type=float,default=2)
    parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
    parser.add_argument('-sfc_lr', type=float, default=100)
    parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('-sfc_update_step', type=float, default=100)
    parser.add_argument('-sfc_bs', type=int, default=4)
    args = parser.parse_args()

    main(args)
