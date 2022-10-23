import os.path as osp
from PIL import Image
from collecting_negatives.label_names import label_names
import matplotlib.pyplot as plt
from Models.utils import ensure_path

figures_dir = osp.join("outputs", "figures")
ensure_path(figures_dir)


def plot_comparison(batch_idx, batch_path, model1_logits, model2_logits, query_ind):
    fig, ax = plt.subplots(1, 6, figsize=(8, 10))
    for i in range(5):
        im = Image.open(batch_path[i]).convert("RGB")
        ax[i].imshow(im)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    for k, idx in enumerate(query_ind):
        im = Image.open(batch_path[idx]).convert("RGB")
        ax[5].imshow(im)
        ax[5].set_title("Query")
        ax[5].set_xticks([])
        ax[5].set_yticks([])
        for i in range(5):
            ax[i].set_title(f"M1:{model1_logits[idx][i]:.2f}\nM2:{model2_logits[idx][i]:.2f}")

        fig_path = osp.join(figures_dir, f"comparision_{batch_idx}_{k}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")


def plot_batch(support_path, qurey_paths, logits, mode="goods", filename="emd"):
    fig, ax = plt.subplots(1, 6, figsize=(8, 10))
    for i in range(5):
        im = Image.open(support_path[i]).convert("RGB")
        ax[i].imshow(im)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    for k, path in enumerate(qurey_paths):
        im = Image.open(path).convert("RGB")
        ax[5].imshow(im)
        ax[5].set_title("Query")
        ax[5].set_xticks([])
        ax[5].set_yticks([])
        for i in range(5):
            ax[i].set_title(f"logit:{logits[k, i]:.2f}")

        fig_path = osp.join(figures_dir, f"{mode}_{filename}_{k}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")


def plot_support_set(support_paths, labels_str, set_name="train"):
    m, n = support_paths.shape
    fig, ax = plt.subplots(m, n, figsize=(10, 10))
    for i in range(m):
        for j in range(n):
            im = Image.open(support_paths[i, j]).convert("RGB")
            ax[i, j].imshow(im)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(labels_str[j])
    fig_path = osp.join(figures_dir, f"{set_name}_support.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")


def plot_query_set(query_paths, labels_str, set_name="train"):
    fig, ax = plt.subplots(1, len(query_paths), figsize=(8, 10))
    for i in range(len(query_paths)):
        im = Image.open(query_paths[i]).convert("RGB")
        ax[i].imshow(im)
        ax[i].set_title(labels_str[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    fig_path = osp.join(figures_dir, f"{set_name}_query.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
