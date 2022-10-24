import os.path as osp
from PIL import Image
from collecting_negatives.label_names import label_names
import matplotlib.pyplot as plt
from Models.utils import ensure_path
import numpy as np

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


def plot_top_k(logits, paths, k=10, set_name="train"):
    sorted_idx = np.argsort(logits, axis=1)[:, ::-1]
    paths = paths.flatten()

    num_query = sorted_idx.shape[0]
    fig, ax = plt.subplots(k, num_query, figsize=(15, 15))
    for i in range(num_query):
        im_logits = logits[i, sorted_idx[i, :k]]
        im_paths = paths[sorted_idx[i, :k]]
        for j in range(k):
            ax[j, i].imshow(Image.open(im_paths[j]).convert("RGB"))
            ax[j, i].set_title(f"{im_logits[j]:.4f}")
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])

    fig_path = osp.join(figures_dir, f"{set_name}_top{k}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")


def plot_episodes(support_paths, query_paths, ep_er, ep_name, logits, set_name="train"):
    ep_dir = osp.join(figures_dir, f"{set_name}_{ep_name}_episodes")
    ensure_path(ep_dir)

    counter = 0
    way_count = support_paths.shape[1]
    support_ids = np.where(ep_er)[0] // way_count
    query_ids = np.where(ep_er)[0] % way_count
    fig, ax = plt.subplots(1, way_count + 1, figsize=(12, 12))
    for sup_idx, query_idx in zip(support_ids, query_ids):
        query_path = query_paths[query_idx]
        sup_paths = support_paths[sup_idx]
        ax[0].imshow(Image.open(query_path).convert("RGB"))
        ax[0].set_title(f"Query")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        title_set = []
        for model_name, logit_arr in logits.items():
            logit_set = logit_arr[query_idx, sup_idx*way_count:]
            title_set.append([f"{model_name}:{l:.3f}" for l in logit_set])

        for j, path in enumerate(sup_paths):
            title_str = '\n'.join([t_set[j] for t_set in title_set])
            ax[j+1].imshow(Image.open(path).convert("RGB"))
            ax[j+1].set_title(title_str, fontsize=8)
            ax[j+1].set_xticks([])
            ax[j+1].set_yticks([])

        fig_path = osp.join(ep_dir, f"{counter}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        counter += 1
