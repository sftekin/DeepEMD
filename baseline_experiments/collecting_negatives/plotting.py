import os.path as osp
from PIL import Image
from label_names import label_names
import matplotlib.pyplot as plt
from Models.utils import ensure_path

figures_dir = osp.join("/content", "DeepEMD", "outputs", "figures")
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