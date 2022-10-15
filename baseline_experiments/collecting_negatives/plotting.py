from msvcrt import kbhit
import os.path as osp
from PIL import Image
from label_names import label_names
import matplotlib.pyplot as plt
from Models.utils import ensure_path


def plot_comparison(batch_idx, batch_path, model1_logits, model2_logits, query_ind):
    figures_dir = osp.join("/content", "DeepEMD", "outputs", "figures")
    ensure_path(figures_dir)

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

def plot_batch():
    pass
