import os.path as osp
from PIL import Image
from label_names import label_names
import matplotlib.pyplot as plt
from Models.utils import ensure_path


def plot_support_query(batch_path, batch_label, query_idx=5):
    figures_dir = osp.join("/content", "DeepEMD", "outputs", "figures")
    ensure_path(figures_dir)

    fig, ax = plt.subplots(1, 5, figsize=(8, 10))
    for i in range(5):
        im = Image.open(batch_path[i]).convert("RGB")
        ax[i].imshow(im)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f"{batch_label[i]}")
    fig_path = osp.join(figures_dir, "support.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")

    #plot query
    fig, ax = plt.subplots()
    im = Image.open(batch_path[query_idx]).convert("RGB")
    ax.imshow(im)
    ax.set_title(batch_label[query_idx])
    fig_path = osp.join(figures_dir, "query.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
