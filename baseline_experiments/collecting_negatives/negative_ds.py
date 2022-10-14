import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import pandas as pd

class Negatives(Dataset):

    def __init__(self, rule_name, model_name):
        data_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(data_path, rule_name, model_name, "negatives.csv")

        data_df = pd.read_csv(data_path, index_col=0)
        num_batches = data_df.index.max()

        data_batch, label_batch = [], []
        for i in range(1, num_batches + 1):
            batch_df = data_df.loc[data_df.index == i]
            data_batch.append(batch_df["data_path"].values.flatten())
            label_batch.append(batch_df["label"].values.flatten())
    
        self.data =  data_batch # data path of all data
        self.label = label_batch  # label of all data

        image_size = 84
        self.transform = transforms.Compose([
            transforms.Resize([92, 92]),
            transforms.CenterCrop(image_size),

            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)


    def next(self):
        for i in range(len(self.data)):
            label = torch.tensor(self.label[i])
            image = self.load_image(self.data[i])
            yield self.data[i], image, label

    def load_image(self, batch_path):
        images = []
        for path in batch_path:
            image = self.transform(Image.open(path).convert("RGB"))
            images.append(image)
        images = torch.stack(images)
        return images

if __name__ == '__main__':
    from label_names import label_names
    import matplotlib.pyplot as plt

    negatives_set = Negatives(rule_name="threshold", model_name="DeepEMD")

    for path, im, label in negatives_set.next():
        # plot supports
        fig, ax = plt.subplots(1, 5, figsize=(8, 10))
        for i in range(5):
            im = Image.open(path[i]).convert("RGB")
            ax[i].imshow(im)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"{label[i]}")
        plt.savefig("support.png", dpi=200, bbox_inches="tight")

        #plot query
        query_idx = 5
        fig, ax = plt.subplots()
        im = Image.open(path[query_idx]).convert("RGB")
        ax.imshow(im)
        plt.savefig("query.png", dpi=200, bbox_inches="tight")


