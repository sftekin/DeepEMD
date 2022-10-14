import os.path as osp
from PIL import Image
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


        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))

        image_size = 84
        self.transform = transforms.Compose([
            transforms.Resize([92, 92]),
            transforms.CenterCrop(image_size),

            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return path, image, label


if __name__ == '__main__':
    a = Negatives(rule_name="threshold", model_name="DeepEMD")
