from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet


class BaseLineModel(nn.Module):
    def __init__(self, args, mode="meta") -> None:
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet(args)
        self.epsilon = 1e-8

    def forward(self, input, **kwargs):
        pass
        

    def pairwise_distances(self, x: torch.Tensor, y: torch.Tensor, matching_fn: str) -> torch.Tensor:
        '''
        Efficiently calculate pairwise distances (or other similarity scores) between
        two sets of samples.

        # Arguments
            x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
            y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
            matching_fn: Distance metric/similarity score to compute between samples
        '''
        n_x = x.shape[0]
        n_y = y.shape[0]

        distances = None
        if matching_fn == 'l2':
            distances = (
                    x.unsqueeze(1).expand(n_x, n_y, -1) -
                    y.unsqueeze(0).expand(n_x, n_y, -1)
            ).pow(2).sum(dim=2)

        elif matching_fn == 'cosine':
            normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.epsilon)
            normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + self.epsilon)

            expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
            expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

            cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
            distances = 1 - cosine_similarities

        elif matching_fn == 'dot':
            expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
            expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)
            distances = -(expanded_x * expanded_y).sum(dim=2)

        if distances is None:
            raise(ValueError('Unsupported similarity function'))

        return distances


class Prototype(BaseLineModel):
    def __init__(self, args, mode="meta") -> None:
        super().__init__(args, mode)
    
    def forward(self, input, **kwargs):
        if self.mode == "encode":
            return nn.Flatten()(self.encoder(input))
        elif self.mode == "meta":
            support, queries = input
            k, n = kwargs["k"], kwargs["n"]
            prototypes = self.compute_prototypes(support, k, n)

            distances = self.pairwise_distances(queries, prototypes, distance="l2")
            logits = distances.softmax(dim=1)
            return logits
        else:
            raise ValueError('Unknown mode')


    @staticmethod
    def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
        """Compute class prototypes from support samples.

        # Arguments
            support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
                dimension.
            k: int. "k-way" i.e. number of classes in the classification task
            n: int. "n-shot" of the classification task

        # Returns
            class_prototypes: Prototypes aka mean embeddings for each class
        """
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(k, n, -1).mean(dim=1)
        return class_prototypes


class Matching(BaseLineModel):
    def __init__(self, args, mode="meta") -> None:
        super().__init__(args, mode)
    
    def forward(self, input, **kwargs):
        if self.mode == "encode":
            return nn.Flatten()(self.encoder(input))
        elif self.mode == "meta":
            pass

