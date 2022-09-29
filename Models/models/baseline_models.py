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

        self.k = args.way
        self.n = args.shot
        self.q = args.query

        self.encoder = ResNet(args)
        self.epsilon = 1e-8

    def forward(self, input):
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
    
    def forward(self, input):
        if self.mode == "encoder":
            return nn.Flatten()(self.encoder(input))
        elif self.mode == "meta":
            support, queries = input
            prototypes = self.compute_prototypes(support)

            distances = self.pairwise_distances(queries, prototypes, matching_fn="l2")
            logits = (-distances).softmax(dim=1)
            return logits
        else:
            raise ValueError('Unknown mode')

    def compute_prototypes(self, support):
        """Compute class prototypes from support samples."""
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(self.k, self.n, -1).mean(dim=1)
        return class_prototypes


class Matching(BaseLineModel):
    def __init__(self, args, mode="meta") -> None:
        super().__init__(args, mode)
    
    def forward(self, input, **kwargs):
        if self.mode == "encoder":
            return nn.Flatten()(self.encoder(input))
        elif self.mode == "meta":
            support, queries = input

            distances = self.pairwise_distances(queries, support, matching_fn="cosine")
            attention = (-distances).softmax(dim=1)

            y_pred = self.matching_net_predictions(attention)

            # Calculated loss with negative log likelihood
            # Clip predictions for numerical stability
            clipped_y_pred = y_pred.clamp(self.epsilon, 1 - self.epsilon)
            
            return clipped_y_pred
    
    def matching_net_predictions(self, attention):
        """
        Calculates Matching Network predictions based on equation (1) of the paper.
        The predictions are the weighted sum of the labels of the support set where the
        weights are the "attentions" (i.e. softmax over query-support distances) pointing
        from the query set samples to the support set samples.
        """
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        if attention.shape != (self.q * self.k, self.k * self.n):
            raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) \
             = ({self.q * self.k, self.k * self.n})'))

        # Create one hot label vector for the support set
        y_onehot = torch.zeros(self.k * self.n, self.k)

        # Unsqueeze to force y to be of shape (K*n, 1) as this
        # is needed for .scatter()
        y = self.create_nshot_task_label(self.k, self.n).unsqueeze(-1)
        y_onehot = y_onehot.scatter(1, y, 1)

        y_pred = torch.mm(attention, y_onehot.cuda().float())

        return y_pred

    @staticmethod
    def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
        """Creates an n-shot task label.

        Label has the structure:
            [0]*q + [1]*q + ... + [k-1]*q

        # TODO: Test this

        # Arguments
            k: Number of classes in the n-shot classification task
            q: Number of query samples for each class in the n-shot classification task

        # Returns
            y: Label vector for n-shot task of shape [q * k, ]
        """
        y = torch.arange(0, k, 1 / q).long()
        return y


class Finetune(BaseLineModel):
    def __init__(self, args, mode="meta") -> None:
        super().__init__(args, mode)

        self.fine_tuning_steps = args.fine_tuning_steps
        self.fine_tuning_lr = args.fine_tuning_lr

    def forward(self, input):
        if self.mode == "encoder":
            return nn.Flatten()(self.encoder(input))
        elif self.mode == "meta":
            support, queries = input
            
