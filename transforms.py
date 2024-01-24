import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from mechanisms import supported_feature_mechanisms


class FeatureTransform:
    supported_features = ['raw', 'rnd', 'one', 'ohd']

    def __init__(self, feature: dict(help='feature transformation method',
                                     choices=supported_features, option='-f') = 'raw'):

        self.feature = feature

    def __call__(self, data):

        if self.feature == 'rnd':
            data.x = torch.rand_like(data.x)
        elif self.feature == 'ohd':
            data = OneHotDegree(max_degree=data.num_features - 1)(data)
        elif self.feature == 'one':
            data.x = torch.ones_like(data.x)

        return data


class FeaturePerturbation:
    def __init__(self,
                 mechanism: dict(help='feature perturbation mechanism', choices=list(supported_feature_mechanisms),
                                 option='-m') = 'mbm',
                 x_eps: dict(help='privacy budget for feature perturbation', type=float,
                             option='-ex') = np.inf,
                 data_range=None):

        self.mechanism = mechanism
        self.input_range = data_range
        self.x_eps = x_eps

    def __call__(self, data):
        if np.isinf(self.x_eps):
            return data

        if self.input_range is None:
            self.input_range = data.x.min().item(), data.x.max().item()

        data.x, user_levels_perturb = supported_feature_mechanisms[self.mechanism](
            eps=self.x_eps,
            input_range=self.input_range
        )(data.x)

        return data, user_levels_perturb


class OneHotDegree:
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, data):
        degree = data.adj_t.sum(dim=0).long()
        degree.clamp_(max=self.max_degree)
        data.x = F.one_hot(degree, num_classes=self.max_degree + 1).float()  # add 1 for zero degree
        return data


class Normalize:
    def __init__(self, low, high):
        self.min = low
        self.max = high

    def __call__(self, data):
        alpha = data.x.min(dim=0)[0]
        beta = data.x.max(dim=0)[0]
        delta = beta - alpha
        data.x = (data.x - alpha) * (self.max - self.min) / delta + self.min
        data.x = data.x[:, torch.nonzero(delta, as_tuple=False).squeeze()]  # remove features with delta = 0
        return data


class FilterTopClass:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        y = torch.nn.functional.one_hot(data.y)
        c = y.sum(dim=0).sort(descending=True)
        y = y[:, c.indices[:self.num_classes]]
        idx = y.sum(dim=1).bool()

        data.x = data.x[idx]
        data.y = y[idx].argmax(dim=1)
        data.num_nodes = data.y.size(0)

        if 'adj_t' in data:
            data.adj_t = data.adj_t[idx, idx]
        elif 'edge_index' in data:
            data.edge_index, data.edge_attr = subgraph(idx, data.edge_index, data.edge_attr, relabel_nodes=True)

        if 'train_mask' in data:
            data.train_mask = data.train_mask[idx]
            data.val_mask = data.val_mask[idx]
            data.test_mask = data.test_mask[idx]

        return data
