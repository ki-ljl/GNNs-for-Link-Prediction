# -*- coding:utf-8 -*-

import os
import sys

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models import SAGE_LP
from util import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False, disjoint_train_ratio=0),
])

dataset = Planetoid(root_path + '/data', name='Cora', transform=transform)
train_data, val_data, test_data = dataset[0]

print(train_data)
print(val_data)
print(test_data)


def main():
    model = SAGE_LP(dataset.num_features, 64, 128).to(device)
    test_auc, test_ap = train(model,
                              train_data,
                              val_data,
                              test_data,
                              save_model_path=root_path + '/models/sage.pkl')
    print('final best auc:', test_auc)
    print('final best ap:', test_ap)


if __name__ == '__main__':
    main()
