# -*- coding:utf-8 -*-

import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from util import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      disjoint_train_ratio=0),
])
# print(dataset)
dataset = Planetoid(root_path + '/data', name='Cora', transform=transform)
train_data, val_data, test_data = dataset[0]


def train():
    model = Node2Vec(train_data.edge_index, embedding_dim=128, walk_length=10,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    num_workers = 0
    loader = model.loader(batch_size=256, shuffle=True,
                          num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for epoch in tqdm(range(100)):
        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print('Epoch {:03d} Loss {:.4f}'.format(epoch, total_loss / len(loader)))

    torch.save(model.embedding.weight.data.cpu(),
               root_path + '/models/node2vec_embedding.pt')


def test():
    z = torch.load(root_path + '/models/node2vec_embedding.pt')

    train_src = z[train_data.edge_label_index[0]]
    train_dst = z[train_data.edge_label_index[1]]
    train_x = torch.cat([train_src, train_dst], dim=-1).numpy()
    train_y = train_data.edge_label.cpu().numpy()

    val_src = z[val_data.edge_label_index[0]]
    val_dst = z[val_data.edge_label_index[1]]
    val_x = torch.cat([val_src, val_dst], dim=-1).numpy()
    val_y = val_data.edge_label.cpu().numpy()

    test_src = z[test_data.edge_label_index[0]]
    test_dst = z[test_data.edge_label_index[1]]
    test_x = torch.cat([test_src, test_dst], dim=-1).numpy()
    test_y = test_data.edge_label.cpu().numpy()

    clf = LogisticRegression()
    clf.fit(train_x, train_y)
    score = clf.predict_proba(test_x)[:, -1]

    test_auc = roc_auc_score(test_y, score)
    test_ap = average_precision_score(test_y, score)

    print('test auc:', test_auc)
    print('test ap:', test_ap)


def main():
    train()
    test()


if __name__ == '__main__':
    main()

