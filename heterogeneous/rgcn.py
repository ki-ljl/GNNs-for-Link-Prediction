# -*- coding:utf-8 -*-

import os
import sys

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

from util import train
import torch
import torch_geometric.transforms as T
from torch import nn
from torch_geometric.datasets import DBLP
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
])

dataset = DBLP(root_path + '/data/DBLP', transform=transform)
graph = dataset[0]
print(graph)

num_classes = torch.max(graph['author'].y).item() + 1
graph['conference'].x = torch.randn((graph['conference'].num_nodes, 128))
graph = graph.to(device)

node_types, edge_types = graph.metadata()
num_nodes = graph['author'].x.shape[0]
num_relations = len(edge_types)
init_sizes = [graph[node_type].x.shape[1] for node_type in node_types]
init_x = [graph[node_type].x for node_type in node_types]

homogeneous_graph = graph.to_homogeneous()
train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        disjoint_train_ratio=0,
        edge_types=[('author', 'to', 'paper'), ('paper', 'to', 'term'),
                    ('paper', 'to', 'conference')],
        rev_edge_types=[('paper', 'to', 'author'), ('term', 'to', 'paper'),
                        ('conference', 'to', 'paper')]
    )(homogeneous_graph)

print(train_data)
print(val_data)
print(test_data)


class RGCN_LP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RGCN_LP, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels,
                              num_relations=num_relations, num_bases=30)
        self.conv2 = RGCNConv(hidden_channels, out_channels,
                              num_relations=num_relations, num_bases=30)
        self.lins = torch.nn.ModuleList()
        self.norm = nn.LayerNorm(in_channels)
        for i in range(len(node_types)):
            lin = nn.Linear(init_sizes[i], in_channels)
            self.lins.append(lin)

        self.fc = nn.Sequential(
            nn.Linear(2 * out_channels, 1)
        )

    def trans_dimensions(self, xs):
        res = []
        for x, lin in zip(xs, self.lins):
            res.append(lin(x))
        return torch.cat(res, dim=0)

    def encode(self, data):
        x = self.trans_dimensions(init_x)
        edge_index, edge_type = data.edge_index, data.edge_type
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)

        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        x = torch.cat([src, dst], dim=-1)
        x = self.fc(x)

        return x

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)


def main():
    model = RGCN_LP(128, 64, 128).to(device)
    test_auc, test_ap = train(model,
                              train_data,
                              val_data,
                              test_data,
                              save_model_path=root_path + '/models/rgcn.pkl')
    print('final best auc:', test_auc)
    print('final best ap:', test_ap)


if __name__ == '__main__':
    main()
