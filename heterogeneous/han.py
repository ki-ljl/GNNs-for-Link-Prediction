# -*- coding:utf-8 -*-

import os
import sys

from tqdm import tqdm

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import numpy as np

from pytorchtools import EarlyStopping
from util import get_metrics

import copy

import torch
import torch_geometric.transforms as T
from torch import nn
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HANConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
])

dataset = DBLP(root_path + '/data/DBLP', transform=transform)
# print(dataset)
graph = dataset[0]
print(graph)

graph['conference'].x = torch.randn((graph['conference'].num_nodes, 128))
graph = graph.to(device)

node_types, edge_types = graph.metadata()
homogeneous_graph = graph.to_homogeneous()
in_feats, hidden_feats = 128, 64

# edge_types = [('paper', 'to', 'paper'), ('paper', 'to', 'author'),
#               ('paper', 'to', 'term'), ('paper', 'to', 'conference')],
# rev_edge_types = [('paper', 'to', 'paper'), ('author', 'to', 'paper'),
#                   ('term', 'to', 'paper'), ('conference', 'to', 'paper')]

graph = T.ToUndirected()(graph)
# print(graph)
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    edge_types=[('paper', 'to', 'author')],
    rev_edge_types=[('author', 'to', 'paper')]
)(graph)

print(train_data)
print(val_data)
print(test_data)


class HAN_LP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HAN_LP, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_channels, graph.metadata(), heads=4)
        self.conv2 = HANConv(hidden_channels, out_channels, graph.metadata(), heads=4)

    def encode(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x = self.conv1(x_dict, edge_index_dict)
        x = self.conv2(x, edge_index_dict)
        return x

    def decode(self, z, edge_label_index):
        src = z['paper'][edge_label_index[0]]
        dst = z['author'][edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)


@torch.no_grad()
def test(model):
    model.eval()
    # cal val loss
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    out = model(val_data,
                val_data['paper', 'to', 'author'].edge_label_index).view(-1)
    val_loss = criterion(out, val_data['paper', 'to', 'author'].edge_label)
    # cal metrics
    out = model(test_data,
                test_data['paper', 'to', 'author'].edge_label_index).view(-1).sigmoid()
    model.train()

    auc, f1, ap = get_metrics(out, test_data['paper', 'to', 'author'].edge_label)

    return val_loss, auc, ap


def train():
    save_model_path = root_path + '/models/han.pkl'
    model = HAN_LP(-1, hidden_feats, 128).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    early_stopping = EarlyStopping(patience=50, verbose=True)
    min_epochs = 10
    min_val_loss = np.Inf
    best_model = None
    final_test_auc = 0
    final_test_ap = 0
    model.train()
    for epoch in tqdm(range(100)):
        optimizer.zero_grad()
        edge_label_index = train_data['paper', 'to', 'author'].edge_label_index
        edge_label = train_data['paper', 'to', 'author'].edge_label
        out = model(train_data, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        # validation
        val_loss, test_auc, test_ap = test(model)
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            final_test_auc = test_auc
            final_test_ap = test_ap
            best_model = copy.deepcopy(model)
            # save model
            state = {'model': best_model.state_dict()}
            torch.save(state, save_model_path)

        # scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('epoch {:03d} train_loss {:.8f} val_loss {:.4f} test_auc {:.4f} test_ap {:.4f}'
              .format(epoch, loss.item(), val_loss, test_auc, test_ap))

    state = {'model': best_model.state_dict()}
    torch.save(state, save_model_path)

    return final_test_auc, final_test_ap


def main():
    test_auc, test_ap = train()
    print('final best auc:', test_auc)
    print('final best ap:', test_ap)


if __name__ == '__main__':
    main()
