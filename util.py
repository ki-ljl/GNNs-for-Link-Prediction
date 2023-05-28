# -*- coding:utf-8 -*-

import copy
import os
import pickle
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from pytorchtools import EarlyStopping

device = torch.device('cuda')


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_metrics(out, edge_label):
    edge_label = edge_label.cpu().numpy()
    out = out.cpu().numpy()
    pred = (out > 0.5).astype(int)
    auc = roc_auc_score(edge_label, out)
    f1 = f1_score(edge_label, pred)
    ap = average_precision_score(edge_label, out)

    return auc, f1, ap


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f, protocol=4)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset


def train_negative_sample(train_data):
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index


@torch.no_grad()
def test(model, val_data, test_data):
    model.eval()
    # cal val loss
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    out = model(val_data,
                val_data.edge_label_index).view(-1)
    val_loss = criterion(out, val_data.edge_label)
    # cal metrics
    out = model(test_data,
                test_data.edge_label_index).view(-1).sigmoid()
    model.train()

    auc, f1, ap = get_metrics(out, test_data.edge_label)

    return val_loss, auc, ap


def train(model, train_data, val_data, test_data, save_model_path):
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    early_stopping = EarlyStopping(patience=50, verbose=True)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    min_epochs = 10
    min_val_loss = np.Inf
    final_test_auc = 0
    final_test_ap = 0
    best_model = None
    model.train()
    for epoch in tqdm(range(100)):
        optimizer.zero_grad()
        edge_label, edge_label_index = train_negative_sample(train_data)
        out = model(train_data, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        # validation
        val_loss, test_auc, test_ap = test(model, val_data, test_data)
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

