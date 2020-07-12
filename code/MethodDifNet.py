'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from code.EvaluateAcc import EvaluateAcc
from code.cellGDU import myGDU


class MethodDifNet(nn.Module):
    data = None
    lr = 0.01
    dropout = 0.5
    weight_decay = 5e-4
    epoch = 1000
    node_dict = None
    learning_record_dict = {}
    residual_type = 'raw'
    diffusion_type = 'sum'

    spy_tag = False

    def __init__(self, gdu_type, graph_sz, x_raw_sz, x_sz, z_sz, h_sz, out_sz, y_sz, depth):
        super(MethodDifNet, self).__init__()

        self.gdu_type = gdu_type

        # --- model dimension configurations ----
        self.graph_sz = graph_sz
        self.x_raw_sz = x_raw_sz
        self.x_sz = x_sz
        self.z_sz = z_sz
        self.h_sz = h_sz
        self.out_sz = out_sz
        self.y_sz = y_sz
        # --- depth ----
        self.depth = depth

        #---- raw input embedding layer ----
        self.raw_x_embedding = nn.Linear(x_raw_sz, x_sz)
        #---- initialization parameter ----
        self.h_init = nn.Linear(x_sz, x_sz)
        #---- hidden state vectors are of the same dimensions -----
        self.gdu_layers = [None] * depth
        for layer in range(depth):
            self.gdu_layers[layer] = myGDU(self.gdu_type, graph_sz, x_sz, z_sz, h_sz, out_sz)
        #---- output fc layer ----
        self.output = myGDU(self.gdu_type, graph_sz, x_sz, z_sz, h_sz, y_sz)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                stdv = 1. / math.sqrt(p.data.size(1))
            else:
                stdv = 1. / math.sqrt(p.data.size(0))
            p.data.uniform_(-stdv, stdv)

    def forward(self, adj, x):
        #---- residual x ----
        x_embed = torch.sigmoid(self.raw_x_embedding(x))
        x_residual = self.graph_residual(adj, x_embed)
        # ---- initialize h, z ----
        if x_residual is None:
            x_residual = torch.zeros(x_embed.shape)
            h = self.h_init(x_embed)
            z = torch.spmm(adj, h)
        else:
            h = self.h_init(x_residual)
            z = torch.spmm(adj, h)
        #---- layer 1 ----
        h = self.gdu_layers[0](x_residual, z, h)
        # ---- other layers ----
        for layer in range(1, len(self.gdu_layers)):
            z = self.diffusion_layer(adj, h)
            h = self.gdu_layers[layer](x_residual, z, h)
        # ---- output ----
        z = self.diffusion_layer(adj, h)
        logits = self.output(x_residual, z, h)
        return F.log_softmax(logits, dim=1)

    # ---- graph residual terms ----

    def graph_residual(self, adj, x):
        if self.residual_type == 'graph_raw':
            return self.sum_diffusion(adj, x)
        elif self.residual_type == 'raw':
            return x
        elif self.residual_type == 'none':
            return None

    # ---- diffusion layers ----

    def diffusion_layer(self, adj, h):
        if self.diffusion_type == 'sum':
            return self.sum_diffusion(adj, h)
        elif self.diffusion_type == 'attention':
            coalesced_adj = adj.coalesce()
            return self.att_diffusion(coalesced_adj, h)

    def att_diffusion(self, adj, h):
        z = h.clone()

        if self.node_dict is None:
            node_dict = {}
            index_list = adj.indices().tolist()
            for index in range(len(adj.values())):
                row = index_list[0][index]
                col = index_list[1][index]
                if row not in node_dict:
                    node_dict[row] = {}
                node_dict[row][col] = 1.0
            self.node_dict = node_dict

        visited_node_dict = {}
        for node in self.node_dict:
            weight_list = []
            h_target = h[node]
            for neighbor in self.node_dict[node]:
                if (neighbor, node) in visited_node_dict:
                    weight = visited_node_dict[(neighbor, node)]
                else:
                    h_neighbor = h[neighbor]
                    weight = (h_target * h_neighbor).sum()
                    visited_node_dict[(node, neighbor)] = weight
                weight_list.append(weight)
            sum = np.sum([math.exp(i) for i in weight_list])
            normalized_weight_list = [math.exp(i)/sum for i in weight_list]
            for neighbor, w in zip(self.node_dict[node], normalized_weight_list):
                z[node] += w*h[neighbor]
        return z


    def sum_diffusion(self, adj, h):
        z = torch.spmm(adj, h)
        z = F.relu(z)
        z = F.dropout(z, self.dropout, training=self.training)
        return z

    #----------------------------------------------------------

    def myparameters(self):
        parameter_list = list(self.parameters())
        for i in range(1, self.depth):
            parameter_list += self.gdu_layers[i].parameters()
        return parameter_list

    def train_model(self, epoch_iter):
        t_begin = time.time()
        optimizer = optim.Adam(self.myparameters(), lr=self.lr, weight_decay=self.weight_decay)
        accuracy = EvaluateAcc('', '')
        for epoch in range(epoch_iter):
            #self.myparameters()
            t_epoch_begin = time.time()
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data['A'], self.data['X'])
            loss_train = F.cross_entropy(output[self.data['idx_train']], self.data['y'][self.data['idx_train']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_train']], 'pred_y': output[self.data['idx_train']].max(1)[1]}
            acc_train = accuracy.evaluate()
            loss_train.backward()
            optimizer.step()


            if self.spy_tag:
                self.eval()
                output = self.forward(self.data['A'], self.data['X'])

                loss_val = F.cross_entropy(output[self.data['idx_val']], self.data['y'][self.data['idx_val']])
                accuracy.data = {'true_y': self.data['y'][self.data['idx_val']], 'pred_y': output[self.data['idx_val']].max(1)[1]}
                acc_val = accuracy.evaluate()

                loss_test = F.cross_entropy(output[self.data['idx_test']], self.data['y'][self.data['idx_test']])
                accuracy.data = {'true_y': self.data['y'][self.data['idx_test']],
                                 'pred_y': output[self.data['idx_test']].max(1)[1]}
                acc_test = accuracy.evaluate()

                self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                                                    'loss_val': loss_val.item(), 'acc_val': acc_val.item(),
                                                    'loss_test': loss_test.item(), 'acc_test': acc_test.item(),
                                                    'time': time.time() - t_epoch_begin}

                if epoch % 50 == 0:
                    print('Epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss_train.item()),
                          'acc_train: {:.4f}'.format(acc_train.item()),
                          'loss_val: {:.4f}'.format(loss_val.item()),
                          'acc_val: {:.4f}'.format(acc_val.item()),
                          'loss_test: {:.4f}'.format(loss_test.item()),
                          'acc_test: {:.4f}'.format(acc_test.item()),
                          'time: {:.4f}s'.format(time.time() - t_epoch_begin))
            else:
                if epoch % 50 == 0:
                    print('Epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss_train.item()),
                          'acc_train: {:.4f}'.format(acc_train.item()),
                          'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin, np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])

    def test_model(self):
        self.eval()
        accuracy = EvaluateAcc()
        output = self.forward(self.data['A'], self.data['X'])

        loss_test = F.cross_entropy(output[self.data['idx_test']], self.data['y'][self.data['idx_test']])
        accuracy.data = {'true_y': self.data['y'][self.data['idx_test']], 'pred_y': output[self.data['idx_test']].max(1)[1]}
        acc_test = accuracy.evaluate()

        loss_train = F.cross_entropy(output[self.data['idx_train']], self.data['y'][self.data['idx_train']])
        accuracy.data = {'true_y': self.data['y'][self.data['idx_train']], 'pred_y': output[self.data['idx_train']].max(1)[1]}
        acc_train = accuracy.evaluate()

        return {'stat': {'test': {'loss': loss_test.item(), 'acc': acc_test}, 'train': {'loss': loss_train.item(), 'acc': acc_train}},
                'true_y': self.data['y'][self.data['idx_test']], 'pred_y': output[self.data['idx_test']].max(1)[1]}, acc_test.item()

    def run(self):
        time_cost, best_test_acc = self.train_model(self.epoch)
        result, test_acc = self.test_model()
        result['stat']['time_cost'] = time_cost
        result['learning_record'] = self.learning_record_dict
        return result, best_test_acc
