from DatasetLoader import DatasetLoader
from MethodDifNet import MethodDifNet
from ResultSaving import ResultSaving
from SettingCV import SettingCV
from EvaluateAcc import EvaluateAcc
import numpy as np
import torch
import random
import os

#--------------------------------------------------------

#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'cora'

if dataset_name == 'cora-small':
    nclass = 7
    nfeature = 1433
    ngraph = 10
elif dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
    ngraph = 2708
elif dataset_name == 'citeseer':
    nclass = 6
    nfeature = 3703
    ngraph = 3312
elif dataset_name == 'pubmed':
    nclass = 3
    nfeature = 500
    ngraph = 19717

#---- Deep DifNet method ----
if 1:
    for depth in [1]:
        #---- parameter section -------------------------------

        epoch = 1000
        dropout = 0.5
        lr = 0.01
        weight_decay = 5e-4

        graph_sz = ngraph
        x_raw_sz = nfeature
        x_sz = z_sz = h_sz = out_sz = 32
        y_sz = nclass

        # --- diffusion type incldue: sum, attention ---
        diffusion_type = 'sum'
        # --- residual type include: raw, graph_raw ---
        residual_type = 'graph_raw'
        # --- gdu type include: original, simplified, extreme, gcn_gat ---
        gdu_type = 'original'
        #------------------------------------------------------

        # ---- objection initialization setction ---------------
        print('Method: DifNet, dataset: ' + dataset_name + ', depth: ' + str(depth) + ', residual type: ' + residual_type + ' , diffusion type: ' + diffusion_type + ', gdu type: ' + gdu_type)
        print('Start')

        data_obj = DatasetLoader('', '')
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.method_type = 'DifNet'

        method_obj = MethodDifNet(gdu_type, graph_sz, x_raw_sz, x_sz, z_sz, h_sz, out_sz, y_sz, depth)
        method_obj.epoch = epoch
        method_obj.dropout = dropout
        method_obj.lr = lr
        method_obj.weight_decay = weight_decay
        method_obj.residual_type = residual_type
        method_obj.diffusion_type = diffusion_type


        result_obj = ResultSaving('', '')
        result_obj.result_destination_folder_path = './result/DifNet/'
        result_obj.result_destination_file_name = 'DifNet_' + dataset_name + '_' + diffusion_type + '_' + residual_type + '_depth_' + str(depth) + '_iter_' + str(iter)

        setting_obj = SettingCV('', '')

        evaluate_obj = EvaluateAcc('', '')
        #------------------------------------------------------

        #---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        acc_test = setting_obj.load_run_save_evaluate()
        print('*******************************')
        #------------------------------------------------------
    print('Test set results: accuracy = ' + str(acc_test))
    print('Finished')


#--------------------------------------------------------

