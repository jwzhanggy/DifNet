from code.DatasetLoader import DatasetLoader
from code.MethodDifNet import MethodDifNet
from code.ResultSaving import ResultSaving
from code.SettingCV import SettingCV
from code.EvaluateAcc import EvaluateAcc

#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'cora'


#---- Deep DifNet method ----
if 1:
    if dataset_name == 'cora-small':
        nclass = 7
        nfeature = 1433
        ngraph = 10
    elif dataset_name == 'cora':
        nclass = 7
        nfeature = 1433
        ngraph = 2708

    # ---- depth = 1 actually involves 2 GDU layers (output contains 1 GDU layer already) ----
    for depth in [1]:
        #---- parameter section -------------------------------
        iter = 1
        epoch = 1000
        dropout = 0.5
        lr = 0.005
        weight_decay = 5e-4

        graph_sz = ngraph
        x_raw_sz = nfeature
        x_sz = z_sz = h_sz = out_sz = 16
        y_sz = nclass

        # --- diffusion type incldue: sum, attention ---
        diffusion_type = 'sum'
        # --- residual type include: raw, graph_raw ---
        residual_type = 'graph_raw'
        # --- gdu type include: original, short_gate, single_gate, recombine, extreme_simplified ---
        gdu_type = 'original'
        #------------------------------------------------------

        # ---- objection initialization setction ---------------
        print('Method: DifNet, dataset: ' + dataset_name + ', depth: ' + str(depth) + ', residual type: ' + residual_type + ' , diffusion type: ' + diffusion_type + ', gdu type: ' + gdu_type)
        print('Start')

        data_obj = DatasetLoader('', '')
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.method_type = 'DifNet'
        data_obj.dataset_name = dataset_name

        method_obj = MethodDifNet(gdu_type, graph_sz, x_raw_sz, x_sz, z_sz, h_sz, out_sz, y_sz, depth)
        method_obj.epoch = epoch
        method_obj.dropout = dropout
        method_obj.lr = lr
        method_obj.weight_decay = weight_decay
        method_obj.residual_type = residual_type
        method_obj.diffusion_type = diffusion_type
        method_obj.spy_tag = True


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

