import matplotlib.pyplot as plt
from code.ResultSaving import ResultSaving

#---- script used for drawing figures ----
#--------------- DifNet --------------

dataset_name = 'pubmed'

if 0:
    residual_type = 'graph_raw'
    diffusion_type = 'sum'
    depth_list = [1, 2, 3, 4]#, 2, 3, 4, 5, 6, 9, 19, 29, 39, 49]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/DifNet/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = 'DifNet_' + dataset_name + '_' + diffusion_type + '_' + residual_type+'_depth_' + str(depth) + '_iter_' + str(1)
        print(result_obj.result_destination_file_name)
        depth_result_dict[depth] = result_obj.load()

    x = range(1000)

    plt.figure(figsize=(4, 3))
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_acc = [depth_result_dict[depth]['learning_record'][i]['acc_train'] for i in x]
        plt.plot(x, train_acc, label='DifNet(' + str(depth+1) + '-layer)')

    plt.xlim(0, 1000)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right", fontsize='small')
    plt.show()

    plt.figure(figsize=(4, 3))
    for depth in depth_list:
        test_acc = [depth_result_dict[depth]['learning_record'][i]['acc_test'] for i in x]
        plt.plot(x, test_acc, label='DifNet(' + str(depth + 1) + '-layer)')
        best_score[depth] = max(test_acc)

    plt.xlim(0, 1000)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right", fontsize='small')
    plt.show()

    print(best_score)

#--------------- GCN ---------------

if 0:
    residual_type = 'graph_raw'
    depth_list = [2, 5]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/GResNet/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = 'One_DeepGCNResNet_' + dataset_name + '_' + residual_type+'_depth_' + str(depth) + '_iter_' + str(0)
        depth_result_dict[depth] = result_obj.load()

    x = range(1000)

    plt.figure(figsize=(4, 3))
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_acc = [depth_result_dict[depth]['learning_record'][i]['acc_train'] for i in x]
        if residual_type == 'none':
            plt.plot(x, train_acc, label='GCN(' + str(depth) + '-layer)')
        else:
            plt.plot(x, train_acc, label='GResNet(GCN,' + residual_type + ',' + str(depth) + '-layer)')
    plt.xlim(0, 1000)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    #plt.show()

    plt.figure(figsize=(4, 3))
    for depth in depth_list:
        test_acc = [depth_result_dict[depth]['learning_record'][i]['acc_test'] for i in x]
        if residual_type == 'none':
            plt.plot(x, test_acc, label='GCN(' + str(depth) + '-layer)')
        else:
            plt.plot(x, test_acc, label='GResNet(GCN,' + residual_type + ',' + str(depth) + '-layer)')
        best_score[depth] = max(test_acc)
    plt.xlim(0, 1000)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    #plt.show()

    print(best_score)