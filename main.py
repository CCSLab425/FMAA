"""
Author:  Qi tong Chen
Date: 2024.08.19
Main function.
"""

import argparse
import torch
import dataloader
from dataloader import data_reader
from models.Lightweight_res_net import Lightweight_model, Dmodel, Domain_model
from MMD.MMD_calculation import MMDLoss
# from ConfusionMatrix.Confusion_Matrix import DrawConfusionMatrix
from files_path import files_path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils_dsbn.save_and_other_functions import save_model, save_his, save_predict_labels, \
    train_history
from train_process_module.pretrain_feature_extractor import pretrain_process_fn
from train_process_module.pretrain_discriminator import pretrain_discriminator_fn
from train_process_module.adv_train_process import adversarial_process
import os
import pandas as pd
import numpy as np

tasks = [0]
# tasks = [0, 1]

for task in tasks:
    best_acc_per_experimental_task = {}
    best_acc_per_experimental_task['best_accuracy'] = []
    sour_tar_dataset_names = [['CWRU_1hp_4', 'SBDS_1K_4_06']]

    path = './save_dir/result/'
    experiments = ['FMAA+LSMMD+Attention']
    os.makedirs(path + experiments[task], exist_ok=True)
    fp = open(path + experiments[task] + '/' + 'The mean and std deviation of the 10 best experimental accuracies.csv',
              'w')
    for load_transfer in sour_tar_dataset_names:
        source_data_name, target_data_name = load_transfer
        for number in range(0, 1):
            parser = argparse.ArgumentParser()
            parser.add_argument('--n_epochs_1', type=int, default=30)  # Pre-train source domain
            parser.add_argument('--n_epochs_2', type=int, default=50)  # Pre-train group discriminator
            parser.add_argument('--n_epochs_3', type=int, default=100)  # Adversarial training
            parser.add_argument('--warm_steps', type=int, default=40)  # Warm steps
            parser.add_argument('--n_target_samples', type=int, default=1)  # The number of few-shots of target domain
            parser.add_argument('--batch_size', type=int, default=64)  # 64
            parser.add_argument('--domain_loss_hyp', type=int, default=0.1)  # Weight of domain adversarial loss
            parser.add_argument('--positive_dcd_hyp', type=int, default=1)  # Weight of positive adversarial loss
            parser.add_argument('--negative_dcd_hyp', type=int, default=0.2)  # Weight of negative adversarial loss
            parser.add_argument('--sour_tar_per_cls_hyp', type=int, default=1)  # See readme for details.
            parser.add_argument('--tar_tar_mmd_hyp', type=int, default=1)  # See readme for details.
            opt = vars(parser.parse_args())

            s_dataset_path, t_dataset_path = files_path()

            ablation_task = task
            ablation_task_name = '/T' + source_data_name[5] + '-' + target_data_name[5] + '-w' + str(
                opt['warm_steps']) + '/'
            transfer_task = 'T' + source_data_name[5] + '-' + target_data_name[5] + '-w' + str(opt['warm_steps'])

            # ===========================================================================================================
            #                                          Initialization
            # ===========================================================================================================
            train_hist = train_history()
            global_step = 0
            use_cuda = True if torch.cuda.is_available() else False
            device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
            torch.manual_seed(1)
            if use_cuda:
                torch.cuda.manual_seed(1)
            # ===========================================================================================================
            #                     Extract the names of the source and target domain datasets
            # ===========================================================================================================
            s_dataset_path, t_dataset_path = files_path()
            sdatadir = s_dataset_path + source_data_name + '.mat'
            vdatadir = t_dataset_path + target_data_name + '.mat'
            save_name = source_data_name + '_to_' + target_data_name

            # ===========================================================================================================
            #                  Loading data, feature extractor, discriminator and loss function
            # ===========================================================================================================
            source_test_dataloader = dataloader.source_dataload(batch_size=128,
                                                                source_data_name=source_data_name)  # opt['batch_size']
            net = Lightweight_model(input_size=32, class_num=4)
            discriminators = Dmodel(output_groups=16)
            discriminators_domain = Domain_model(output_domain=1)
            # The model was skip to cuda
            net.to(device)
            discriminators.to(device)
            discriminators_domain.to(device)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_bce = nn.BCEWithLogitsLoss()
            mmd = MMDLoss()
            # ===========================================================================================================
            #                                Loading the source and target datasets
            # ===========================================================================================================
            X_s, Y_s = dataloader.source_sample(source_data_name=source_data_name)
            X_t, Y_t = dataloader.target_sample(opt['n_target_samples'],
                                                target_data_name=target_data_name)
            # ===========================================================================================================
            #                                Step 1: Pretrain F
            # ===========================================================================================================
            # step1: Initialization g and h by using source dataset
            pretrain_process_fn(net=net, batch_size=opt['batch_size'], pretrain_epochs=opt['n_epochs_1'], device=device,
                                loss_fn=loss_fn, source_data_name=source_data_name)
            # ===========================================================================================================
            #      Step 2: Training D  Purpose: To be able to identify which group the feature comes from
            # ===========================================================================================================
            pretrain_discriminator_fn(discriminators=discriminators, net=net, pretrain_d_epochs=opt['n_epochs_2'],
                                      n_target_samples=opt['n_target_samples'], device=device, loss_fn=loss_fn,
                                      X_s=X_s, Y_s=Y_s, X_t=X_t, Y_t=Y_t)
            # ===========================================================================================================
            #                                Step 3: Adversarial Training
            # ===========================================================================================================
            adversarial_process_cls = adversarial_process(net=net, discriminators=discriminators,
                                                          discriminators_domain=discriminators_domain,
                                                          pretrain_d_epochs=opt['n_epochs_2'],
                                                          n_target_samples=opt['n_target_samples'],
                                                          adv_epochs=opt['n_epochs_3'], device=device,
                                                          train_hist=train_hist,
                                                          source_data_name=source_data_name,
                                                          target_data_name=target_data_name,
                                                          warm_steps=opt['warm_steps'],
                                                          domain_loss_hyp=opt['domain_loss_hyp'],
                                                          positive_dcd_hyp=opt['positive_dcd_hyp'],
                                                          negative_dcd_hyp=opt['negative_dcd_hyp'],
                                                          sour_tar_per_cls_hyp=opt['sour_tar_per_cls_hyp'],
                                                          tar_tar_mmd_hyp=opt['tar_tar_mmd_hyp'])
            best_acc, best_pre_labels, y_test_ori, best_model = adversarial_process_cls.fit()
            # ===========================================================================================================
            print('best_acc = ', best_acc)
            best_acc_per_experimental_task['best_accuracy'].append(best_acc)

            acc_str = str(number + 1) + '-' + str(best_acc) + '/'

            save_path = path + experiments[ablation_task] + ablation_task_name + acc_str
            os.makedirs(save_path, exist_ok=True)
            save_his(train_hist=train_hist, save_dir=save_path, save_name=save_name)
            save_predict_labels(yt_label=y_test_ori, yt_pre_label=best_pre_labels, save_dir=save_path, save_name=save_name)
            save_model(model=best_model, save_dir=save_path, model_name='SCARA_lightweight',
                       source_data_name=source_data_name, target_data_name=target_data_name)
        save_dir = path + experiments[ablation_task] + ablation_task_name
        save_name = source_data_name + '_to_' + target_data_name
        data_df = pd.DataFrame(best_acc_per_experimental_task)
        data_df.to_csv(save_dir + save_name + '_best_acc_results.csv')

        best_acc_per_experimental_mean = np.mean(best_acc_per_experimental_task['best_accuracy'])
        best_acc_per_experimental_std = np.std(best_acc_per_experimental_task['best_accuracy'], ddof=1)
        per_task_best_acc_mean_std = '{}, Best_acc_per_task_mean:, {:.5f}, Best_acc_per_task_mean_var:, {:.2f}±{:.2f}'.format(
            transfer_task, best_acc_per_experimental_mean, best_acc_per_experimental_mean * 100,
                                                           best_acc_per_experimental_std * 100)
        fp.write(per_task_best_acc_mean_std + '\n')
    fp.close()


