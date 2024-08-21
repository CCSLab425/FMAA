"""
Author:  Qi tong Chen
Date: 2024.08.20
Pre-train group discriminator
"""

import torch
import dataloader
import numpy as np


def pretrain_discriminator_fn(discriminators=None, net=None, pretrain_d_epochs=30, n_target_samples=1, device='cuda:0',
                              loss_fn=torch.nn.CrossEntropyLoss(), X_s=None, Y_s=None, X_t=None, Y_t=None):
    """
    Pre-training discriminator D
    :param discriminators:
    :param net:
    :param pretrain_d_epochs:
    :param n_target_samples:
    :param device:
    :param loss_fn:
    :param X_s:
    :param Y_s:
    :param X_t:
    :param Y_t:
    :return:
    """
    for param_d in discriminators.parameters():
        param_d.requires_grad = True  # Updating the parameters of the discriminator
    for param_f in net.parameters():
        param_f.requires_grad = False

    optimizer_D = torch.optim.Adam(discriminators.parameters(), lr=0.001)
    for epoch in range(pretrain_d_epochs):
        dcd_pre_train_loss_mean = []
        groups_16, groups_16_y = dataloader.sample_groups(X_s, Y_s, X_t, Y_t,
                                                                            seed=epoch)
        n_iters = len(groups_16) * len(groups_16[1])
        index_list = torch.randperm(n_iters)
        mini_batch_size = n_iters // n_target_samples  # use mini_batch train can be more stable

        X1_16 = []  # Initialization
        X2_16 = []  # Initialization
        ground_truths_16 = []
        for index in range(n_iters):
            # Select group G, which means to extract from G1-G16, with a range of 0-15
            ground_truth_16 = index_list[index] // len(groups_16[1])
            # Select data pairs, select the detailed_group-th data pair (X1, X2) of the G-th group
            detailed_group_16 = index_list[index] - len(
                groups_16[1]) * ground_truth_16
            x1_16, x2_16 = groups_16[ground_truth_16][detailed_group_16]
            y1_16, y2_16 = groups_16_y[ground_truth_16][detailed_group_16]
            X1_16.append(x1_16)  # The first sample of the data pair
            X2_16.append(x2_16)  # The second sample of the data pair
            ground_truths_16.append(ground_truth_16)  # Record the group G to which (X1, X2) belongs
            # =====================================================================================================
            #                               select data for a mini-batch to train
            # =====================================================================================================
            if (index + 1) % mini_batch_size == 0:
                X1_16 = torch.stack(X1_16)
                X2_16 = torch.stack(X2_16)
                ground_truths_16 = torch.LongTensor(ground_truths_16)
                X1_16 = X1_16.to(device)
                X2_16 = X2_16.to(device)
                ground_truths_16 = ground_truths_16.to(device)
                _, x1_f_16, _, _, _ = net(X1_16)
                _, x2_f_16, _, _, _ = net(X2_16)
                optimizer_D.zero_grad()
                X_cat_16 = torch.cat([x1_f_16, x2_f_16], 1)
                G_pred_16 = discriminators(X_cat_16.detach())
                loss = loss_fn(G_pred_16, ground_truths_16)
                loss.backward()
                optimizer_D.step()
                dcd_pre_train_loss_mean.append(loss.item())

                X1_16 = []  # Initialization
                X2_16 = []  # Initialization
                ground_truths_16 = []  # Initialization

        print("step2----Epoch %d/%d loss:%.3f" % (epoch + 1, pretrain_d_epochs, np.mean(dcd_pre_train_loss_mean)))