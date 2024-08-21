"""
Author:  Qi tong Chen
Date: 2024.08.20
The adversarial training process between feature extractor and discriminator.
"""
import torch
import dataloader
from utils_dsbn.adjust_factors import amplification_factor, attenuation_factor
from utils_dsbn.functions import dcd_labels_calculate_positive, dcd_labels_calculate_negative_1, dcd_labels_calculate_negative_2, dcd_labels_calculate_negative_3, soft_max
from torch import nn
import numpy as np
from utils_dsbn.save_and_other_functions import save_model
from MMD.MMD_calculation import MMDLoss


class adversarial_process():
    def __init__(self, net=None, discriminators=None, discriminators_domain=None, conditional_d=None, pretrain_d_epochs=30,
                 n_target_samples=1, adv_epochs=200, device='cuda:0', train_hist=None, source_data_name='',
                 target_data_name='', warm_steps=100, domain_loss_hyp=0.1, positive_dcd_hyp=1, negative_dcd_hyp=0.2,
                 sour_tar_per_cls_hyp=1, tar_tar_mmd_hyp=1):
        super(adversarial_process, self).__init__()
        self.net = net
        self.discriminators = discriminators
        self.discriminators_domain = discriminators_domain
        self.conditional_d_model = conditional_d
        self.optimizer_g_h = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.optimizer_d = torch.optim.Adam(self.discriminators.parameters(), lr=0.001)
        self.optimizer_domain = torch.optim.Adam(self.discriminators_domain.parameters(), lr=0.001)
        self.mmd = MMDLoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.pretrain_d_epochs = pretrain_d_epochs
        self.n_target_samples = n_target_samples
        self.adv_epochs = adv_epochs
        self.device = device
        self.train_hist = train_hist
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.dcd_label_8_negative2 = 0
        self.dcd_label_8_negative3 = 0
        self.best_pre_labels = 0
        self.y_test_ori = 0

        self.domain_loss_hyp = domain_loss_hyp
        self.positive_dcd_hyp = positive_dcd_hyp
        self.negative_dcd_hyp = negative_dcd_hyp
        self.sour_tar_per_cls_hyp = sour_tar_per_cls_hyp
        self.tar_tar_mmd_hyp = tar_tar_mmd_hyp

        self.conditional_domain_loss_adjust_factor = 0.
        self.warm_steps = warm_steps

    def create_groups(self, epoch):
        """
        Create Group
        :param epoch: Current epoch
        :return:
        """
        X_s, Y_s = dataloader.source_sample(source_data_name=self.source_data_name)
        X_t, Y_t = dataloader.target_sample(self.n_target_samples, target_data_name=self.target_data_name)
        groups_16, groups_16_y = dataloader.sample_groups(X_s, Y_s, X_t, Y_t, seed=self.pretrain_d_epochs + epoch)
        return groups_16, groups_16_y

    def groups_processing(self, groups_16, groups_16_y):
        """
        Select the group you want to compete against
        :param groups_16: The 16 group samples generated
        :param groups_16_y: Labels corresponding to 16 groups of samples
        :return:
        """
        # G1, G2,   G3,   G4,   G5,   G6,   G7,   G8,   G9,   G10,  G11,  G12,  G13,  G14,  G15,  G16
        G1_1, G1_2, G1_3, G1_4, G2_1, G2_2, G2_3, G2_4, G3_1, G3_2, G3_3, G3_4, G4_1, G4_2, G4_3, G4_4 = groups_16
        Y1_1, Y1_2, Y1_3, Y1_4, Y2_1, Y2_2, Y2_3, Y2_4, Y3_1, Y3_2, Y3_3, Y3_4, Y4_1, Y4_2, Y4_3, Y4_4 = groups_16_y
        groups_8 = [G2_1, G2_2, G2_3, G2_4,  # Positive pairs
                    G4_1, G4_2, G4_3, G4_4]  # Negative pairs
        groups_y_8 = [Y2_1, Y2_2, Y2_3, Y2_4,
                      Y4_1, Y4_2, Y4_3, Y4_4]
        return groups_8, groups_y_8

    def mini_batch_size_g_and_dcd(self, groups_16, groups_16_y):
        """
        Calculate the grouping sequence of the feature extractor F and the discriminator D, and update the
        feature extractor and discriminator once every mini_batch_size samples loaded during confrontation
        :param groups_16: The 16 group samples generated
        :param groups_16_y: Labels corresponding to 16 groups of samples
        :return:
        """
        groups_8, groups_y_8 = self.groups_processing(groups_16, groups_16_y)
        n_iters = len(groups_8) * len(groups_8[0])
        index_list = torch.randperm(n_iters)
        n_iters_dcd = len(groups_16) * len(groups_8[0])
        index_list_dcd = torch.randperm(n_iters_dcd)
        mini_batch_size_g_h = n_iters // self.n_target_samples
        mini_batch_size_dcd = n_iters_dcd // self.n_target_samples

        return n_iters, n_iters_dcd, index_list, index_list_dcd, mini_batch_size_g_h, mini_batch_size_dcd

    def sour_per_cls_fea(self):
        """
        1.Extract predictive features for each class in the source domain
        2.Calculate the prediction loss of the source domain dataset
        Note that this article takes the four-category task as an example.
        :return:sour_train_pre_labels_cat.shape=[800, 10]
        """
        sour_test_loss = 0
        sour_test_acc = 0
        counts_sour = 0
        sour_test_pre_labels = []
        sour_cls_0_pre, sour_cls_1_pre, sour_cls_2_pre, sour_cls_3_pre = [], [], [], []
        source_test_dataloader = dataloader.source_dataload(batch_size=64, source_data_name=self.source_data_name)
        for source_test, source_test_labels in source_test_dataloader:
            counts_sour += 1
            source_test = source_test.to(self.device)
            source_test_labels = source_test_labels.to(self.device)
            # print('source_test.shape = ', source_test.shape)
            source_test_pre, fea_4_s, fea_3_s, fea_2_s, fea_1_s = self.net(source_test)
            # ===============================================
            # Extract the prediction features of each class in the source domain to facilitate MMD
            # with the class corresponding to the target domain
            for i in range(len(source_test_labels)):
                if source_test_labels[i].item() == 0:  # Find the index i corresponding to the source domain 0 label
                    sour_cls_0_pre.append(source_test_pre[i])
                elif source_test_labels[i].item() == 1:
                    sour_cls_1_pre.append(source_test_pre[i])
                elif source_test_labels[i].item() == 2:
                    sour_cls_2_pre.append(source_test_pre[i])
                elif source_test_labels[i].item() == 3:
                    sour_cls_3_pre.append(source_test_pre[i])
            # ================================================
            sour_test_pre_labels.append(source_test_pre)
            sour_test_acc_iter = torch.sum((torch.argmax(source_test_pre, 1) == source_test_labels)) / len(
                source_test_labels)
            sour_test_acc += sour_test_acc_iter
            sour_test_loss_iter = self.loss_fn(source_test_pre, source_test_labels)
            sour_test_loss += sour_test_loss_iter

        sour_test_loss = sour_test_loss / counts_sour
        sour_test_acc = round((sour_test_acc.item() / counts_sour), 5)
        # Features of all samples corresponding to the source domain label
        sour_cls_0_pre = torch.stack(sour_cls_0_pre)
        sour_cls_1_pre = torch.stack(sour_cls_1_pre)
        sour_cls_2_pre = torch.stack(sour_cls_2_pre)
        sour_cls_3_pre = torch.stack(sour_cls_3_pre)
        # ===================================
        # Concatenate source domain predicted labels
        sour_train_pre_labels_cat = 0
        for i in range(len(sour_test_pre_labels)):
            if i == 0:
                sour_train_pre_labels_cat = sour_test_pre_labels[0]
            else:
                sour_train_pre_labels_cat = torch.cat((sour_train_pre_labels_cat, sour_test_pre_labels[i]), dim=0)
        return sour_cls_0_pre, sour_cls_1_pre, sour_cls_2_pre, sour_cls_3_pre, sour_train_pre_labels_cat, sour_test_loss, sour_test_acc

    def tar_and_shot_tar_mmd(self, T_shot_pre):
        """
        Calculate the mmd between the complete target domain dataset and the small sample of the target domain
        :param T_shot_pre:Few-shot features of the target domain extracted by the feature extractor
        :return:mmd loss between target domain and small sample of target domain; pseudo label of target domain dataset
        """
        tar_mmd_loss = 0
        tar_mmd_loss_numpy = []
        counts_tar = 0
        tar_train_pre = []
        target_test_dataloader = dataloader.target_dataload(batch_size=64, target_data_name=self.target_data_name)
        for target_train, _ in target_test_dataloader:
            """
            labels.shape=[64]
            """
            counts_tar += 1
            target_train = target_train.to(self.device)
            tar_pre, _, _, _, _ = self.net(target_train)
            tar_train_pre.append(tar_pre)  # Add the predicted label of each iter target domain
            mmd_tar_and_tar = self.mmd(T_shot_pre, tar_pre)  # MMD between small samples and complete dataset of target domain.
            tar_mmd_loss += mmd_tar_and_tar
            tar_mmd_loss_numpy.append(mmd_tar_and_tar.item())
        tar_and_tar_mmd_loss = tar_mmd_loss / counts_tar  # Calculate the mean of an epoch

        # ===================================
        # Concatenate the predicted labels of the target domain
        tar_train_pre_labels_cat = 0
        for i in range(len(tar_train_pre)):
            if i == 0:
                tar_train_pre_labels_cat = tar_train_pre[0]
            else:
                tar_train_pre_labels_cat = torch.cat((tar_train_pre_labels_cat, tar_train_pre[i]), dim=0)
        return tar_and_tar_mmd_loss, tar_train_pre_labels_cat

    def tar_per_cls_fea(self, tar_train_pre_labels_cat):
        """
        1.Extract predictive features for each class in the target domain
        Note that this article takes the four-category task as an example.
        :param tar_train_pre_labels_cat: Pseudo labels for target domain datasets, shape=[800, 4]
        :return: Features of the four pseudo-class
        """
        tar_pred_lab = torch.argmax(tar_train_pre_labels_cat, 1)
        tar_cls_0_pre, tar_cls_1_pre, tar_cls_2_pre, tar_cls_3_pre = [], [], [], []
        for i in range(len(tar_train_pre_labels_cat)):
            if tar_pred_lab[i].item() == 0:
                tar_cls_0_pre.append(tar_train_pre_labels_cat[i])
            elif tar_pred_lab[i].item() == 1:
                tar_cls_1_pre.append(tar_train_pre_labels_cat[i])
            elif tar_pred_lab[i].item() == 2:
                tar_cls_2_pre.append(tar_train_pre_labels_cat[i])
            elif tar_pred_lab[i].item() == 3:
                tar_cls_3_pre.append(tar_train_pre_labels_cat[i])
        if len(tar_cls_0_pre) > 0:
            tar_cls_0_pre = torch.stack(tar_cls_0_pre)  # All sample features corresponding to the label 0 of the TD
        else:
            tar_cls_0_pre = torch.zeros_like(tar_train_pre_labels_cat[0:200, :])
        if len(tar_cls_1_pre) > 0:
            tar_cls_1_pre = torch.stack(tar_cls_1_pre)
        else:
            tar_cls_1_pre = torch.zeros_like(tar_train_pre_labels_cat[0:200, :])
        if len(tar_cls_2_pre) > 0:
            tar_cls_2_pre = torch.stack(tar_cls_2_pre)
        else:
            tar_cls_2_pre = torch.zeros_like(tar_train_pre_labels_cat[0:200, :])
        if len(tar_cls_3_pre) > 0:
            tar_cls_3_pre = torch.stack(tar_cls_3_pre)
        else:
            tar_cls_3_pre = torch.zeros_like(tar_train_pre_labels_cat[0:200, :])

        return tar_cls_0_pre, tar_cls_1_pre, tar_cls_2_pre, tar_cls_3_pre

    def sour_and_tar_per_cls_mmd(self, sour_cls_0_pre, sour_cls_1_pre, sour_cls_2_pre, sour_cls_3_pre,
                                 tar_cls_0_pre, tar_cls_1_pre, tar_cls_2_pre, tar_cls_3_pre):
        """
        CMMD，Calculate the mmd between each pseudo-category feature in the source domain
        and each pseudo-category feature in the target domain
        :param sour_cls_0_pre: Features corresponding to pseudo class 0 in the SD extracted by the feature extractor
        :param sour_cls_1_pre: Features corresponding to pseudo class 1 in the SD extracted by the feature extractor
        :param sour_cls_2_pre: Features corresponding to pseudo class 2 in the SD extracted by the feature extractor
        :param sour_cls_3_pre: Features corresponding to pseudo class 3 in the SD extracted by the feature extractor
        :param tar_cls_0_pre: Features corresponding to pseudo class 0 in the TD extracted by the feature extractor
        :param tar_cls_1_pre: Features corresponding to pseudo class 1 in the TD extracted by the feature extractor
        :param tar_cls_2_pre: Features corresponding to pseudo class 2 in the TD extracted by the feature extractor
        :param tar_cls_3_pre: Features corresponding to pseudo class 3 in the TD extracted by the feature extractor
        :return:The total MMD loss and the MMD loss between each class in the source and target domains
        """
        sour_and_tar_mmd_loss_cls_1 = self.mmd(sour_cls_0_pre, tar_cls_0_pre)
        sour_and_tar_mmd_loss_cls_2 = self.mmd(sour_cls_1_pre, tar_cls_1_pre)
        sour_and_tar_mmd_loss_cls_3 = self.mmd(sour_cls_2_pre, tar_cls_2_pre)
        sour_and_tar_mmd_loss_cls_4 = self.mmd(sour_cls_3_pre, tar_cls_3_pre)
        sour_and_tar_mmd_loss_per_cls = sour_and_tar_mmd_loss_cls_1 + sour_and_tar_mmd_loss_cls_2 + \
                                        sour_and_tar_mmd_loss_cls_3 + sour_and_tar_mmd_loss_cls_4

        num_sour_samples = len(sour_cls_0_pre) + len(sour_cls_1_pre) + len(sour_cls_2_pre) + len(sour_cls_3_pre)
        num_tar_samples = len(tar_cls_0_pre) + len(tar_cls_1_pre) + len(tar_cls_2_pre) + len(tar_cls_3_pre)
        p_s0 = len(sour_cls_0_pre) / num_sour_samples  # The class posterior probability P(Xs|Ys=0) of the source domain
        p_s1 = len(sour_cls_1_pre) / num_sour_samples
        p_s2 = len(sour_cls_2_pre) / num_sour_samples
        p_s3 = len(sour_cls_3_pre) / num_sour_samples
        p_t0 = len(tar_cls_0_pre) / num_tar_samples  # The class posterior probability P(Xt|Yt=0) of the target domain
        p_t1 = len(tar_cls_1_pre) / num_tar_samples
        p_t2 = len(tar_cls_2_pre) / num_tar_samples
        p_t3 = len(tar_cls_3_pre) / num_tar_samples

        return sour_and_tar_mmd_loss_per_cls, sour_and_tar_mmd_loss_cls_1, sour_and_tar_mmd_loss_cls_2, \
               sour_and_tar_mmd_loss_cls_3, sour_and_tar_mmd_loss_cls_4, p_s0, p_s1, p_s2, p_s3, p_t0, p_t1, p_t2, p_t3

    def joint_adv_fea_extractor_loss(self):
        """
        Extract fault features of source and target domain datasets, and input the features into the discriminator to
        calculate domain loss, which is used to optimize the feature extractor and extract domain invariant features
        (so that the feature extractor can confuse the discriminator to the greatest extent)
        :return: Source domain discrimination loss; target domain discrimination loss; domain discrimination loss
        """
        sour_and_tar_loader = dataloader.source_and_target_loader(batch_size=64, source_data_name=self.source_data_name,
                                                                  target_data_name=self.target_data_name)
        sour_adv_loss = 0
        tar_adv_loss = 0
        for iter, (xs, ys, xt, yt) in enumerate(sour_and_tar_loader):
            xs = xs.to(self.device)
            xt = xt.to(self.device)
            xs_pre, fs, _, _, _ = self.net(xs)
            xt_pre, ft, _, _, _ = self.net(xt)
            d_out_s = self.discriminators_domain(fs)
            d_out_t = self.discriminators_domain(ft)
            sour_labels = torch.ones_like(d_out_s).to(self.device)
            tar_labels = torch.zeros_like(d_out_t).to(self.device)
            sour_adv_loss += self.loss_bce(d_out_s, sour_labels)  # SD-->1
            tar_adv_loss += self.loss_bce(d_out_t, tar_labels)  # TD-->0
        sour_adv_loss_mean = sour_adv_loss / iter
        tar_adv_loss_mean = tar_adv_loss / iter
        # print(iter, sour_adv_loss_mean, tar_adv_loss_mean)
        Gloss = +(sour_adv_loss_mean + tar_adv_loss_mean)

        return sour_adv_loss_mean, tar_adv_loss_mean, Gloss

    def joint_adv_discriminator_loss(self):
        """
        Calculate the domain discrimination loss of the discriminator to optimize the discriminator
        SD true labels: 0
        TD true labels: 1
        :return: Domain discrimination loss
        """
        sour_and_tar_loader = dataloader.source_and_target_loader(batch_size=64, source_data_name=self.source_data_name,
                                                                  target_data_name=self.target_data_name)
        sour_adv_loss = 0
        tar_adv_loss = 0
        for iter, (xs, ys, xt, yt) in enumerate(sour_and_tar_loader):
            xs = xs.to(self.device)
            xt = xt.to(self.device)
            xs_pre, fs, _, _, _ = self.net(xs)
            xt_pre, ft, _, _, _ = self.net(xt)
            d_out_s = self.discriminators_domain(fs.detach())
            d_out_t = self.discriminators_domain(ft.detach())
            sour_labels = torch.zeros_like(d_out_s).to(self.device)
            tar_labels = torch.ones_like(d_out_t).to(self.device)
            sour_adv_loss += self.loss_bce(d_out_s, sour_labels)  # SD-->0
            tar_adv_loss += self.loss_bce(d_out_t, tar_labels)  # TD-->1
        sour_adv_loss_mean = sour_adv_loss / iter
        tar_adv_loss_mean = tar_adv_loss / iter
        Dloss = sour_adv_loss_mean + tar_adv_loss_mean

        return Dloss

    def train_dcd(self, groups_16, groups_16_y):
        """
        Train the group discriminator and the domain adversarial discriminator at the same time
        :param groups_16: The 16 group samples generated
        :param groups_16_y: label y
        :return:
        """
        for param_d in self.discriminators.parameters():
            param_d.requires_grad = True
        for param_f in self.net.parameters():
            param_f.requires_grad = False
        for param_domain in self.discriminators_domain.parameters():
            param_domain.requires_grad = True
        n_iters, n_iters_dcd, index_list, index_list_dcd, mini_batch_size_g_h, mini_batch_size_dcd = \
            self.mini_batch_size_g_and_dcd(groups_16, groups_16_y)
        X1_16 = []
        X2_16 = []
        ground_truths_16 = []
        for index in range(n_iters_dcd):
            ground_truth_16 = index_list_dcd[index] // len(groups_16[1])
            detailed_group_16 = index_list_dcd[index] - len(groups_16[1]) * ground_truth_16
            x1_16, x2_16 = groups_16[ground_truth_16][detailed_group_16]
            X1_16.append(x1_16)
            X2_16.append(x2_16)
            ground_truths_16.append(ground_truth_16)

            if (index + 1) % mini_batch_size_dcd == 0:
                X1_16 = torch.stack(X1_16)
                X2_16 = torch.stack(X2_16)
                ground_truths_16 = torch.LongTensor(ground_truths_16)
                X1_16 = X1_16.to(self.device)
                X2_16 = X2_16.to(self.device)
                ground_truths_16 = ground_truths_16.to(self.device)
                self.optimizer_d.zero_grad()

                _, f_X1_16, _, _, _ = self.net(X1_16)
                _, f_X2_16, _, _, _ = self.net(X2_16)
                X_cat_16 = torch.cat([f_X1_16, f_X2_16], 1)
                y_pred_16 = self.discriminators(X_cat_16.detach())
                loss = self.loss_fn(y_pred_16, ground_truths_16)
                loss.backward()
                self.optimizer_d.step()

                # Dloss
                self.optimizer_domain.zero_grad()
                Dloss = self.domain_loss_hyp * self.joint_adv_discriminator_loss()
                Dloss.backward()
                self.optimizer_domain.step()

                X1_16 = []
                X2_16 = []
                ground_truths_16 = []

    def train_feature_extractor(self, global_step, adaptation_attenuation_factor,
                                adaptation_amplification_factor, groups_16, groups_16_y):
        """
        Training the feature extractor
        :param global_step: The step size used to update Visdom
        :param adaptation_attenuation_factor: Attenuation Factor
        :param adaptation_amplification_factor: Amplification Factor
        :param groups_16: The generated samples of 16 groups
        :param groups_16_y: The labels y corresponding to the samples of the 16 groups
        :return:
        """
        X1_8 = []  # The first element of the data pair
        X2_8 = []  # The second element of the data pair
        ground_truths_y1_8 = []
        ground_truths_y2_8 = []
        dcd_labels_8 = []
        dcd_labels_8_negative = []  # Used to increase the distribution differences among different classes
        dcd_labels_8_negative2 = []  # Used to increase the distribution differences among different classes
        dcd_labels_8_negative3 = []  # Used to increase the distribution differences among different classes
        # =====================================================================================================
        #                               training F , D is frozen
        # =====================================================================================================
        for param_d in self.discriminators.parameters():
            param_d.requires_grad = False
        for param_f in self.net.parameters():
            param_f.requires_grad = True
        for param_domain in self.discriminators_domain.parameters():
            param_domain.requires_grad = False
        X_t, Y_t = dataloader.target_sample(self.n_target_samples, target_data_name=self.target_data_name)

        groups_8, groups_y_8 = self.groups_processing(groups_16, groups_16_y)
        n_iters, n_iters_dcd, index_list, index_list_dcd, mini_batch_size_g_h, mini_batch_size_dcd = \
            self.mini_batch_size_g_and_dcd(groups_16, groups_16_y)
        for index in range(n_iters):
            ground_truth_for_groups_8 = index_list[index] // len(groups_8[0])
            detailed_group_8 = index_list[index] - len(groups_8[0]) * ground_truth_for_groups_8  # A set of corresponding data pairs
            x1_8, x2_8 = groups_8[ground_truth_for_groups_8][detailed_group_8]  # Select data pairs
            y1_8, y2_8 = groups_y_8[ground_truth_for_groups_8][detailed_group_8]  # Labels corresponding to data pairs
            # print('Take the [{}]th data pair of the [{}]th group, the data pair is ({}, {})'.format(
            #     ground_truth_for_groups_8, detailed_group_8, y1_8, y2_8))
            # ====================================================
            # Groups containing the same category are aligned
            dcd_label_8 = dcd_labels_calculate_positive(ground_truth_for_groups_8=ground_truth_for_groups_8)
            # Align groups containing different categories
            dcd_label_8_negative = dcd_labels_calculate_negative_1(ground_truth_for_groups_8=ground_truth_for_groups_8)
            if ground_truth_for_groups_8 < 4:
                self.dcd_label_8_negative2 = dcd_labels_calculate_negative_2(
                    ground_truth_for_groups_8=ground_truth_for_groups_8)
                self.dcd_label_8_negative3 = dcd_labels_calculate_negative_3(
                    ground_truth_for_groups_8=ground_truth_for_groups_8)
            # ====================================================
            X1_8.append(x1_8)
            X2_8.append(x2_8)
            ground_truths_y1_8.append(y1_8)
            ground_truths_y2_8.append(y2_8)
            dcd_labels_8.append(dcd_label_8)
            dcd_labels_8_negative.append(dcd_label_8_negative)
            dcd_labels_8_negative2.append(self.dcd_label_8_negative2)
            dcd_labels_8_negative3.append(self.dcd_label_8_negative3)

            if (index + 1) % mini_batch_size_g_h == 0:
                # Each mini_batch_size_g_h is stacked once, which determines the batch of the input model.
                X1_8 = torch.stack(X1_8)
                X2_8 = torch.stack(X2_8)
                ground_truths_y1_8 = torch.LongTensor(ground_truths_y1_8)
                ground_truths_y2_8 = torch.LongTensor(ground_truths_y2_8)
                dcd_labels_8 = torch.LongTensor(dcd_labels_8)
                dcd_labels_8_negative = torch.LongTensor(dcd_labels_8_negative)
                dcd_labels_8_negative2 = torch.LongTensor(dcd_labels_8_negative2)
                dcd_labels_8_negative3 = torch.LongTensor(dcd_labels_8_negative3)
                X1_8 = X1_8.to(self.device)
                X2_8 = X2_8.to(self.device)
                ground_truths_y1_8 = ground_truths_y1_8.to(self.device)
                ground_truths_y2_8 = ground_truths_y2_8.to(self.device)
                dcd_labels_8 = dcd_labels_8.to(self.device)
                dcd_labels_8_negative = dcd_labels_8_negative.to(self.device)
                dcd_labels_8_negative2 = dcd_labels_8_negative2.to(self.device)
                dcd_labels_8_negative3 = dcd_labels_8_negative3.to(self.device)

                self.optimizer_g_h.zero_grad()
                self.optimizer_domain.zero_grad()
                pre_X1_8, pre_X1_fea_8, _, _, _ = self.net(X1_8)
                pre_X2_8, pre_X2_fea_8, _, _, _ = self.net(X2_8)
                X_cat_fea_8 = torch.cat([pre_X1_fea_8, pre_X2_fea_8], 1)
                y_pred_X1_8 = pre_X1_8
                y_pred_X2_8 = pre_X2_8
                y_pred_dcd_8 = self.discriminators(X_cat_fea_8)
                loss_X1_8 = self.loss_fn(y_pred_X1_8, ground_truths_y1_8)
                loss_X2_8 = self.loss_fn(y_pred_X2_8, ground_truths_y2_8)

                train_pre_labels = torch.cat((y_pred_X1_8, y_pred_X2_8), dim=0)
                train_labels = torch.cat((ground_truths_y1_8, ground_truths_y2_8), dim=0)
                train_loss = self.loss_fn(train_pre_labels, train_labels)
                train_pre_labels = torch.argmax(train_pre_labels, 1)
                train_acc = torch.sum((train_pre_labels == train_labels)) / len(train_labels)

                # print('train_acc = ', train_acc)
                self.train_loss_per_epoch.append(train_loss.item())
                self.train_acc_per_epoch.append(train_acc.item())
                loss_dcd = self.loss_fn(y_pred_dcd_8, dcd_labels_8)
                loss_dcd_negative = self.loss_fn(y_pred_dcd_8, dcd_labels_8_negative)
                loss_dcd_negative2 = self.loss_fn(y_pred_dcd_8, dcd_labels_8_negative2)
                loss_dcd_negative3 = self.loss_fn(y_pred_dcd_8, dcd_labels_8_negative3)

                # =====================================================================================================
                # 1.Extract predictive features for each class in the source domain
                # 2.Calculate the prediction loss of the source domain dataset
                sour_cls_0_pre, sour_cls_1_pre, sour_cls_2_pre, sour_cls_3_pre, sour_train_pre_labels_cat, \
                sour_test_loss, sour_test_acc = self.sour_per_cls_fea()
                # =====================================================================================================
                #    Calculate the mmd loss between the small sample and the complete dataset of the target domain.
                # =====================================================================================================
                X_t_new = X_t.to(self.device)
                T_shot_pre, _, _, _, _ = self.net(X_t_new)
                tar_and_tar_mmd_loss, tar_train_pre_labels_cat = self.tar_and_shot_tar_mmd(T_shot_pre=T_shot_pre)

                # Find the pseudo features of each category based on the pseudo labels of the target domain dataset
                tar_cls_0_pre, tar_cls_1_pre, tar_cls_2_pre, tar_cls_3_pre = self.tar_per_cls_fea(
                    tar_train_pre_labels_cat=tar_train_pre_labels_cat)
                # Calculate the mmd distance between each category of pseudo features in the SD and the TD
                sour_and_tar_mmd_loss_per_cls, sour_and_tar_mmd_loss_cls_1, sour_and_tar_mmd_loss_cls_2, \
                sour_and_tar_mmd_loss_cls_3, sour_and_tar_mmd_loss_cls_4, \
                p_s0, p_s1, p_s2, p_s3, p_t0, p_t1, p_t2, p_t3 = self.sour_and_tar_per_cls_mmd(
                    sour_cls_0_pre=sour_cls_0_pre, sour_cls_1_pre=sour_cls_1_pre,
                    sour_cls_2_pre=sour_cls_2_pre, sour_cls_3_pre=sour_cls_3_pre,
                    tar_cls_0_pre=tar_cls_0_pre, tar_cls_1_pre=tar_cls_1_pre,
                    tar_cls_2_pre=tar_cls_2_pre, tar_cls_3_pre=tar_cls_3_pre)
                # Gloss
                sour_adv_loss_mean, tar_adv_loss_mean, Gloss = self.joint_adv_fea_extractor_loss()

                loss_sum = loss_X1_8 + self.domain_loss_hyp * Gloss \
                           + self.positive_dcd_hyp * loss_dcd * adaptation_attenuation_factor \
                           - self.negative_dcd_hyp * (loss_dcd_negative2 + loss_dcd_negative3) \
                           + self.sour_tar_per_cls_hyp * sour_and_tar_mmd_loss_per_cls * adaptation_amplification_factor \
                           + self.tar_tar_mmd_hyp * tar_and_tar_mmd_loss

                loss_sum.backward()
                self.optimizer_g_h.step()

                X1_8 = []  # Initialization
                X2_8 = []
                ground_truths_y1_8 = []
                ground_truths_y2_8 = []
                dcd_labels_8 = []
                dcd_labels_8_negative = []
                dcd_labels_8_negative2 = []
                dcd_labels_8_negative3 = []

        self.train_hist['sour_and_tar_mmd_loss_cls_1'].append(sour_and_tar_mmd_loss_cls_1.item())
        self.train_hist['sour_and_tar_mmd_loss_cls_2'].append(sour_and_tar_mmd_loss_cls_2.item())
        self.train_hist['sour_and_tar_mmd_loss_cls_3'].append(sour_and_tar_mmd_loss_cls_3.item())
        self.train_hist['sour_and_tar_mmd_loss_cls_4'].append(sour_and_tar_mmd_loss_cls_4.item())
        self.train_hist['Sour_test_loss'].append(sour_test_loss.item())
        self.train_hist['Sour_test_accuracy'].append(sour_test_acc)

        train_loss_mean = np.mean(self.train_loss_per_epoch)
        train_acc_mean = np.mean(self.train_acc_per_epoch)

        return train_acc_mean, train_loss_mean, sour_cls_0_pre, sour_cls_1_pre, sour_cls_2_pre, sour_cls_3_pre, \
               tar_cls_0_pre, tar_cls_1_pre, tar_cls_2_pre, tar_cls_3_pre

    def fit(self):
        best_acc = 0
        global_step = 0
        for epoch in range(self.adv_epochs):
            global_step += 1
            self.train_loss_per_epoch = []
            self.train_acc_per_epoch = []
            groups_16, groups_16_y = self.create_groups(epoch)
            adaptation_amplification_factor = amplification_factor(epoch=epoch, warm_epoch=self.warm_steps,
                                                                   max_epoch=self.adv_epochs, gamma=10, rate=1)
            adaptation_attenuation_factor = attenuation_factor(epoch=epoch, warm_epoch=self.warm_steps, alpha=10,
                                                               beta=0.75, max_epoch=self.adv_epochs)
            # ===================================================================================================
            #                     1. Train feature extractor and freeze discriminator parameters
            # ===================================================================================================
            train_acc_mean, train_loss_mean, sour_cls_0_pre, sour_cls_1_pre, sour_cls_2_pre, sour_cls_3_pre, \
            tar_cls_0_pre, tar_cls_1_pre, tar_cls_2_pre, tar_cls_3_pre = self.train_feature_extractor(
                global_step=global_step, adaptation_attenuation_factor=adaptation_attenuation_factor,
                adaptation_amplification_factor=adaptation_amplification_factor, groups_16=groups_16, groups_16_y=groups_16_y)
            # ==================================================================================================
            #                2. Train the discriminator and freeze the feature extractor parameters
            # ==================================================================================================
            self.train_dcd(groups_16, groups_16_y)
            # ===================================================================================================

            best_acc, best_pre_labels, y_test_ori, best_model = self.model_evaluation(
                epoch=epoch, global_step=global_step, best_acc=best_acc, train_acc_mean=train_acc_mean,
                train_loss_mean=train_loss_mean)
        return best_acc, self.best_pre_labels, self.y_test_ori, best_model

    def model_evaluation(self, epoch, global_step, best_acc, train_acc_mean, train_loss_mean):
        self.net.eval()
        acc = 0
        print('Start Testing!!!')
        y_test_pre_labels = []
        y_test_labels = []
        test_loss_total = []
        target_test_dataloader = dataloader.target_dataload(batch_size=64, target_data_name=self.target_data_name)
        for data, labels in target_test_dataloader:
            """
            labels.shape=[64]
            """
            data = data.to(self.device)
            labels = labels.to(self.device)
            y_test_pred, _, _, _, _ = self.net(data)
            test_loss = self.loss_fn(y_test_pred, labels)
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
            test_loss_total.append(test_loss.item())
            y_test_pre_labels.append(torch.argmax(y_test_pred, 1))
            y_test_labels.append(labels)

        # ====================================================
        #   Because 64 samples are loaded each time during testing, the predicted labels need to be spliced
        y_test_pre_labels_cat = 0
        y_test_labels_cat = 0
        for i in range(len(y_test_pre_labels)):
            if i == 0:
                y_test_pre_labels_cat = y_test_pre_labels[0]
                y_test_labels_cat = y_test_labels[0]
            else:
                y_test_pre_labels_cat = torch.cat((y_test_pre_labels_cat, y_test_pre_labels[i]), dim=0)
                y_test_labels_cat = torch.cat((y_test_labels_cat, y_test_labels[i]), dim=0)
        # ====================================================
        test_accuracy = round(acc / float(len(target_test_dataloader)), 4)
        test_loss = np.mean(test_loss_total)
        self.train_hist['Tar_test_loss'].append(test_loss)
        self.train_hist['Tar_test_accuracy'].append(test_accuracy)
        print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, self.adv_epochs, test_accuracy))
        best_model = self.net
        if test_accuracy >= best_acc:
            best_acc = test_accuracy
            self.best_pre_labels = y_test_pre_labels_cat
            self.y_test_ori = y_test_labels_cat
            save_model(model=self.net, save_dir='save_dir/', model_name='Lightweight_model',
                       source_data_name=self.source_data_name, target_data_name=self.target_data_name)
            best_model = self.net
        self.net.train()
        return best_acc, self.best_pre_labels, self.y_test_ori, best_model