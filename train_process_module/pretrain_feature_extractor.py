"""
Author:  Qi tong Chen
Date: 2024.08.20
Pre-train feature extractor
"""
import torch
import dataloader
from MMD.MMD_calculation import MMDLoss
from models.Lightweight_res_net import Lightweight_model


def pretrain_process_fn(net=None, batch_size=64, pretrain_epochs=5, device='cuda:0',
                        loss_fn=torch.nn.CrossEntropyLoss(), source_data_name=''):
    """
        Pre-training feature extractor using source domain datasets.
        :param net:model
        :param batch_size:
        :param pretrain_epochs:
        :param device:
        :param loss_fn:
        :param source_data_name:
        :param target_data_name:
        :return:
        """
    pretrain_optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    source_train_dataloader = dataloader.source_dataload(batch_size=batch_size, source_data_name=source_data_name)
    source_test_dataloader = dataloader.source_dataload(batch_size=batch_size, source_data_name=source_data_name)
    for epoch in range(pretrain_epochs):
        # Training
        net.train()
        for i_iter, (data, labels) in enumerate(source_train_dataloader):
            data = data.to(device)
            labels = labels.to(device)
            pretrain_optimizer.zero_grad()
            y_pred, _, _, _, _ = net(data)  # prediction
            loss = loss_fn(y_pred, labels)
            loss.requires_grad_(True)
            loss.backward()  # back propagation
            pretrain_optimizer.step()  # update the gradient
        # Testing
        net.eval()
        acc = 0
        for data, labels in source_test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_source_test_pred, _, _, _, _ = net(data)
            acc += (torch.max(y_source_test_pred, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(source_test_dataloader)), 3)

        print("step1----Epoch %d/%d  accuracy: %.3f " % (epoch+1, pretrain_epochs, accuracy))


if __name__ == '__main__':
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    net = Lightweight_model(class_num=4)
    net.to('cuda:0')
    source_data_name = 'CWRU_1hp_4'
    pretrain_process_fn(net=net, batch_size=64, pretrain_epochs=5, device=device,
                        source_data_name=source_data_name)