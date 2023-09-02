import numpy as np
import argparse
import pickle
import os
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from encoder import StackedEncoders
from decoder import StackedDecoders
from utils.utils import *



class Ladder(torch.nn.Module):

    def __init__(self):
        super(Ladder, self).__init__()
        self.se = StackedEncoders()
        self.de = StackedDecoders()
    def forward_encoders_clean(self, data):
        return self.se.forward_clean(data)
    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    # def get_encoders_tilde_z(self, reverse=True):
    #     return self.se.get_encoders_tilde_z(reverse)
    def get_encoders_z_pre(self):
        return self.se.get_encoders_z_pre()
    def forward_decoders(self, data):
        return self.de.forward(data)


def evaluate_performance(ladder, valid_loader):
    correct = AverageMeter()
    total = AverageMeter()
    for batch_idx, (data, target) in enumerate(valid_loader):
        data = data.cuda()
        data, target = Variable(data), Variable(target)
        output = ladder.forward_encoders_clean(data)

        output = output.cpu()
        target = target.cpu()

        output = output.data.numpy()
        preds = np.argmax(output, axis=1)
        target = target.data.numpy()
        correct.update(np.sum(target == preds))
        total.update(target.shape[0])
    return  correct.sum / total.sum
    # print("Validation Accuracy:", correct.sum / total.sum)



def main():
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument("--s_batch", type=int, default=64)
    parser.add_argument("--u_batch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cuda", type=bool, default=True)


    args = parser.parse_args()
    s_size = args.s_batch
    u_size = args.u_batch
    epochs = args.epochs
    noise_std = args.noise_std



    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.\n")
        args.cuda = False


    print("=====================")
    print("EPOCHS:",epochs)
    print("NOISE STD:", noise_std)
    print("CUDA:", args.cuda)
    print("=====================\n")

    # 加载数据
    # 标签数据
    train_labelled_images_filename = os.path.join(args.data_dir, "train_labelled_images.p")
    train_labelled_labels_filename = os.path.join(args.data_dir, "train_labelled_labels.p")
    # 无标签数据
    train_unlabelled_images_filename = os.path.join(args.data_dir, "train_unlabelled_images.p")
    train_unlabelled_labels_filename = os.path.join(args.data_dir, "train_unlabelled_labels.p")
    # 验证集数据
    validation_images_filename = os.path.join(args.data_dir, "validation_images.p")
    validation_labels_filename = os.path.join(args.data_dir, "validation_labels.p")

    print("Loading Data")

    with open(train_labelled_images_filename, 'rb') as f:
        train_labelled_images = pickle.load(f)
    train_labelled_images = train_labelled_images.reshape(train_labelled_images.shape[0], 1, 28, 28)
    with open(train_labelled_labels_filename, 'rb') as f:
        train_labelled_labels = pickle.load(f).astype(int)

    with open(train_unlabelled_images_filename, 'rb') as f:
        train_unlabelled_images = pickle.load(f)
    train_unlabelled_images = train_unlabelled_images.reshape(train_unlabelled_images.shape[0], 1, 28, 28)

    with open(train_unlabelled_labels_filename, 'rb') as f:
        train_unlabelled_labels = pickle.load(f).astype(int)

    with open(validation_images_filename, 'rb') as f:
        validation_images = pickle.load(f)
    validation_images = validation_images.reshape(validation_images.shape[0],1, 28,28)
    with open(validation_labels_filename, 'rb') as f:
        validation_labels = pickle.load(f).astype(int)

    print("有标签数据量：",train_labelled_images.shape[0])
    print("无标签数据量：",train_unlabelled_images.shape[0])


    #数据 以及 验证集
    labelled_dataset = TensorDataset(torch.FloatTensor(train_labelled_images), torch.LongTensor(train_labelled_labels))
    labelled_loader = DataLoader(labelled_dataset, batch_size=s_size, shuffle=True)
    unlabelled_dataset = TensorDataset(torch.FloatTensor(train_unlabelled_images), torch.LongTensor(train_unlabelled_labels))
    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=u_size, shuffle=True)
    validation_dataset = TensorDataset(torch.FloatTensor(validation_images), torch.LongTensor(validation_labels))
    validation_loader = DataLoader(validation_dataset)

    # Configure the Ladder
    starter_lr = 0.002  # 初始学习率
    ladder = Ladder().cuda()
    optimizer = Adam(ladder.parameters(), lr=starter_lr)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()

    labeled_iter = iter(labelled_loader)
    unlabeled_iter = iter(unlabelled_loader)

    text = tqdm(range(epochs))

    for current_epoch in range(epochs):

        agg_cost = AverageMeter()
        agg_supervised_cost = AverageMeter()
        agg_unsupervised_cost = AverageMeter()
        ladder.train()

        try:
            inputs_x, targets_x = labeled_iter.__next__()
        except:
            labeled_iter = iter(labelled_loader)
            inputs_x, targets_x = labeled_iter.__next__()
        try:
            inputs_u, _ = unlabeled_iter.__next__()
        except:
            unlabeled_iter = iter(unlabelled_loader)
            inputs_u, _ = unlabeled_iter.__next__()

        optimizer.zero_grad()
        inputs_x = inputs_x.cuda()
        targets_x = targets_x.cuda()
        inputs_u = inputs_u.cuda()

        output_noise_labelled = ladder.forward_encoders_noise(inputs_x)#通过了加了noise的encode的label预测结果
        cost_supervised = loss_supervised(output_noise_labelled, targets_x)  # 有监督

        cost_unsupervised = 0.  # 无监督
        _ = ladder.forward_encoders_clean(inputs_u)#获取无噪声encode的输出预测
        z_pre_layers_unlabelled = ladder.get_encoders_z_pre()
        _ = ladder.forward_encoders_noise(inputs_u)#获取噪声encode的输入列表
        decode_input_data = ladder.se.decode_input
        hat_z_layers_unlabelled = ladder.forward_decoders(decode_input_data) #获取decode的每一层输出列表

        for z, hat_z in zip(z_pre_layers_unlabelled, hat_z_layers_unlabelled):
            c = loss_unsupervised(hat_z, z)
            cost_unsupervised += c

        loss = cost_supervised + 0.1 * cost_unsupervised

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        agg_cost.update(loss.item())
        agg_supervised_cost.update(cost_supervised.item())
        agg_unsupervised_cost.update(cost_unsupervised.item())


        ladder.eval()


        text.set_description(
            "Train Epoch: {epoch}/{epochs:4}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f} accuracy: {accuracy:.4f}. ".format(
                epoch=current_epoch + 1,
                epochs=epochs,
                loss=agg_cost.avg,
                loss_x=agg_supervised_cost.avg,
                loss_u=agg_unsupervised_cost.avg,
                accuracy = evaluate_performance(ladder, validation_loader),
                ))
        text.update()
    text.close()
    print("=====================\n")
    print("Done")


if __name__ == "__main__":
    main()
