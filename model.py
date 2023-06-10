import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ecg_dataset as wd
import metrics
import usst_rest_exercise_data
# np.set_printoptions(threshold=np.inf)  # Full output


# Create ECG dataset
class ECGDataset(torch.utils.data.Dataset):
    """
    Function: return x, y (is train or test data depend on "is_train_data" is True or not)\n
    data_use: 0 represent use for training, 1 represent use for test,
              2 represent use for generate vector, 3 represent use for valid
    """
    def __init__(self, data_set, data_use=0, feature_transform=None, target_transform=None):
        x, y = data_set
        self.data_length = len(x[0])
        self.data_use = data_use
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.vector_x = []
        self.vector_y = []
        self.valid_x = []
        self.valid_y = []
        for index in range(len(x)):
            if index % 10 == 0:
                self.test_x.append(x[index])
                self.test_y.append(y[index])
            elif index % 10 in [2, 7]:
                self.valid_x.append(x[index])
                self.valid_y.append(y[index])
            else:
                self.train_x.append(x[index])
                self.train_y.append(y[index])
        self.feature_transform = feature_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = None
        y = None
        if self.data_use == 0:
            x = self.train_x[index].reshape(1, -1)
            x = torch.Tensor(x)
            y = self.train_y[index]
        elif self.data_use == 1:
            x = self.test_x[index].reshape(1, -1)
            x = torch.Tensor(x)
            y = self.test_y[index]
        elif self.data_use == 2:
            x = self.vector_x[index].reshape(1, -1)
            x = torch.Tensor(x)
            y = self.vector_y[index]
        elif self.data_use == 3:
            x = self.valid_x[index].reshape(1, -1)
            x = torch.Tensor(x)
            y = self.valid_y[index]
        if self.feature_transform is not None:
            x = self.feature_transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        if self.data_use == 0:
            return len(self.train_x)
        elif self.data_use == 1:
            return len(self.test_x)
        elif self.data_use == 2:
            return len(self.vector_x)
        elif self.data_use == 3:
            return len(self.valid_x)


class ECGNet(nn.Module):
    def __init__(self, device=None):
        super(ECGNet, self).__init__()
        self.device = device
        self.PQ_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(32, 128, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
            nn.Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            # SMP, will be added in "network" function
        )
        self.QRS_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(32, 128, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
            nn.Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.ST_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(32, 128, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
            nn.Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            # SMP, will be added in "network" function
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(self.output_size(), 256),
        )

    def SMP(self, x, m):
        # Calculate the half of padding, will add on the two side evenly
        padding1 = math.floor((math.ceil(x.size(2) / m) * m - x.size(2) + 1) / 2)
        pad_func = torch.nn.ZeroPad2d((padding1, padding1, 0, 0))  # left right top bottom
        x = pad_func(x)
        kernel_size = math.floor(x.size(2) / m)
        pool = nn.MaxPool1d(kernel_size=(kernel_size,), stride=(kernel_size,), padding=0)
        x = pool(x)
        return x

    def network(self, x):
        # The shape of x: [1, 1, n], n is the length
        t1 = 250
        t2 = 380
        f = 500
        PQ_len = int(x.size(2) * t1 / (t1 + t2) - 0.05 * f)
        QRS_len = int(0.1 * f)
        ST_len = int(x.size(2) * t2 / (t1 + t2) - 0.05 * f)
        PQ = x[:, :, : PQ_len]
        QRS = x[:, :, PQ_len: PQ_len + QRS_len]
        ST = x[:, :, PQ_len + QRS_len:]
        PQ = self.PQ_conv(PQ)
        QRS = self.QRS_conv(QRS)
        ST = self.ST_conv(ST)
        PQ = self.SMP(PQ, 8)
        ST = self.SMP(ST, 12)
        x = torch.cat([PQ, QRS, ST], dim=2)
        x = self.final_conv(x)
        return x

    def output_size(self):
        # 1000 is a random number because the output size is not relate to it
        x = torch.FloatTensor(1, 1, 1000)
        x = self.network(x)
        # flatten
        x = x.view(1, x.size(1) * x.size(2))
        return x.size(1)

    def forward(self, x):
        logits = []
        # Because length is variant, we only pass one input each time
        for i in range(len(x)):
            y = x[i].unsqueeze(0)
            if self.device is not None:
                y = y.to(self.device)
            y = self.network(y)
            # flatten
            y = y.view(1, y.size(1) * y.size(2))
            # fully connected
            logit = self.fc_unit(y)
            logits.append(logit)
        logits = torch.cat(logits, dim=0)
        return logits


class ECGNetTradition1(nn.Module):
    def __init__(self, sampling_rate, device=None):
        super(ECGNetTradition1, self).__init__()
        self.sampling_rate = sampling_rate
        self.device = device
        self.PQ_conv_unit = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(5,), stride=(3,)),
            nn.Conv1d(32, 128, kernel_size=(5,), stride=(3,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.QRS_conv_unit = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(32, 128, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.ST_conv_unit = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(7,), stride=(5,)),
            nn.Conv1d(32, 128, kernel_size=(7,), stride=(5,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.conv_unit = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
            # nn.Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding="same"),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.PQ_length = int(200 * self.sampling_rate / 1000)  # 200ms
        self.QRS_length = int(100 * self.sampling_rate / 1000)  # 100ms
        self.ST_length = int(330 * self.sampling_rate / 1000)  # 330ms
        # flatten_size depends on the structure of network
        self.flatten_size = self.output_size()
        self.fc_unit = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
        )

    def output_size(self):
        x = torch.FloatTensor(1, 1, self.PQ_length + self.QRS_length + self.ST_length)
        x = self.SMP(x)
        x = self.conv_unit(x)
        # flatten
        x = x.view(1, x.size(1) * x.size(2))
        return x.size(1)

    def SMP(self, x):
        PQ = x[:, :, : self.PQ_length]
        QRS = x[:, :, self.PQ_length: self.PQ_length + self.QRS_length]
        ST = x[:, :, self.PQ_length + self.QRS_length:]

        PQ = self.PQ_conv_unit(PQ)
        QRS = self.QRS_conv_unit(QRS)
        ST = self.ST_conv_unit(ST)

        ecg = torch.cat([PQ, QRS, ST], dim=2)
        return ecg

    def forward(self, x):
        batch_size = x.size(0)
        x = self.SMP(x)
        x = self.conv_unit(x)
        # flatten
        x = x.view(batch_size, x.size(1) * x.size(2))
        # fully connected
        logits = self.fc_unit(x)
        return logits


class ECGNetTradition2(nn.Module):
    def __init__(self, sampling_rate, device=None):
        super(ECGNetTradition2, self).__init__()
        self.sampling_rate = sampling_rate
        self.device = device
        self.PQ_conv_unit = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(3,), stride=(1,)),
            nn.Conv1d(32, 128, kernel_size=(3,), stride=(1,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.QRS_conv_unit = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(32, 128, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.ST_conv_unit = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(3,), stride=(1,)),
            nn.Conv1d(32, 128, kernel_size=(3,), stride=(1,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.conv_unit = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=(3,), stride=(1,), padding="same"),
            nn.Conv1d(64, 32, kernel_size=(3,), stride=(1,), padding="same"),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
            # nn.Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding="same"),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=0),
        )
        self.PQ_length = int(200 * self.sampling_rate / 1000)  # 200ms
        self.QRS_length = int(100 * self.sampling_rate / 1000)  # 100ms
        self.ST_length = int(330 * self.sampling_rate / 1000)  # 330ms
        # flatten_size depends on the structure of network
        self.flatten_size = self.output_size()
        self.fc_unit = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
        )

    def output_size(self):
        x = torch.FloatTensor(1, 1, self.PQ_length + self.QRS_length + self.ST_length)
        x = self.SMP(x)
        x = self.conv_unit(x)
        # flatten
        x = x.view(1, x.size(1) * x.size(2))
        return x.size(1)

    def SMP(self, x):
        PQ = x[:, :, : self.PQ_length]
        QRS = x[:, :, self.PQ_length: self.PQ_length + self.QRS_length]
        ST = x[:, :, self.PQ_length + self.QRS_length:]

        PQ = self.PQ_conv_unit(PQ)
        QRS = self.QRS_conv_unit(QRS)
        ST = self.ST_conv_unit(ST)

        ecg = torch.cat([PQ, QRS, ST], dim=2)
        return ecg

    def forward(self, x):
        batch_size = x.size(0)
        x = self.SMP(x)
        x = self.conv_unit(x)
        # flatten
        x = x.view(batch_size, x.size(1) * x.size(2))
        # fully connected
        logits = self.fc_unit(x)
        return logits


def train_model(dataset, feature_extractor_path, num_class, sampling_rate):
    batch_size = 128

    train_data = ECGDataset(dataset)
    test_data = ECGDataset(dataset, 1)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=non_fixed_collate_fn)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=non_fixed_collate_fn)
    # train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Network and metrics
    device = torch.device("cuda")
    # net = ECGNetTradition1(sampling_rate)
    net = ECGNet(device)
    net = net.to(device)
    print(net)
    # metric_fc = metrics.AddMarginProduct(256, num_class, s=30, m=0.35)
    # metric_fc = metrics.SphereProduct(256, num_class, m=4)
    metric_fc = metrics.ArcMarginProduct(256, num_class, s=30, m=0.5, easy_margin=False)
    metric_fc.to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': metric_fc.parameters()}], lr=1e-4)

    # for plot
    loss_plot = []
    train_acc_plot = []
    test_acc_plot = []

    """
    ===== How to iterate train data =====
    it = iter(train_data)
    feature, label = it.next()
    print("feature length:", len(feature), "label length:", len(label))
    print("feature:", feature, "label: ", label)
    """

    epoch = -1
    max_acc = -1
    counter_of_less = 0
    while True:
        epoch += 1
        loss = None
        net.train()
        total_correct = 0
        total_num = 0
        for batch_index, (feature, label) in enumerate(train_data):
            if net.device is None:
                feature = feature.to(device)
            label = label.to(device)
            logits = net(feature)
            pred = metric_fc.pred(logits).argmax(dim=1)
            output = metric_fc(logits, label.long())
            loss = criteon(output, label.long())

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += len(label)
        print("epoch:", epoch, "loss:", loss.item())
        loss_plot.append(loss.item())
        train_acc = total_correct / total_num
        print("epoch:", epoch, "train_acc: ", train_acc)
        train_acc_plot.append(train_acc)

        net.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for feature, label in test_data:
                # feature, label = feature.to(device), label.to(device)
                # "feature.to(device)" in non_fixed input model is executed in "forward" function
                label = label.to(device)
                logits = net(feature)
                # output = metric_fc(logits, label.long())
                output = metric_fc.pred(logits)
                pred = output.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += len(feature)

            test_acc = total_correct / total_num

            print("epoch:", epoch, "test_acc: ", test_acc)
            test_acc_plot.append(test_acc)
            if test_acc > max_acc:
                max_acc = test_acc
                counter_of_less = 0
                torch.save(net, feature_extractor_path)
            else:
                counter_of_less += 1
            if counter_of_less == 20:
                break

    print("metric_fc.weight:", metric_fc.weight)
    print("metric_fc size:", metric_fc.weight.size())
    np_metric_fc = metric_fc.weight.data.cpu().numpy()
    np.save(feature_extractor_path.split(".")[0] + "_metric_fc.npy", np_metric_fc)
    # plot
    plt.figure(figsize=(12, 6), dpi=80)
    plt.plot(train_acc_plot, label="train_acc")
    plt.plot(test_acc_plot, label="test_acc")
    plt.xticks(range(epoch + 1))
    plt.legend()
    plt.show()
    plt.plot(loss_plot, label="loss")
    plt.xticks(range(epoch + 1))
    plt.legend()
    plt.show()
    print(max(test_acc_plot))


def create_feature_vector(feature, feature_extractor, use_cpu=False):
    device = torch.device("cuda")
    if use_cpu:
        device = torch.device("cpu")
    feature = feature.reshape([1, 1, -1])
    feature = torch.FloatTensor(feature)
    feature = feature.to(device)
    feature_extractor.device = None
    vector = feature_extractor(feature)
    return vector


def final_layer(one_vector, weight):
    cos_theta = F.linear(F.normalize(torch.reshape(one_vector, (1, -1))),
                         F.normalize(weight))
    cos_theta = cos_theta[0]
    # print(cos_theta)
    # scale = torch.tan(cos_theta * (math.pi / 2 - 0.5))
    # scale = cos_theta
    # scale = cos_theta ** 3
    scale = torch.FloatTensor([math.exp(i) for i in cos_theta])
    # print(scale)
    importance = torch.softmax(scale, 0)
    # print(importance)
    feature_vector = torch.zeros(one_vector.size())
    for imp_index in range(len(importance)):
        feature_vector = feature_vector + importance[imp_index] * weight[imp_index]
    return feature_vector


def valid_model(dataset, feature_extractor_path, shot_plot=True):
    ecg_dataset = ECGDataset(dataset, 0)
    x1, y1 = ecg_dataset.train_x, ecg_dataset.train_y
    x2, y2 = ecg_dataset.valid_x, ecg_dataset.valid_y

    # Acquire the stop index of each class(train data)
    t_target_first_index = []
    t_last_target = -1
    for i in range(len(y1)):
        if y1[i] != t_last_target:
            t_last_target = y1[i]
            t_target_first_index.append(i)
    t_target_stop_index = t_target_first_index[1:]
    t_target_stop_index.append(len(y1))

    weight = np.load(feature_extractor_path.split(".")[0] + "_metric_fc.npy")
    # device = torch.device("cuda")
    weight = torch.FloatTensor(weight)
    # weight = weight.to(device)
    feature_extractor = torch.load(feature_extractor_path).to(device="cpu")
    last_index = 0
    feature_vectors = {}
    for row, next_index in enumerate(t_target_stop_index):
        one_class_feature_vectors = []
        for index in range(last_index, next_index, 1):
            one_vector = create_feature_vector(x1[index], feature_extractor, use_cpu=True)
            feature_vector = final_layer(one_vector, weight)
            # print(feature_vector)
            # print(feature_vector.shape)
            feature_vector = torch.Tensor.cpu(feature_vector[0])
            feature_vector = feature_vector.detach().numpy()
            one_class_feature_vectors.append(feature_vector)
        last_index = next_index
        feature_vectors[row] = one_class_feature_vectors

    # construct new weight use mean
    final_weight = []
    for k in feature_vectors:
        final_weight.append(np.mean(np.array(feature_vectors[k]), 0))
    final_weight = np.array(final_weight)
    # print(final_weight)
    print("final_weight", final_weight.shape)

    # Acquire the stop index of each class(valid data)
    v_target_first_index = []
    v_last_target = -1
    for i in range(len(y2)):
        if y2[i] != v_last_target:
            v_last_target = y2[i]
            v_target_first_index.append(i)
    v_target_stop_index = v_target_first_index[1:]
    v_target_stop_index.append(len(y2))

    # Vaild result
    """ matrix
    true\predict 1   2   3   4   ...
    1
    2
    3
    4
    .
    .
    .
    """
    matrix_size = len(v_target_stop_index)
    matrix = np.zeros((matrix_size, matrix_size))
    row_sum = np.zeros((matrix_size,))
    column_sum = np.zeros((matrix_size,))
    total_sum = 0

    last_index = 0
    for row, next_index in enumerate(v_target_stop_index):
        for index in range(last_index, next_index, 1):
            one_vector = create_feature_vector(x2[index], feature_extractor, use_cpu=True)
            one_vector = final_layer(one_vector, weight)
            cos_theta = F.linear(F.normalize(torch.reshape(one_vector, (1, -1))),
                                 F.normalize(torch.FloatTensor(final_weight)))
            test_output = torch.argmax(cos_theta)
            matrix[row, test_output] += 1
        last_index = next_index

    # matrix, row_sum, column_sum, total_sum
    print("matrix:")
    print(matrix)

    """ plot gray picture about matrix """
    if shot_plot:
        # plt.matshow(matrix, cmap=plt.cm.gray)
        plt.matshow(matrix)

    row_sum = np.sum(matrix, axis=1)
    err_matrix = matrix / row_sum
    np.fill_diagonal(err_matrix, 0)
    if shot_plot:
        # plt.matshow(err_matrix, cmap=plt.cm.gray)
        plt.matshow(err_matrix)
    """ ------------------------------ """

    for i in range(matrix_size):
        row_sum[i] = sum(matrix[i, :])
        column_sum[i] = sum(matrix[:, i])
    total_sum = sum(row_sum)
    print("row_sum:", row_sum)
    print("column_sum:", column_sum)
    print("Is total_sum from row_sum equal to total_sum from column_sum:", sum(row_sum) == sum(column_sum))
    print("total_sum:", total_sum)
    # mean_accuracy
    tmp = 0
    for i in range(matrix_size):
        tmp += matrix[i, i]
    mean_accuracy = tmp / total_sum
    print("mean_accuracy:", mean_accuracy)
    # mean precision
    tmp = 0
    for i in range(matrix_size):
        if column_sum[i] == 0:
            tmp += 1
        else:
            tmp += matrix[i, i] / column_sum[i]
    mean_precision = tmp / matrix_size
    print("mean_precision:", mean_precision)
    # mean recall
    tmp = 0
    for i in range(matrix_size):
        if column_sum[i] == 0:
            tmp += 1
        else:
            tmp += matrix[i, i] / row_sum[i]
    mean_recall = tmp / matrix_size
    print("mean_recall:", mean_recall)
    # F1
    F1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
    print("F1:", F1)
    # kappa
    tmp = 0
    for a, b in zip(row_sum, column_sum):
        tmp += a * b
    pe = tmp / (total_sum * total_sum)
    po = mean_accuracy
    kappa = (po - pe) / (1 - pe)
    print("kappa:", kappa)

    F1_for_plot_hist = []  # collect all acc to plot hist
    num_F1_greater_than_0dot5 = 0
    for i in range(matrix_size):
        if column_sum[i] == 0:
            single_precision = 1
        else:
            single_precision = matrix[i, i] / column_sum[i]
        if row_sum[i] == 0:
            single_recall = 1
        else:
            single_recall = matrix[i, i] / row_sum[i]
        # single_precision = matrix[i, i] / column_sum[i]
        # single_recall = matrix[i, i] / row_sum[i]
        single_F1 = 2 * single_precision * single_recall / (single_precision + single_recall)
        # print(i, single_F1)
        F1_for_plot_hist.append(single_F1)
        if single_F1 >= 0.5:
            num_F1_greater_than_0dot5 += 1

    print("The number of F1 >= 0.5: ", num_F1_greater_than_0dot5)
    # -------------------------------start
    # bins = 10
    plt.figure(figsize=(10, 6), dpi=60)
    bins = 20
    # ---------------------------------end
    # -------------------------------start
    # plt.hist(F1_for_plot_hist, range=(0.5, 1), bins=bins, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(F1_for_plot_hist, range=(0, 1), bins=bins, facecolor="blue", edgecolor="black", alpha=0.7)
    # ---------------------------------end
    # number of class hist
    # -------------------------------start
    # plt.xticks([i / 100 for i in range(50, 100, 5)] + [1])
    plt.xticks([i / 100 for i in range(0, 100, 5)] + [1])
    # ---------------------------------end
    # percentage of class hist
    # plt.yticks([len(acc_for_plot_hist) / 5 * i for i in range(6)], [i for i in range(0, 101, 20)])
    plt.xlabel('F1', fontsize=14)
    plt.ylabel('rate(%)', fontsize=14)
    if shot_plot:
        plt.show()
    return mean_accuracy, mean_precision, mean_recall, F1, kappa


"""=================== Function for experiment ==================="""
ecg_id = wd.load_ecg_id(True)
mit_db = wd.load_mit_db()
mit_db_re = wd.load_mit_db_resample()
usst_db = wd.load_usst_db_filtering()

rest_usst_db = usst_rest_exercise_data.load_usst_db_filtering("./xxx_rest_usst_db_ecgdata.npy",
                                                                  "./xxx_rest_usst_db_target.npy")
exercise_usst_db = usst_rest_exercise_data.load_usst_db_filtering("./xxx_exercise_usst_db_ecgdata.npy",
                                                                      "./xxx_exercise_usst_db_target.npy")


def valid_rest_train_exercise_test(rest_usst_db, exercise_usst_db, feature_extractor_path, shot_plot=True):
    rest_usst_db = ECGDataset(rest_usst_db, 0)
    exercise_usst_db = ECGDataset(exercise_usst_db, 0)
    x1, y1 = rest_usst_db.train_x, rest_usst_db.train_y
    x2, y2 = exercise_usst_db.train_x, exercise_usst_db.train_y

    # Acquire the stop index of each class(train data)
    t_target_first_index = []
    t_last_target = -1
    for i in range(len(y1)):
        if y1[i] != t_last_target:
            t_last_target = y1[i]
            t_target_first_index.append(i)
    t_target_stop_index = t_target_first_index[1:]
    t_target_stop_index.append(len(y1))

    weight = np.load(feature_extractor_path.split(".")[0] + "_metric_fc.npy")
    # device = torch.device("cuda")
    weight = torch.FloatTensor(weight)
    # weight = weight.to(device)
    feature_extractor = torch.load(feature_extractor_path).to(device="cpu")
    last_index = 0
    feature_vectors = {}
    for row, next_index in enumerate(t_target_stop_index):
        one_class_feature_vectors = []
        for index in range(last_index, next_index, 1):
            one_vector = create_feature_vector(x1[index], feature_extractor, use_cpu=True)
            feature_vector = final_layer(one_vector, weight)
            # print(feature_vector)
            # print(feature_vector.shape)
            feature_vector = torch.Tensor.cpu(feature_vector[0])
            feature_vector = feature_vector.detach().numpy()
            one_class_feature_vectors.append(feature_vector)
        last_index = next_index
        feature_vectors[row] = one_class_feature_vectors

    # construct new weight use mean
    final_weight = []
    for k in feature_vectors:
        final_weight.append(np.mean(np.array(feature_vectors[k]), 0))
    final_weight = np.array(final_weight)
    # print(final_weight)
    print("final_weight", final_weight.shape)

    # Acquire the stop index of each class(valid data)
    v_target_first_index = []
    v_last_target = -1
    for i in range(len(y2)):
        if y2[i] != v_last_target:
            v_last_target = y2[i]
            v_target_first_index.append(i)
    v_target_stop_index = v_target_first_index[1:]
    v_target_stop_index.append(len(y2))

    # Vaild result
    """ matrix
    true\predict 1   2   3   4   ...
    1
    2
    3
    4
    .
    .
    .
    """
    matrix_size = len(v_target_stop_index)
    matrix = np.zeros((matrix_size, matrix_size))
    row_sum = np.zeros((matrix_size,))
    column_sum = np.zeros((matrix_size,))
    total_sum = 0

    last_index = 0
    for row, next_index in enumerate(v_target_stop_index):
        for index in range(last_index, next_index, 1):
            one_vector = create_feature_vector(x2[index], feature_extractor, use_cpu=True)
            one_vector = final_layer(one_vector, weight)
            cos_theta = F.linear(F.normalize(torch.reshape(one_vector, (1, -1))),
                                 F.normalize(torch.FloatTensor(final_weight)))
            test_output = torch.argmax(cos_theta)
            matrix[row, test_output] += 1
        last_index = next_index

    # matrix, row_sum, column_sum, total_sum
    print("matrix:")
    print(matrix)

    """ plot gray picture about matrix """
    if shot_plot:
        plt.matshow(matrix, cmap=plt.cm.gray)

    row_sum = np.sum(matrix, axis=1)
    err_matrix = matrix / row_sum
    np.fill_diagonal(err_matrix, 0)
    if shot_plot:
        plt.matshow(err_matrix, cmap=plt.cm.gray)
    """ ------------------------------ """

    for i in range(matrix_size):
        row_sum[i] = sum(matrix[i, :])
        column_sum[i] = sum(matrix[:, i])
    total_sum = sum(row_sum)
    print("row_sum:", row_sum)
    print("column_sum:", column_sum)
    print("Is total_sum from row_sum equal to total_sum from column_sum:", sum(row_sum) == sum(column_sum))
    print("total_sum:", total_sum)
    # mean_accuracy
    tmp = 0
    for i in range(matrix_size):
        tmp += matrix[i, i]
    mean_accuracy = tmp / total_sum
    print("mean_accuracy:", mean_accuracy)
    # mean precision
    tmp = 0
    for i in range(matrix_size):
        if column_sum[i] == 0:
            tmp += 1
        else:
            tmp += matrix[i, i] / column_sum[i]
    mean_precision = tmp / matrix_size
    print("mean_precision:", mean_precision)
    # mean recall
    tmp = 0
    for i in range(matrix_size):
        if column_sum[i] == 0:
            tmp += 1
        else:
            tmp += matrix[i, i] / row_sum[i]
    mean_recall = tmp / matrix_size
    print("mean_recall:", mean_recall)
    # F1
    F1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
    print("F1:", F1)
    # kappa
    tmp = 0
    for a, b in zip(row_sum, column_sum):
        tmp += a * b
    pe = tmp / (total_sum * total_sum)
    po = mean_accuracy
    kappa = (po - pe) / (1 - pe)
    print("kappa:", kappa)

    F1_for_plot_hist = []  # collect all acc to plot hist
    num_F1_greater_than_0dot5 = 0
    for i in range(matrix_size):
        if column_sum[i] == 0:
            single_precision = 1
        else:
            single_precision = matrix[i, i] / column_sum[i]
        if row_sum[i] == 0:
            single_recall = 1
        else:
            single_recall = matrix[i, i] / row_sum[i]
        # single_precision = matrix[i, i] / column_sum[i]
        # single_recall = matrix[i, i] / row_sum[i]
        single_F1 = 2 * single_precision * single_recall / (single_precision + single_recall)
        # print(i, single_F1)
        F1_for_plot_hist.append(single_F1)
        if single_F1 >= 0.5:
            num_F1_greater_than_0dot5 += 1

    print("The number of F1 >= 0.5: ", num_F1_greater_than_0dot5)
    # -------------------------------start
    # bins = 10
    plt.figure(figsize=(10, 6), dpi=60)
    bins = 20
    # ---------------------------------end
    # -------------------------------start
    # plt.hist(F1_for_plot_hist, range=(0.5, 1), bins=bins, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(F1_for_plot_hist, range=(0, 1), bins=bins, facecolor="blue", edgecolor="black", alpha=0.7)
    # ---------------------------------end
    # number of class hist
    # -------------------------------start
    # plt.xticks([i / 100 for i in range(50, 100, 5)] + [1])
    plt.xticks([i / 100 for i in range(0, 100, 5)] + [1])
    # ---------------------------------end
    # percentage of class hist
    # plt.yticks([len(acc_for_plot_hist) / 5 * i for i in range(6)], [i for i in range(0, 101, 20)])
    plt.xlabel('F1', fontsize=14)
    plt.ylabel('rate(%)', fontsize=14)
    if shot_plot:
        plt.show()
    return mean_accuracy, mean_precision, mean_recall, F1, kappa


def rest_train_exercise_test():
    # train_model(rest_usst_db, "./xxx_rest_extractor.pt", 60, 500)
    valid_rest_train_exercise_test(rest_usst_db, exercise_usst_db, "./xxx_rest_extractor.pt", False)


def both_train_test():
    bothx = []
    bothy = []
    rest_index = 0
    exer_index = 0
    rest_ecgdata, rest_target = rest_usst_db
    exer_ecgdata, exer_target = exercise_usst_db
    for i in range(60):
        while rest_index < len(rest_ecgdata) and rest_target[rest_index] == i:
            bothx.append(rest_ecgdata[rest_index])
            bothy.append(rest_target[rest_index])
            rest_index += 1
        while exer_index < len(exer_ecgdata) and exer_target[exer_index] == i:
            bothx.append(exer_ecgdata[exer_index])
            bothy.append(exer_target[exer_index])
            exer_index += 1

    dataset = (bothx, bothy)
    # train_model(dataset, "./xxx_both_extractor.pt", 60, 500)
    valid_model(dataset, "./xxx_both_extractor.pt", False)


def single_database():
    """------------------ Single database ------------------"""
    """ model training """
    train_model(ecg_id, "wx_save_10/ecg_id_extractor.pt", 90, 500)
    # train_model(mit_db, "wx_save_10/mit_db_extractor.pt", 48, 360)
    # train_model(mit_db_re, "wx_save_10/mit_db_resample_extractor.pt", 48, 500)
    # train_model(usst_db, "wx_save_10/usst_db_filtering_extractor.pt", 60, 500)
    """ model validation """
    valid_model(ecg_id, "wx_save_10/ecg_id_extractor.pt", True)
    # valid_model(mit_db, "wx_save_10/mit_db_extractor.pt", True)
    # valid_model(mit_db_re, "wx_save_10/mit_db_resample_extractor.pt", True)
    # valid_model(usst_db, "wx_save_10/usst_db_filtering_extractor.pt", True)


def add_new_class_inner():
    num_new_class = 3
    ecg_id_model_name = "wx_save_10/ecg_id_extractor_0_" + str(90 - num_new_class) + ".pt"
    mit_db_model_name = "wx_save_10/mit_db_extractor_0_" + str(48 - num_new_class) + ".pt"
    usst_db_model_name = "wx_save_10/usst_db_extractor_0_" + str(60 - num_new_class) + ".pt"
    """------------------ add new class database(inner) ------------------"""
    # train_model(wd.load_dataset_with_start_end(ecg_id, 0, 90 - num_new_class),
    #             ecg_id_model_name, 90 - num_new_class, 500)
    # train_model(wd.load_dataset_with_start_end(mit_db, 0, 48 - num_new_class),
    #             mit_db_model_name, 48 - num_new_class, 360)
    # train_model(wd.load_dataset_with_start_end(usst_db, 0, 60 - num_new_class),
    #             usst_db_model_name, 60 - num_new_class, 500)
    """ add new class (total) """
    # valid_model(wd.load_dataset_with_start_end(ecg_id, 0, 90), ecg_id_model_name, True)
    # valid_model(wd.load_dataset_with_start_end(mit_db, 0, 48), mit_db_model_name, True)
    valid_model(wd.load_dataset_with_start_end(usst_db, 0, 60), usst_db_model_name, True)
    """ add new class (old) """
    # valid_model(wd.load_dataset_with_start_end(ecg_id, 0, 90 - num_new_class), ecg_id_model_name, True)
    # valid_model(wd.load_dataset_with_start_end(mit_db, 0, 48 - num_new_class), mit_db_model_name, True)
    valid_model(wd.load_dataset_with_start_end(usst_db, 0, 60 - num_new_class), usst_db_model_name, True)
    """ add new class (new) """
    # valid_model(wd.load_dataset_with_start_end(ecg_id, 90 - num_new_class, 90), ecg_id_model_name, True)
    # valid_model(wd.load_dataset_with_start_end(mit_db, 48 - num_new_class, 48), mit_db_model_name, True)
    valid_model(wd.load_dataset_with_start_end(usst_db, 60 - num_new_class, 60), usst_db_model_name, True)


def other_database_input(old_dataset, new_dataset, test_model, num_class):
    ta = tp = tr = tf = tk = 0
    counter = 0
    for i in range(0, num_class, 3):
        mix_data = wd.make_mix_data(old_dataset, wd.load_dataset_with_start_end(new_dataset, i, i + 3))
        counter += 1
        """ add new class (total) """
        # a, p, r, f, k = valid_model(mix_data, test_model, shot_plot=False)
        """ add new class (old) """
        # ta, tp, tr, tf, tk = valid_model(old_dataset, test_model, shot_plot=False)
        # break
        """ add new class (new) """
        a, p, r, f, k = valid_model(wd.load_dataset_with_start_end(new_dataset, i, i + 3), test_model, shot_plot=False)
        ta += a
        tp += p
        tr += r
        tf += f
        tk += k
        # if counter == 5:
        #     break
    ta /= counter
    tp /= counter
    tr /= counter
    tf /= counter
    tk /= counter
    print(ta, tp, tr, tf, tk)


def add_new_class_inter():
    ecg_id_model_name = "wx_save_10/ecg_id_extractor.pt"
    mit_db_resample_model_name = "wx_save_10/mit_db_resample_extractor.pt"
    usst_db_model_name = "wx_save_10/usst_db_filtering_extractor.pt"
    """------------------ add new class database(inter) ------------------"""
    """ use model acquired in single_database """
    # other_database_input(ecg_id, mit_db_re, ecg_id_model_name, 48)
    # other_database_input(ecg_id, usst_db, ecg_id_model_name, 60)

    # other_database_input(mit_db_re, ecg_id, mit_db_resample_model_name, 90)
    # other_database_input(mit_db_re, usst_db, mit_db_resample_model_name, 60)

    # other_database_input(usst_db, ecg_id, usst_db_model_name, 90)
    other_database_input(usst_db, mit_db_re, usst_db_model_name, 48)


def num_new_class_influence_helper(old_dataset, new_dataset, test_model, num_new_class):
    mix_data = wd.make_mix_data(old_dataset, wd.load_dataset_with_start_end(new_dataset, 0, num_new_class))
    a, p, r, f, k = valid_model(mix_data, test_model, shot_plot=False)
    print(a, p, r, f, k)
    return a, p, r, f, k


def plot_variation_diagram_for_num_new_class_influence(old_dataset, new_dataset, test_model):
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    x = np.array([1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
    # ta = []
    # tp = []
    # tr = []
    # tf = []
    # tk = []
    # for i in x:
    #     a, p, r, f, k = num_new_class_influence_helper(old_dataset, new_dataset, test_model, i)
    #     ta.append(a)
    #     tp.append(p)
    #     tr.append(r)
    #     tf.append(f)
    #     tk.append(k)

    ta = [0.9706375838926175, 0.9695222405271828, 0.9700889248181084, 0.9646278555637435, 0.956982131039047, 0.9526378896882494, 0.9326710816777042, 0.9290966386554622, 0.9248991935483871, 0.9161821705426356, 0.9083646616541353, 0.904391582799634, 0.9019607843137255]
    tp = [0.9613288152460562, 0.9603862419227526, 0.9611480449626997, 0.9577858895896773, 0.9503142533501253, 0.9489143000869261, 0.9350690218301315, 0.9306410223162394, 0.924874042957326, 0.9105235865551641, 0.9007961420028548, 0.8958563618825144, 0.8926325166868926]
    tr = [0.970701534971459, 0.969493483483605, 0.9700801472627665, 0.96615509699944, 0.960734485290522, 0.9574101639537267, 0.95037316866038, 0.9447196918131305, 0.9390769439003549, 0.9250219542640985, 0.9178241594861196, 0.9102130589299416, 0.9061834129131647]
    tf = [0.9659924405085891, 0.964918373838418, 0.9655934402286123, 0.9619522901289844, 0.9554959605122445, 0.9531433003787815, 0.9426589833143295, 0.9376275115152103, 0.931921381921451, 0.9177155115017075, 0.9092304327903952, 0.9029776487153162, 0.899356923783126]
    tk = [0.9699471914814839, 0.9688267308111379, 0.969426635012712, 0.9638835597449403, 0.956204304700139, 0.9518764151632458, 0.931713148517971, 0.9281672475737164, 0.9239847525163902, 0.9152279694141511, 0.9073793829399144, 0.9034117731871976, 0.9010001455880212]

    print(ta)
    print(tp)
    print(tr)
    print(tf)
    print(tk)

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 4))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(x, ta, marker='+', color="orange", label="Accuracy", linewidth=1.5)
    plt.plot(x, tp, marker='<', color="green", label="Precision", linewidth=1.5)
    plt.plot(x, tr, marker='o', color="red", label="Recall", linewidth=1.5)
    plt.plot(x, tf, marker='*', color="blue", label="F1-score", linewidth=1.5)
    plt.plot(x, tk, marker='v', color="purple", label="Kappa", linewidth=1.5)

    group_labels = x
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("The number of new classes", fontsize=13, fontweight='bold')
    # plt.ylabel("", fontsize=13, fontweight='bold')
    # plt.xlim(0.9, 6.1)  # 设置x轴的范围
    # plt.ylim(1.5, 16)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()


def num_new_class_influence():
    old_dataset = wd.make_mix_data(wd.load_dataset_with_start_end(ecg_id, 0, 45),
                                     wd.load_dataset_with_start_end(mit_db_re, 0, 24))
    old_dataset = wd.make_mix_data(old_dataset, wd.load_dataset_with_start_end(usst_db, 0, 30))

    new_dataset = wd.make_mix_data(wd.load_dataset_with_start_end(ecg_id, 45, 90),
                                    wd.load_dataset_with_start_end(mit_db_re, 24, 48))
    new_dataset = wd.make_mix_data(new_dataset, wd.load_dataset_with_start_end(usst_db, 30, 60))

    mix_model_name = "wx_save_10/mix_extractor.pt"
    # train_model(old_dataset, mix_model_name, 45 + 24 + 30, 500)
    """------------------ origin ------------------"""
    # valid_model(old_dataset, mix_model_name, False)
    """------------------ origin + 1 class ------------------"""
    # num_new_class_influence_helper(old_dataset, new_dataset, mix_model_name, 1)
    """------------------ origin + 3 class ------------------"""
    # num_new_class_influence_helper(old_dataset, new_dataset, mix_model_name, 3)
    """------------------ origin + 5 class ------------------"""
    # num_new_class_influence_helper(old_dataset, new_dataset, mix_model_name, 5)
    """------------------ origin + 10 class ------------------"""
    # num_new_class_influence_helper(old_dataset, new_dataset, mix_model_name, 10)
    """------------------ origin + 20 class ------------------"""
    # num_new_class_influence_helper(old_dataset, new_dataset, mix_model_name, 20)
    """------------------ origin + 50 class ------------------"""
    # num_new_class_influence_helper(old_dataset, new_dataset, mix_model_name, 50)
    """------------------ origin + 99 class ------------------"""
    # num_new_class_influence_helper(old_dataset, new_dataset, mix_model_name, 99)
    """------------------ plot ------------------"""
    plot_variation_diagram_for_num_new_class_influence(old_dataset, new_dataset, mix_model_name)


def plot_variation_diagram_for_different_capacity_model(new_dataset, mix_model_name_prev, mix_model_name_next):
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    x = range(10, 99, 10)
    # ta = []
    # tp = []
    # tr = []
    # tf = []
    # tk = []
    # for i in x:
    #     a, p, r, f, k = valid_model(new_dataset, mix_model_name_prev + str(i) + mix_model_name_next, False)
    #     ta.append(a)
    #     tp.append(p)
    #     tr.append(r)
    #     tf.append(f)
    #     tk.append(k)

    ta = [0.7465309898242368, 0.8418131359851989, 0.8427382053654024, 0.8538390379278445, 0.8566142460684552, 0.8649398704902868, 0.877890841813136, 0.883441258094357, 0.883441258094357]
    tp = [0.7354946358773319, 0.8257101344222554, 0.8301524225609905, 0.8368445615367094, 0.8470402827099077, 0.8592384290809216, 0.8813618110989101, 0.8832585387297591, 0.8843221475054508]
    tr = [0.7754092537143045, 0.8614601023803707, 0.8647641472452886, 0.8815831758358424, 0.8712336785234758, 0.8848868673544164, 0.9036962401656504, 0.9068560286225121, 0.9111096559440238]
    tf = [0.7549247184354764, 0.843206359879758, 0.8471048835891862, 0.8586314922598843, 0.858966658417148, 0.8718740601651896, 0.8923893028032709, 0.8949017513042432, 0.8975160694039084]
    tk = [0.7429197341309174, 0.8394237877365088, 0.8404094106081672, 0.8516784928691081, 0.8544436577812333, 0.8628992466781564, 0.8760626895643625, 0.8817007247031357, 0.8816931209578092]

    print(ta)
    print(tp)
    print(tr)
    print(tf)
    print(tk)

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 4))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(x, ta, marker='+', color="orange", label="Accuracy", linewidth=1.5)
    plt.plot(x, tp, marker='<', color="green", label="Precision", linewidth=1.5)
    plt.plot(x, tr, marker='o', color="red", label="Recall", linewidth=1.5)
    plt.plot(x, tf, marker='*', color="blue", label="F1-score", linewidth=1.5)
    plt.plot(x, tk, marker='v', color="purple", label="Kappa", linewidth=1.5)

    group_labels = x
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("The model is trained with different numbers of classes", fontsize=13, fontweight='bold')
    # plt.ylabel("", fontsize=13, fontweight='bold')
    # plt.xlim(0.9, 6.1)  # 设置x轴的范围
    # plt.ylim(1.5, 16)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()


def different_capacity_model():
    old_dataset = wd.make_mix_data(wd.load_dataset_with_start_end(ecg_id, 0, 45),
                                   wd.load_dataset_with_start_end(mit_db_re, 0, 24))
    old_dataset = wd.make_mix_data(old_dataset, wd.load_dataset_with_start_end(usst_db, 0, 30))

    new_dataset = wd.make_mix_data(wd.load_dataset_with_start_end(ecg_id, 45, 90),
                                   wd.load_dataset_with_start_end(mit_db_re, 24, 48))
    new_dataset = wd.make_mix_data(new_dataset, wd.load_dataset_with_start_end(usst_db, 30, 60))
    mix_model_name_prev = "wx_save_10/mix_extractor_"
    mix_model_name_next = ".pt"
    # for i in range(10, 99, 10):
        # train_model(wd.load_dataset_with_start_end(old_dataset, 0, i),
        #             mix_model_name_prev + str(i) + mix_model_name_next, i, 500)
    plot_variation_diagram_for_different_capacity_model(new_dataset, mix_model_name_prev, mix_model_name_next)


def non_fixed_collate_fn(batch):
    """
    Function: The default collate_fn need all x with same dimension, our x has different length, \
    so we put them in a list
    """
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    y = torch.Tensor(y)
    return x, y


if __name__ == '__main__':
    # rest_train_exercise_test()

    # both_train_test()

    single_database()

    # add_new_class_inner()

    # add_new_class_inter()

    # num_new_class_influence()

    # different_capacity_model()

    pass
