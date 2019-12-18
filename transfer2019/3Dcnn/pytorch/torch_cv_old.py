import numpy as np
import argparse, os
import nibabel as nib
import torch
import torch.nn as nn
# from math import sqrt
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
# import math
# from read_data import ReadData
import time


def total_gradient(parameters):
    """Computes a gradient clipping coefficient based on gradient norm."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0
    for p in parameters:
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = totalnorm ** (1. / 2)
    return totalnorm


parser = argparse.ArgumentParser(description='PyTorch for AD')

parser.add_argument('--NC', type=int, default=0, help='Constant label for NC')
parser.add_argument('--MCI', type=int, default=1, help='Constant label for MCI')
parser.add_argument('--AD', type=int, default=2, help='Constant label for AD')
parser.add_argument('--lossFunction', type=int, default=0, help='Loss Function: 0 CrossEntropy, 1 Mean Square Error')
parser.add_argument('--optimizer', type=int, default=0, help='0 SGD, 1 ADAM')
parser.add_argument('--crossValidation', type=int, default=10, help='Cross Validation: <= 1, no CV performed')
parser.add_argument('--augmentation', type=bool, default=False, help='Augment data using flip')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--step', type=int, default=15,
                    help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=15')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--clip', type=float, default=10, help='Clipping Gradients. Default=10')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--momentum', default=0, type=float, help='momentum, default:0.9')
parser.add_argument('--weight-decay', '--wd', default=0.001, type=float, help='weight decay, Default: 0, 1e-3')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument('--lrDecay', default=0.01, type=float,
                    help='Decay rate of learning rate lr=opt.lr/(1+ lrDecay*epoch)')
opt = parser.parse_args()


class CNN(nn.Module):
    def __init__(self, width, height, depth, nLabel):
        super(CNN, self).__init__()

        self.width = width
        self.height = height
        self.depth = depth

        # if nLabel <= 2:
        #     self.nLable = 1
        # else:
        #     self.nLable = nLabel
        self.nLable = nLabel
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=True)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=True)
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=True)
        h = int((((((((self.height - 3) + 1) // 2 - 3) + 1) // 2 - 3 + 1) // 2 - 3) + 1) // 2)
        w = int((((((((self.width - 3) + 1) // 2 - 3) + 1) // 2 - 3 + 1) // 2 - 3) + 1) // 2)
        d = int((((((((self.depth - 3) + 1) // 2 - 3) + 1) // 2 - 3 + 1) // 2 - 3) + 1) // 2)
        f.print_and_log('left:{} {} {}'.format(w, h, d))
        self.fc1 = nn.Linear(32 * h * w * d, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, self.nLable)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), (2, 2, 2), stride=(2, 2, 2))
        # x = F.relu(self.conv1(x))
        x = F.max_pool3d(F.relu(self.conv2(x)), (2, 2, 2), stride=(2, 2, 2))
        x = F.max_pool3d(F.relu(self.conv3(x)), (2, 2, 2), stride=(2, 2, 2))  # 32 12 8 8
        x = F.max_pool3d(F.relu(self.conv4(x)), (2, 2, 2), stride=(2, 2, 2))  # 32 6 4 4

        '''
        x = F.max_pool3d(F.relu(self.conv1(x)), (2, 2, 2), stride=(2, 2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(F.relu(self.conv3(x)), (2, 2, 2), stride=(2, 2, 2))  # 32 12 8 8
        x = F.relu(self.conv4(x))
        '''

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = self.fc3(x)

        return x


class SSPLayer():
    def __init__(self):
        pass


class LogSystem:
    def __init__(self, log_folder, test_flag=False):
        self.test = test_flag
        self.sep = '_'
        log_filename = data_folder + '_' + self.sep.join(comparison) + '_' + time.strftime("%Y_%m_%d_%H_%M",
                                                                                           time.localtime()) + '.txt'
        if not self.test:
            self.logfile = open(log_folder + '/' + log_filename, 'w')
        self.print_and_log(log_folder + '/' + log_filename)

    def log(self, text):
        self.logfile.write(text + '\n')

    def print_and_log(self, text):
        # text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n' + text
        print(text)
        if not self.test:
            self.log(text)

    def flush(self):
        if not self.test:
            self.logfile.flush()


class LoadData():
    def __init__(self, data_folder, nLabel, train_test_rate=0.9,
                 label_path='/home/chao/data_3dcnn/data/label_all.txt', augmentation=False,
                 cross_validation=1):
        self.label_path = label_path
        self.nLabel = nLabel
        self.label = {}
        self.sample_data = []
        self.sample_label = []
        self.rate = train_test_rate
        self.augmentation = augmentation
        self.cross_validation = cross_validation
        self.current_test_fold = 0
        # self.data_folder = data_folder
        self.data_folder = data_folder
        self.d_min = self.h_min = self.w_min = 218
        self.d_max = self.h_max = self.w_max = 0
        self.dic = {'NC': 0, 'MCI': 1, 'AD': 2}
        self.training_set = []
        self.load_data()

    def load_label(self):
        for info in np.loadtxt(self.label_path, dtype=np.str)[:]:
            self.label[info[0]] = info[1]

    def get_data(self):

        if self.cross_validation > 1:
            training_threshold = 1.0 / self.cross_validation
            testing_start = round(self.current_test_fold * training_threshold * self.get_num_data())
            testing_end = round((self.current_test_fold + 1) * training_threshold * self.get_num_data())
            testing_data = self.sample_data[testing_start:testing_end]
            training_data = self.sample_data[0:testing_start] + self.sample_data[testing_end:]

            self.current_test_fold += 1
            if self.current_test_fold == self.cross_validation:
                self.current_test_fold = 0
        else:
            training_threshold = round(len(self.sample_data) * self.rate)
            training_data = self.sample_data[:training_threshold]
            testing_data = self.sample_data[training_threshold:]
        if self.augmentation:
            training_data = self.augment_data(training_data)
        # self.training_set
        # return sample_data[:training_threshold], sample_label[:training_threshold], sample_data[training_threshold:], sample_label[training_threshold:]
        return training_data, testing_data

    def augment_data(self, data):
        tmp_data = data.copy()
        for d in tmp_data:
            d[0] = np.flip(d[0], 3).copy()
            data.append(d)
        np.random.shuffle(data)
        return data
        # training_set = np.concatenate((training_set, np.flip(np.flip(training_set, 2), 4)))
        # training_label = np.concatenate((training_label, training_label))

    def get_num_data(self):
        return len(self.sample_data)

    def get_shape(self):
        return self.d_max - self.d_min, self.h_max - self.h_min, self.w_max - self.w_min

    def get_engagement(self):
        return self.d_min, self.d_max, self.h_min, self.h_max, self.w_min, self.w_max

    def get_engagement_tostring(self):
        sep = ', '
        return sep.join(self.get_engagement())

    def load_data(self):
        self.load_label()
        target_dic = {}
        indx = 0
        for k, l in enumerate(comparison):
            target_dic[self.dic[l]] = k
        num_total = len(self.label)
        if num > 0:
            num_total = num
        mci_count = 300
        data_list = os.listdir(self.data_folder)[:num_total]
        np.random.shuffle(data_list)
        for file in data_list:
            # loaded_file = nib.load(self.directory + "/" + file)
            tmp_data = nib.load(self.data_folder + "/" + file).get_data()  # 182*218*182
            sample_label_tmp = np.zeros((1, self.nLabel), dtype=np.float32)

            separate = '_'
            image_id = separate.join(file.split(separate)[0:5])
            target_label = int(self.label[image_id])
            if target_label not in target_dic:
                continue
            # select 300 MCI instances only
            if target_label == 1:
                mci_count -= 1
                if mci_count < 0:
                    continue

            sample_label_tmp[0][target_dic[target_label]] = 1.0

            self.sample_data.append([tmp_data[:].reshape(1, 182, 218, 182), sample_label_tmp[0].copy(), image_id])

            d_edge, h_edge, w_edge = tmp_data.nonzero()

            if d_edge.min() < self.d_min: self.d_min = d_edge.min()
            if d_edge.max() > self.d_max: self.d_max = d_edge.max()

            if h_edge.min() < self.h_min: self.h_min = h_edge.min()
            if h_edge.max() > self.h_max: self.h_max = h_edge.max()

            if w_edge.min() < self.w_min: self.w_min = w_edge.min()
            if w_edge.max() > self.w_max: self.w_max = w_edge.max()
            if indx % 100 == 0:
                f.print_and_log(
                    "currently processing: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " - " + str(indx))
            indx += 1
        np.random.shuffle(self.sample_data)
        self.d_max += 2
        self.h_max += 2
        self.w_max += 2
        self.d_min -= 2
        self.h_min -= 2
        self.w_min -= 2
        for data in self.sample_data:
            data[0] = data[0][:, self.d_min:self.d_max, self.h_min:self.h_max, self.w_min:self.w_max]


def isset(var_str):
    return var_str in globals().keys()


test = False
num = 0
if test:
    num = 150
width = 182
height = 218
depth = 182

comparison = ['NC', 'AD']
nLabel = len(comparison)

base_path = "/media/chao/claudiolocal/3dcnndata"
data_folder = 'HIPPO_AMYG_THAL_PUTA_PALL_CAUD_MTL_NEW'

if opt.crossValidation <= 1:
    cross_validation = 1
else:
    cross_validation = opt.crossValidation

log_folder = base_path + '/Log'

f = LogSystem(log_folder, test_flag=test)
f.print_and_log(comparison.__str__())
f.print_and_log(opt.__str__())

f.print_and_log("reading data start ")
directory = base_path + '/' + data_folder

learning_rates = [0.1, 0.01, 0.001]
# for lrs in learning_rates:
if opt.lr > 0:
    lr = opt.lr
    data = LoadData(directory, nLabel, augmentation=opt.augmentation, cross_validation=cross_validation)
    # training_set, training_label, testing_set, testing_label = data.get_data()
    f.print_and_log("reading data finish, total:" + str(data.get_num_data()))
    cv = 0
    cv_acc = []
    while cv < cross_validation:
        cv += 1
        f.print_and_log("Cross Validation {}".format(cv))
        training_sample, testing_sample = data.get_data()

        # data augmentation start
        # training_set = np.concatenate((training_set, np.flip(np.flip(training_set, 2), 4)))
        # training_label = np.concatenate((training_label, training_label))
        # data augmentation end

        train_num = len(training_sample)
        test_num = len(testing_sample)

        f.print_and_log("Training, Testing number: {}, {}".format(train_num, test_num))

        sample_depth, sample_height, sample_width = data.get_shape()

        engage = data.get_engagement()

        f.print_and_log("The engagement of data: " + engage.__str__())
        # f.print_and_log(str(sample_depth) + str(sample_height), + str(sample_width))

        # exit()

        model = CNN(sample_depth, sample_height, sample_width, nLabel)

        f.print_and_log(model.__str__())

        print(model.parameters())
        # exit()
        training_data_loader = DataLoader(dataset=training_sample, num_workers=opt.threads, batch_size=opt.batchSize,
                                          shuffle=False)
        # label_data_loader = DataLoader(dataset=training_label, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

        testing_data_loader = DataLoader(dataset=testing_sample, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                         shuffle=False)
        # testing_label_loader = DataLoader(dataset=testing_label, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

        cuda = opt.cuda
        if cuda and not torch.cuda.is_available():
            raise Exception('No GPU found, please run without --cuda')
        opt.seed = random.randint(1, 10000)
        f.print_and_log("Random Seed: " + str(opt.seed))
        torch.manual_seed(opt.seed)
        if cuda:
            torch.cuda.manual_seed(opt.seed)

        if cuda:
            if torch.cuda.device_count() >= 2:
                model = nn.DataParallel(model)
            model = model.cuda()
            f.print_and_log('===> Setting {} GPU'.format(torch.cuda.device_count()))

        else:
            f.print_and_log('===> Setting CPU')

        f.print_and_log('===> Setting Optimizer')
        if opt.optimizer == 1:
            optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                  weight_decay=opt.weight_decay)  # , weight_decay=opt.weight_decay)  # , momentum=opt.momentum, weight_decay=opt.weight_decay)
        f.print_and_log('===> Training')

        if opt.lossFunction == 0:
            creterion = nn.CrossEntropyLoss()
            f.print_and_log('==>CEL')
        elif opt.lossFunction == 1:
            creterion = nn.MSELoss()

        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            f.print_and_log('Corss Validation:' + str(cv) + ' epoch = ' + str(epoch) + ' lr =' + str(
                optimizer.param_groups[0]['lr']) + time.strftime(" %Y-%m-%d %H:%M:%S", time.localtime()))
            # parameters = model.parameters()
            acc_test = 0
            for iteration, batch in enumerate(training_data_loader):  # , label_data_loader)):
                model.train()
                input, target = Variable(batch[0]), Variable(batch[1].type(torch.FloatTensor), requires_grad=False)
                if opt.lossFunction == 0:
                    target = target.max(1)[1]  # CrossEntropyLoss parameter

                # exit()
                if opt.cuda:
                    input = input.cuda()
                    target = target.cuda()
                pre = model(input)
                acc_target = target.data
                if opt.lossFunction == 1:
                    acc_target = target.max(1)[1].data
                acc_test += np.count_nonzero(np.equal(np.argmax(pre.data, axis=1), acc_target))

                # acc_test += np.count_nonzero(np.equal(pre.data.gt(0).t(), acc_target))
                loss = creterion(pre, target)
                # model.zero_grad()
                optimizer.zero_grad()

                loss.backward()

                # nn.utils.clip_grad_norm(model.parameters(), opt.clip)
                optimizer.step()

                if iteration % 300 == 0 or (len(training_data_loader) == (
                        iteration + 1) and epoch == opt.nEpochs):  # or len(training_data_loader) == (iteration+1):
                    model.eval()
                    f.print_and_log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    f.print_and_log(
                        "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                      loss.data[0]))
                    f.print_and_log('total gradient: ' + str(total_gradient(model.parameters())))

                    test_loss = 0
                    correct = 0
                    loss_fn = nn.MSELoss()
                    results = None
                    for batchs in testing_data_loader:  # , testing_label_loader):

                        test_input, test_target, test_img_id = Variable(batchs[0]), Variable(
                            batchs[1].type(torch.FloatTensor), requires_grad=False), batchs[2]
                        if opt.cuda:
                            test_input = test_input.cuda()
                            test_target = test_target.cuda()

                        prediction = model(test_input)

                        tmp_test_loss = loss_fn(prediction, test_target)
                        print(tmp_test_loss.data)
                        test_loss += tmp_test_loss.data[0]
                        # correct += np.count_nonzero(np.equal(prediction.data.gt(0).t(), test_target.data))
                        pred = prediction.data.max(1)[1]
                        targ = test_target.data.max(1)[1]
                        correct += pred.eq(targ).cpu().sum()
                        if len(training_data_loader) == (iteration + 1) and epoch == opt.nEpochs:
                            num_input = len(test_input)
                            r_tmp = None
                            r_tmp = np.concatenate((prediction.data.cpu().numpy(),
                                                    pred.cpu().numpy().reshape(num_input, 1),
                                                    targ.cpu().numpy().reshape(num_input, 1)), axis=1)
                            if results is None:
                                results = r_tmp
                            else:
                                results = np.concatenate((results, r_tmp), axis=0)
                        #         f.print_and_log('Test {:25s}: {:.15f} {:.15f} {} {}'.format(id, prediction.data[key][0], prediction.data[key][1], pred[key], targ[key]))

                        # id +" "+ str(prediction.data[key][0]) +": "+ str(prediction.data[key][1]) +" "+ str(pred[key]) +" "+ str(targ[key]))

                        del tmp_test_loss, prediction
                    # test_loss = test_loss
                    test_loss /= len(testing_data_loader)  # loss function already averages over batch size
                    f.print_and_log(
                        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(test_loss, correct,
                                                                                             test_num,
                                                                                             100.0 * correct / test_num))
                    test_loss = 0
                    f.flush()
                    if len(training_data_loader) == (iteration + 1) and epoch == opt.nEpochs:
                        if nLabel == 2:
                            false_positive = 0.0
                            false_negtive = 0.0
                            true_positive = 0.0
                            true_negtive = 0.0
                            positive = 0.0
                            negtive = 0.0
                            other = 0.0
                            for r in results:
                                if r[-1] == 0:
                                    negtive += 1
                                    if r[-1] == r[-2]:
                                        true_negtive += 1
                                    else:
                                        false_positive += 1
                                elif r[-1] == 1:
                                    positive += 1
                                    if r[-1] == r[-2]:
                                        true_positive += 1
                                    else:
                                        false_negtive += 1
                                else:
                                    other += 1
                            f.print_and_log("Other is {}".format(other))
                            f.print_and_log(results.__str__())
                            f.print_and_log("True Positive (SENS): {}/{} {:.5f}%".format(true_positive, positive,
                                                                                         true_positive / positive))
                            f.print_and_log("False Positive: {}/{} {:.5f}%".format(false_positive, positive,
                                                                                   false_positive / positive))
                            f.print_and_log("True Negtive (SPEC): {}/{} {:.5f}%".format(true_negtive, negtive,
                                                                                        true_negtive / negtive))
                            f.print_and_log(
                                "False Negtive: {}/{} {:.5f}%".format(false_negtive, negtive, false_negtive / negtive))

                        cv_acc.append(100.0 * correct / test_num)
            f.print_and_log('Training Accuracy:{}/{} {:.5f}%\n'.format(acc_test, train_num, acc_test / train_num * 100))
        f.print_and_log(cv_acc.__str__())

    f.print_and_log("Cross Validation Accuracy:{} {:.5f}%".format(cv_acc.__str__(), np.mean(cv_acc)))