import numpy as np
import argparse, os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import time


def total_gradient(parameters):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0
    for p in parameters:
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = totalnorm ** (1. / 2)
    return totalnorm


parser = argparse.ArgumentParser(description='PyTorch for AD, usage example: python3 torch_cv.py /media/chao/claudiolocal/3dcnndata /media/chao/claudiolocal/3dcnndata/HIPPO_AMYG_THAL_PUTA_PALL_CAUD_MTL_NEW /home/chao/data_3dcnn/data/label_all.txt')

parser.add_argument('--lossFunction', '-loss', type=int, default=1, choices=[0, 1], help='Loss Function: 0 CrossEntropy, 1 Mean Square Error, default 1')
parser.add_argument('--optimizer', '-optm', type=int, default=0, choices=[0, 1], help='0 gradient descent, 1 Adam, default 0')
parser.add_argument('--crossValidation', '-cv', type=int, default=10, help='Cross Validation: <= 1, no CV performed, default 10')
parser.add_argument('--augmentation',  action='store_true', help='Augment data using mirror flip, default False')
parser.add_argument('--trainBatchSize', '-trSize', type=int, default=4, help='training batch size, default 4')
parser.add_argument('--testBatchSize', '-teSize',type=int, default=4, help='testing batch size, default 4')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', '-lr', type=float, default=0.01, help='Learning Rate, default=0.01')
parser.add_argument('--cfg', type=str, default='A', help='Specify the network structure from the predifined list')
parser.add_argument('--cuda', action='store_true', help='use cuda or not? default: False')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate of the standard neural network, default 0.1')
parser.add_argument('--momentum', default=0, type=float, help='momentum, default:0')
parser.add_argument('--weight-decay', '-wd', default=0.001, type=float, help='weight decay, default: 0, 1e-3')
parser.add_argument('--nWay', '-nWay', default=2, type=int, choices=[2, 3], help='Type of classification. 2 two-way classification, 3 three-way classfication')
parser.add_argument('outputFolder', default='', type=str, help='The output folder. eg. /media/chao/claudiolocal/3dcnndata')
parser.add_argument('MRIdataFolder', type=str, help='MRI images folder. eg. /media/chao/claudiolocal/3dcnndata/HIPPO_AMYG_THAL_PUTA_PALL_CAUD_MTL_NEW')
parser.add_argument('labelFile',  type=str, help='Label file. eg. /home/chao/data_3dcnn/data/label_all.txt')




class CNN(nn.Module):
    def __init__(self, CNNLayer, last_output, width, height, depth, nLabel):
        super(CNN, self).__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.nLable = nLabel
        self.cnn = CNNLayer
        print(CNNLayer)
        #
        self.linear = nn.Sequential(
            nn.Linear(last_output * int(self.width) * int(self.height) * int(self.depth), 1000),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(100, self.nLable),
            nn.Softmax()
        )

        f.print_and_log('Size of MRI after CNN:{} {} {}'.format(self.width, self.height, self.depth))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class SSPLayer():
    def __init__(self):
        pass


class LogSystem:
    def __init__(self, log_folder, test_flag=False):
        self.test = test_flag
        self.sep = '_'
        log_filename = opt.MRIdataFolder.split('/')[-1] + '_' + self.sep.join(comparison) + '_' + time.strftime("%Y_%m_%d_%H_%M",
                                                                                           time.localtime()) + '.txt'
        #if not self.test:
        self.logfile = open(log_folder + '/' + log_filename, 'w')
        self.print_and_log(log_folder + '/' + log_filename)

    def log(self, text):
        self.logfile.write(text + '\n')

    def print_and_log(self, text):
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
        self.data_folder = data_folder
        self.d_min = self.h_min = self.w_min = 500
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
        return training_data, testing_data

    def augment_data(self, data):
        tmp_data = data.copy()
        for d in tmp_data:
            d[0] = np.flip(d[0], 3).copy()
            data.append(d)
        np.random.shuffle(data)
        return data


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

        data_list = os.listdir(self.data_folder)[:num_total]
        np.random.shuffle(data_list)
        for file in data_list:
            tmp_data = nib.load(self.data_folder + "/" + file).get_data()
            sample_label_tmp = np.zeros((1, self.nLabel), dtype=np.float32)
            separate = '_'
            image_id = separate.join(file.split(separate)[0:5])
            target_label = int(self.label[image_id])
            if target_label not in target_dic:
                continue

            sample_label_tmp[0][target_dic[target_label]] = 1.0

            #The size of ADNI1 MRI Data is 182*218*182
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


def layers(config, width, height, depth):
    layer = []
    in_channel = 1

    for v in config:
        if v == 'M':
            layer.append(nn.MaxPool3d(kernel_size=2, stride=2))
            width = width // 2
            height = height // 2
            depth = depth // 2
        elif v == 'R':
            layer.append(nn.ReLU(inplace=True))
        else:
            width = (width + 2 * v['pad'] - v['size'])//v['stride'] + 1
            height = (height + 2 * v['pad'] - v['size']) // v['stride'] + 1
            depth = (depth + 2 * v['pad'] - v['size']) // v['stride'] + 1
            conv3d = nn.Conv3d(in_channel, v['out_channel'], kernel_size=v['size'], stride=v['stride'], padding=v['pad'])
            layer.append(conv3d)
            in_channel = v['out_channel']

    return nn.Sequential(*layer), in_channel, width, height, depth


def path_exist(p):
    if not os.path.exists(p):
        return False
    return True


def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)


#if test = True, 100 instances will be read only
test = True
num = 0
if test:
    num = 100


cfg = {
    'A': [{'out_channel': 32, 'size': 3, 'stride': 1, 'pad': 1}, 'M', 'R',
          {'out_channel': 32, 'size': 3, 'stride': 1, 'pad': 1}, 'M', 'R',
          {'out_channel': 32, 'size': 3, 'stride': 1, 'pad': 1}, 'M', 'R',],

    'B': [{'out_channel': 16, 'size': 3, 'stride': 1, 'pad': 0}, 'R', 'M',
          {'out_channel': 16, 'size': 3, 'stride': 1, 'pad': 0}, 'R', 'M',
          {'out_channel': 16, 'size': 3, 'stride': 1, 'pad': 0}, 'R', 'M',]
}



if __name__ == "__main__":
    opt = parser.parse_args()
    # print(opt.__str__())
    # opt.outputFolder = "/media/chao/claudiolocal/3dcnndata"
    # opt.MRIdataFolder = '/media/chao/claudiolocal/3dcnndata/HIPPO_AMYG_THAL_PUTA_PALL_CAUD_MTL_NEW'
    # opt.labelFile = '/home/chao/data_3dcnn/data/label_all.txt'
    #/bwfefs/home/fumie/data/chao/download/up_riken/label.txt

    if not path_exist(opt.outputFolder) or not path_exist(opt.MRIdataFolder) or not path_exist(opt.labelFile):
        print('Folder or file does not exist!')
        print("python3 preprocess.py -h")
        exit()
    if opt.cfg not in cfg:
        print('Network structure does not exist!')
        exit()
    if opt.outputFolder[-1] == '/' or opt.MRIdataFolder[-1] == '/':
        print('Wrong format, please remove the last /')
        exit()
    log_folder = opt.outputFolder + '/Log'
    mkdir(log_folder)

    if opt.nWay == 2:
        comparison = ['NC', 'AD']
    elif opt.nWay == 3:
        comparison = ['NC', 'MCI', 'AD']
    nLabel = len(comparison)
    if opt.crossValidation <= 1:
        opt.crossValidation = 1


    f = LogSystem(log_folder, test_flag=test)
    f.print_and_log(comparison.__str__())
    f.print_and_log(opt.__str__())

    f.print_and_log("Reading data start ")

    # Load data start
    data = LoadData(opt.MRIdataFolder, nLabel, augmentation=opt.augmentation, cross_validation=opt.crossValidation, label_path=opt.labelFile)
    f.print_and_log("reading data finish, total:" + str(data.get_num_data()))
    cv = 0
    cv_acc = []
    while cv < opt.crossValidation:
        cv += 1
        f.print_and_log("Cross Validation {}".format(cv))
        #get training and testing dataset
        training_sample, testing_sample = data.get_data()

        train_num = len(training_sample)
        test_num = len(testing_sample)

        f.print_and_log("Training, Testing number: {}, {}".format(train_num, test_num))

        sample_depth, sample_height, sample_width = data.get_shape()

        engage = data.get_engagement()

        f.print_and_log("The engagement of data: " + engage.__str__())

        model = CNN(*layers(cfg[opt.cfg], sample_depth, sample_height, sample_width), nLabel)

        f.print_and_log(model.__str__())

        print(model.parameters())

        training_data_loader = DataLoader(dataset=training_sample, num_workers=opt.threads, batch_size=opt.trainBatchSize,
                                          shuffle=False)
        testing_data_loader = DataLoader(dataset=testing_sample, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                         shuffle=False)

        if opt.cuda and not torch.cuda.is_available():
            raise Exception('No GPU found, please run without --cuda')
        opt.seed = random.randint(1, 10000)
        f.print_and_log("Random Seed: " + str(opt.seed))
        torch.manual_seed(opt.seed)
        if opt.cuda:
            torch.cuda.manual_seed(opt.seed)

        if opt.cuda:
            if torch.cuda.device_count() >= 2:
                model = nn.DataParallel(model)
            model = model.cuda()
            f.print_and_log('===> Setting {} GPU'.format(torch.cuda.device_count()))

        else:
            f.print_and_log('===> Setting CPU')

        f.print_and_log('===> Setting Optimizer')
        if opt.optimizer == 1:
            optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            f.print_and_log('===> Adam')
        else:
            optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                  weight_decay=opt.weight_decay)
            f.print_and_log('===> SGD')

        f.print_and_log('===> Setting loss function')
        if opt.lossFunction == 0:
            creterion = nn.CrossEntropyLoss()
            f.print_and_log('===> Cross entropy')
        elif opt.lossFunction == 1:
            creterion = nn.MSELoss()
            f.print_and_log('===> MSE')
        f.print_and_log('===> Start training')
        for epoch in range(1, opt.nEpochs + 1):
            f.print_and_log('Corss Validation:' + str(cv) + ' epoch = ' + str(epoch) + ' lr =' + str(
                optimizer.param_groups[0]['lr']) + time.strftime(" %Y-%m-%d %H:%M:%S", time.localtime()))
            acc_test = 0
            for iteration, batch in enumerate(training_data_loader):
                model.train()
                input, target = Variable(batch[0]), Variable(batch[1].type(torch.FloatTensor), requires_grad=False)
                if opt.lossFunction == 0:
                    target = target.max(1)[1]
                if opt.cuda:
                    input = input.cuda()
                    target = target.cuda()
                pre = model(input)
                acc_target = target.data
                if opt.lossFunction == 1:
                    acc_target = target.max(1)[1].data
                acc_test += np.count_nonzero(np.equal(np.argmax(pre.data, axis=1), acc_target))

                loss = creterion(pre, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iteration % 300 == 0 or (len(training_data_loader) == (
                        iteration + 1) and epoch == opt.nEpochs):
                    model.eval()
                    # f.print_and_log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    f.print_and_log(
                        "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                      loss.data[0]))
                    f.print_and_log('total gradient: ' + str(total_gradient(model.parameters())))

                    test_loss = 0
                    correct = 0.0
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
                        test_loss += tmp_test_loss.data[0]
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

                        del tmp_test_loss, prediction
                    test_loss /= len(testing_data_loader)
                    f.print_and_log(
                        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(test_loss, correct,
                                                                                             test_num,
                                                                                             correct / test_num * 100.0))
                    test_loss = 0
                    f.flush()
                    #calculate ROC metrics
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

                            f.print_and_log(results.__str__())
                            f.print_and_log("Other is {}".format(other))
                            f.print_and_log("True Positive (SENS): {}/{} {:.5f}%".format(true_positive, positive,
                                                                                         true_positive / positive))
                            f.print_and_log("False Positive: {}/{} {:.5f}%".format(false_positive, negtive,
                                                                                   false_positive / negtive))
                            f.print_and_log("True Negtive (SPEC): {}/{} {:.5f}%".format(true_negtive, negtive,
                                                                                        true_negtive / negtive))

                        cv_acc.append(correct / test_num * 100.0)
            f.print_and_log('Training Accuracy:{}/{} {:.5f}%\n'.format(acc_test, train_num, acc_test / train_num * 100))
        f.print_and_log(cv_acc.__str__())
    f.print_and_log("Cross Validation Accuracy:{} {:.5f}%".format(cv_acc.__str__(), np.mean(cv_acc)))