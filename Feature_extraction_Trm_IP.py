import scipy.io as sio
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import datetime
import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
from models.vit_pytorch import ViT
from sklearn.metrics import confusion_matrix
import numpy as np
import time

from generate_pic import generate_png


def get_accuracy(y_true, y_pred):
    num_perclass = np.zeros(int(y_true.max() + 1))
    num = np.zeros(int(y_true.max() + 1))
    for i in range(len(y_true)):
        num_perclass[int(y_true[i])] += 1
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            num[int(y_pred[i])] += 1
    for i in range(len(num)):
        num[i] = num[i] / num_perclass[i]
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ac = np.zeros(int(y_true.max() + 1 + 2))
    ac[:int(y_true.max() + 1)] = num
    ac[-1] = acc
    ac[-2] = kappa
    return ac  # acc,num.mean(),kappa


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


# -------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        print(loss)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


# -------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


def _test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band


def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#--------------------------------------------
def gain_motion_band(x_train, band,band_num):

    x_train_band = np.zeros((x_train.shape[0], band_num, int(band / 3)), dtype=float)

    k = band_num / 3 - 1
    for i in range(int(band / 3 - k)):
        x_train_band[:, :, i] = x_train[:, 3 * i:3 * i + band_num]

    return x_train_band
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='./outputs/models/HSI_Raw_ckpt_50epoch_10HSI.pth.tar')
    parser.add_argument('-s', '--test_shape', default=[192, 192], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    #parser = argparse.ArgumentParser(description="HSI")

    parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset to use')
    parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
    parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='number of seed')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
    parser.add_argument('--patches', type=int, default=1, help='number of patches')
    parser.add_argument('--band_patches', type=int, default=3, help='number of related band')
    parser.add_argument('--epoches', type=int, default=500, help='epoch number')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')
    #args = parser.parse_args(args=[])

    args = parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)




    experiment_num = 1
    dataname = 'IP'
    model_name = 'SF-SMF'
    num_range=[10,20,50,80,100,150,200]
    #num_range = [50]

    path = '/media/liubing/cc5992d7-d217-4ded-97b4-fd47f4fa55f4/syf/HSI_data/'
    # path='D:/HSI_data/'
    img=sio.loadmat(path+'Indian_pines_corrected.mat')
    img=img['indian_pines_corrected']

    gt=sio.loadmat(path+'Indian_pines_gt.mat')
    gt=gt['indian_pines_gt']


    spec = img.copy()
    spec = spec / spec.max()
    m, n, b = img.shape

    hsv = np.zeros((m, n, 3))
    hsv[..., 1] = 255

    feature = []
    # for i in tqdm(range(3, b)):           #连续三帧影像
    #    x1 = img[:, :, i - 3:i]
    #    x2 = img[:, :, i - 2:i + 1]
    for i in range(1, b):  # 单灰度值影像
        x1 = img[:, :, i - 1]
        x2 = img[:, :, i]
        x1 = np.array(x1, dtype='uint8')
        x2 = np.array(x2, dtype='uint8')

        f = cv2.optflow.calcOpticalFlowSparseToDense(x1, x2,None)    #SparseToDense  3 or 1

        if i == 1:
            feature = f
        else:
            feature = np.concatenate((feature, f), 2)


    v_min = feature.min()
    v_max = feature.max()
    feature = (feature - v_min) / (v_max - v_min)

    '''
    for i in range(spec.shape[2]):
        if i >0 :
            m1 = feature[:, :, i * 2 - 2]
            m2 = feature[:, :, i * 2 - 1]
            spec[:,:,i]=spec[:,:,i]*m1*m2
    '''

    feature = np.concatenate((spec, feature), 2)


    num_patches=feature.shape[2]



    label_num = gt.max()
    data = []
    label = []
    data_global = []

    gt_index = []
    for i in tqdm(range(m)):
        for j in range(n):
            if gt[i, j] == 0:
                continue
            else:
                temp_data = feature[i, j, :]
                temp_label = np.zeros((1, label_num), dtype=np.int8)
                temp_label[0, gt[i, j] - 1] = 1
                data.append(temp_data)
                label.append(temp_label)
                gt_index.append((i) * n + j)
    #   print (i,j)

    for i in tqdm(range(m)):
        for j in range(n):
            temp_data = feature[i, j, :]
            data_global.append(temp_data)

    print('end')
    data = np.array(data)
    data = np.squeeze(data)

    data_global = np.array(data_global)
    data_global = np.squeeze(data_global)

    label = np.array(label)
    label = np.squeeze(label)
    label = label.argmax(1)

    data = np.float32(data)
    data_global = np.float32(data_global)

    Experiment_result = np.zeros([label_num + 5, experiment_num + 2])
    for i_num in range(num_range.__len__()):
        num = num_range[i_num]
        for iter_num in range(experiment_num):

            # np.random.seed(123456789)
            np.random.seed(iter_num+123456)
            indices = np.arange(data.shape[0])
            shuffled_indices = np.random.permutation(indices)

            images = data[shuffled_indices]
            labels = label[shuffled_indices]
            y = labels  # np.array([numpy.arange(9)[l==1][0] for l in labels])
            n_classes = y.max() + 1
            i_labeled = []
            for c in range(n_classes):
                if dataname=='IP':
                    if num == 10:
                        i = indices[y == c][:num]
                    if num == 20:

                        if c + 1 == 7:
                            i = indices[y == c][:10]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:10]  # 50
                        else:
                            i = indices[y == c][:num]  # 50

                    if num == 50:
                        if c + 1 == 1:
                            i = indices[y == c][:26]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:16]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:11]  # 50
                        else:
                            i = indices[y == c][:num]  # 50

                    if num == 80:
                        if c + 1 == 1:
                            i = indices[y == c][:26]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:16]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:11]  # 50
                        elif c + 1 == 16:
                            i = indices[y == c][:60]  # 50
                        else:
                            i = indices[y == c][:num]  # 50
                    if num == 100:
                        if c + 1 == 1:
                            i = indices[y == c][:33]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:20]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:14]  # 50
                        elif c + 1 == 16:
                            i = indices[y == c][:75]  # 50
                        else:
                            i = indices[y == c][:num]  # 50
                    if num == 150:
                        if c + 1 == 1:
                            i = indices[y == c][:36]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:22]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:16]  # 50
                        elif c + 1 == 16:
                            i = indices[y == c][:80]  # 50
                        else:
                            i = indices[y == c][:num]  # 50
                    if num == 200:
                        if c + 1 == 1:
                            i = indices[y == c][:39]  # 50
                        elif c + 1 == 7:
                            i = indices[y == c][:24]  # 50
                        elif c + 1 == 9:
                            i = indices[y == c][:18]  # 50
                        elif c + 1 == 16:
                            i = indices[y == c][:85]  # 50
                        else:
                            i = indices[y == c][:num]  # 50
                else:
                    i = indices[y == c][:num]
                    # i = indices[y==c][:10]#50
                i_labeled += list(i)
            l_images = images[i_labeled]
            l_labels = y[i_labeled]

            band_num=3 #3*n

            x_train_band = gain_neighborhood_band(l_images.reshape(l_images.shape[0],1,1,l_images.shape[1]),l_images.shape[1],3,1)#(x_true, band, band_patch, patch)
            x_test_band = gain_neighborhood_band(images.reshape(images.shape[0],1,1,images.shape[1]),images.shape[1],3,1)#(x_true, band, band_patch, patch)
            x_true_band = gain_neighborhood_band(data_global.reshape(data_global.shape[0],1,1,data_global.shape[1]),data_global.shape[1],3,1)#(x_true, band, band_patch, patch)


            x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(torch.FloatTensor)  # [695, 200, 7, 7]
            y_train = torch.from_numpy(l_labels).type(torch.LongTensor)  # [695]
            Label_train = Data.TensorDataset(x_train, y_train)
            x_test = torch.from_numpy(x_test_band.transpose(0, 2, 1)).type(torch.FloatTensor)  # [9671, 200, 7, 7]
            y_test = torch.from_numpy(y).type(torch.LongTensor)  # [9671]
            Label_test = Data.TensorDataset(x_test, y_test)
            x_true = torch.from_numpy(x_true_band.transpose(0, 2, 1)).type(torch.FloatTensor)
            y_true = torch.from_numpy(gt.flatten()).type(torch.LongTensor)

            Label_true = Data.TensorDataset(x_true, y_true)

            label_train_loader = Data.DataLoader(Label_train, batch_size=64, shuffle=True)#(Label_train, batch_size=args.batch_size, shuffle=True)
            label_test_loader = Data.DataLoader(Label_test, batch_size=64, shuffle=False)#(Label_test, batch_size=args.batch_size, shuffle=True)
            label_true_loader = Data.DataLoader(Label_true, batch_size=64, shuffle=False)#(Label_true, batch_size=100, shuffle=False)

            # -------------------------------------------------------------------------------
            # create model
            model = ViT(
                image_size=1,#args.patches,
                near_band=band_num,#args.band_patches,
                num_patches=num_patches,#band,
                num_classes=16,#num_classes,
                dim=64,
                depth=5,
                heads=4,
                mlp_dim=8,
                dropout=0.1,
                emb_dropout=0.1,
                mode='CAF'#args.mode
            )
            model = model.cuda()
            # criterion
            criterion = nn.CrossEntropyLoss().cuda()
            # optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)

            print("start training")
            train_time1 = time.time()
            for epoch in range(args.epoches):
                scheduler.step()

                # train model
                model.train()
                train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
                OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
                print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                      .format(epoch + 1, train_obj, train_acc))

                #if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                #    model.eval()
                #    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
                #    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

            train_time2 = time.time()

            print("**************************************************")

            model.eval()

            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)

            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            print(OA2)
            '''
            svc = SVC(kernel='rbf', class_weight='balanced',C=512, gamma=0.125)
            svc.fit(l_images, l_labels)
            score = svc.score(data, label)
            print('精度为%s' % score)
            '''

            tes_time1 = time.time()
            pred_global =_test_epoch(model, label_true_loader, criterion, optimizer)
            tes_time2 = time.time()
            generate_png(gt, pred_global, dataname, m, n,num)

            #ac = get_accuracy(pred, label)
            ac = get_accuracy(tar_v,pre_v )



            Experiment_result[0, iter_num] = ac[-1] * 100  # OA
            Experiment_result[1, iter_num] = np.mean(ac[:-2]) * 100  # AA
            Experiment_result[2, iter_num] = ac[-2] * 100  # Kappa
            Experiment_result[3, iter_num] = train_time2 - train_time1
            Experiment_result[4, iter_num] = tes_time2 - tes_time1
            Experiment_result[5:, iter_num] = ac[:-2] * 100

            print('########### Experiment {}，Model assessment Finished！ ###########'.format(iter_num))



            ########## mean value & standard deviation #############

        Experiment_result[:, -2] = np.mean(Experiment_result[:, 0:-2], axis=1)  # 计算均值
        Experiment_result[:, -1] = np.std(Experiment_result[:, 0:-2], axis=1)  # 计算平均差

        print('OA_std={}'.format(Experiment_result[0, -1]))
        print('AA_std={}'.format(Experiment_result[1, -1]))
        print('Kappa_std={}'.format(Experiment_result[2, -1]))
        print('time training cost_std{:.4f} secs'.format(Experiment_result[3, -1]))
        print('time testing cost_std{:.4f} secs'.format(Experiment_result[4, -1]))
        for i in range(Experiment_result.shape[0]):
            if i > 4:
                print('Class_{}: accuracy_std {:.4f}.'.format(i - 4, Experiment_result[i, -1]))  # 均差

        day = datetime.datetime.now()
        day_str = day.strftime('%m_%d_%H_%M')

        f = open('./record/' + str(day_str) + '_' + dataname + '_' + model_name +'_'+str(num)+ 'num.txt', 'w')
        for i in range(Experiment_result.shape[0]):
            f.write(str(i + 1) + ':' + str(round(Experiment_result[i, -2],2)) + '+/-' + str(round(Experiment_result[i, -1],2)) + '\n')
        for i in range(Experiment_result.shape[1] - 2):
            f.write('Experiment_num' + str(i) + '_OA:' + str(Experiment_result[0, i]) + '\n')
        f.close()
