import math
import random
import time
import warnings
import pandas as pd
import torch  # torch==1.7.1
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import re
# import tensorflow as tf
import pickle
import numpy as np
from tqdm import tqdm

# from CNN import tokenizer
from pytorch_pretrained_bert import BertModel, BertTokenizer

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, roc_curve
from sklearn import metrics

import json
from textattack.model_args import TEXTATTACK_MODELS
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.shared import utils
import torch.distributions as td  # 用于高斯分布
import matplotlib.pyplot as plt

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

# seed = 1
# torch.manual_seed(seed)  # 为CPU设置随机种子
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


MAX_WORD = 10000  # 只保留最高频的10000词
MAX_LEN = 30  # 句子统一长度为200
word_count = {}  # 词-词出现的词数 词典


def enable_dropout(model):
    """
    在测试阶段使得dropout可用
    :param model:
    :return:
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def fix_batch_normalization(model):
    """
    在测试阶段关掉BN
    在测试阶段关掉BN
    :param model:
    :return:
    """
    for m in model.modules():
        if m.__class__.__name__.find('BatchNorm') != -1:
            m.eval()


# 分词方法
def tokenizer(sentence):
    return sentence.split()


# 定义Dataset
class MyDataset(Dataset):
    def __init__(self, text_file):
        # #     # file = open(text_path, 'r', encoding='UTF-8-sig')
        # #     self.text_with_tag = text_file.readlines()  # 文本标签与内容
        # def get_text_label(self, text_file):
        list = []
        for i in range(len(text_file)):
            each_list = []

            line = text_file[i]
            label = line[0][0]  # 标签信息
            # label = int(str(line[0].encode('utf-8')))
            # label = int(line[0])
            text = line[0][2]  # 文本信息
            user = line[1]
            machine_mean=line[2][0]
            machine_std=line[2][1]
            each_list.append(label)
            each_list.append(text)
            each_list.append(user)
            each_list.append(machine_mean)
            each_list.append(machine_std)
            list.append(each_list)
        # print('list', list)
        self.text_with_tag = list

    def __getitem__(self, index):  # 重写getitem
        line = self.text_with_tag[index]  # 获取一个样本的标签和文本信息
        # print('line:', line)
        label = int(line[0])  # 标签信息
        # print('label', label)

        # print (label)
        # label = int(str(line[0].encode('utf-8')))
        # label = int(line[0])
        text = line[1]

        # print('text', text)
        user = line[2]  # 文本信息
        user = torch.tensor(user)
        machine_mean=line[3]
        machine_std=line[4]
        # print('user', user)
        # print('dataset_user:', user)
        # print('dataset_user_type:', type(user))

        return text, label, user,machine_mean,machine_std

    def __len__(self):
        return len(self.text_with_tag)


def text_transform(sentence_list, vocab):
    sentence_index_list = []
    for sentence in sentence_list:
        sentence_idx = [vocab[token] if token in vocab.keys() else vocab['<UNK>'] for token in
                        tokenizer(sentence)]  # 句子分词转为id

        if len(sentence_idx) < MAX_LEN:
            for i in range(MAX_LEN - len(sentence_idx)):  # 对长度不够的句子进行PAD填充
                sentence_idx.append(vocab['<PAD>'])

        sentence_idx = sentence_idx[:MAX_LEN]  # 取前MAX_LEN长度
        sentence_index_list.append(sentence_idx)
    return torch.LongTensor(sentence_index_list)  # 将转为idx的词转为tensor


def machine_distribution(machine_model, train_x, batch_size, forward_passes):  # 一个batch一起算batch_size个分布
    dropout_predictions = torch.zeros((forward_passes, batch_size, 2))
    for i in range(forward_passes):
        pred = machine_model(train_x)  # tensor(128,2)
        # pred=pred.detach().cpu().numpy()
        dropout_predictions[i] = pred  # dropout_predictions -> shape -> (forward_passes, batch_size, n_classes)
    # 计算均值
    machine_mean = torch.mean(dropout_predictions, axis=0)  # mean -> shape -> (n_samples, n_classes)
    # machine_mean=torch.tensor(mean, requires_grad=True).cuda()
    # machine_mean = torch.from_numpy(mean).cuda()
    # print("mean:", mean)
    # 计算方差
    machine_variance = torch.std(dropout_predictions, axis=0)  # variance -> shape -> (n_samples, n_classes)
    # machine_variance = torch.from_numpy(variance).cuda()
    # machine_variance=torch.tensor(variance, requires_grad=True).cuda()

    # print("variance:", variance)

    return machine_mean, machine_variance


# 在深度学习中，通常使用对数方差（log-variance）而不是方差本身对模型进行建模。这是因为方差必须是正的，但是对数方差可以具有任意的实数值，并且更适合进行梯度计算。在重参数化时，我们需要使用标准差而不是对数方差。因此，我们需要将对数方差还原为标准差。
def reparameterize(mean, logstd, batch_size):
    samples = torch.empty(batch_size, 0)
    std = torch.exp(logstd)  # 得到标准差
    std_size = std.size()
    for i in range(20):
        # eps = torch.randn_like(std) #从标准正态分布中采样一个ε（epsilon），大小与标准差一样。
        eps1 = np.random.normal(0, 1, std_size)  # 参数分别为均值、标准差和生成样本数
        eps = torch.tensor(eps1).cuda()
        sample = eps * std + mean  # std和mean的大小都是(batch_size,1) 所以，sample钰std同大小为(batch_size,1)
        sample = sample.unsqueeze(-1)
        samples = torch.hstack((samples.cuda(), sample.cuda()))  # 水平拼接每个sample,得到samples(batch_size,10)
    return samples


class User(nn.Module):
    def __init__(self, user_score):
        super(User, self).__init__()
        # self.user_weigth = nn.parameter()
        # self.w = nn.Parameter(torch.Tensor([0.5, 0.25, 0.15, 0.1]), requires_grad=True)

        self.w = nn.Parameter(user_score, requires_grad=True)

    #
    # def forward(self, x):
    #     list=[]
    #     #给user编号一个所有0、1、2、3、4.。。。跟初始化的tensor顺序一样
    #     #输入的格式[news,[,user_respond，user_index][user_index,user_respond]....]
    #     for i in x:
    #
    #         user_index = x[0]
    #         w1 = self.w[user_index]
    #         user_respond = x[1]
    #         list.append('user_weight:',w1,'user_respond:', user_respond)

    def forward(self, inputs):  # inputs=[[1 ,id,news_text] ,[[respond 特异性 个性 K]，[respond 特异性 个性 K]，[respond 特异性 个性 K]]]

        input = torch.split(inputs, 1)  # input_size：tensor[batch_size]  每条新闻有50个评论（评论又有好几维的特征）被放在一个list里面。
        # real_pro = real_pro.clone().detach().float().cuda()
        # fake_pro = fake_pro.clone().detach().float().cuda()

        stance_for_Judge_PAD = torch.tensor(3, dtype=torch.float64)
        # stance_for_Judge_PAD = stance_for_Judge_PAD.clone().detach().float().cuda()

        stance_for_Judge = torch.tensor(0, dtype=torch.float64)
        # stance_for_Judge_zero = torch.tensor(1, dtype=torch.float64)

        # stance_for_Judge = stance_for_Judge.clone().detach().float().cuda()

        user_pre = torch.zeros((0, 2))
        user_distribution = torch.zeros((0, 2))
        user_theta_list = torch.zeros((0, 2))
        for each_batch in input:  # each_batch是一个新闻的所有反应  for循环的次数是输入的batch_size
            response_num = 0
            real_pro = torch.tensor([0], dtype=torch.float64, requires_grad=True)
            fake_pro = torch.tensor([0], dtype=torch.float64, requires_grad=True)
            each_batch = each_batch.squeeze(0)  # 降维，函数功能：去除size为1的维度，包括行和列。当维度大于等于2时，squeeze()无作用。
            for line in each_batch:  # line是一条新闻的一条反应  for循环的次数是新闻评论数，根据PAD填充应该都是50

                stance = line[0]  # tensor
                # stance = stance.float()
                stance = stance.double()

                # print('stance:', stance)
                # print('stance_type:', type(stance))
                if stance.equal(stance_for_Judge_PAD):
                    break
                else:
                    response_num += 1
                    user_index = line[2].type(torch.long).cuda()

                    user_weight = self.w[user_index]  # 学习更新过好，有的weight是负数了
                    if stance.equal(stance_for_Judge):
                        real_pro = real_pro.cuda() + user_weight  # 这里的整型0 和tensor能相加嘛
                    else:
                        fake_pro = fake_pro.cuda() + user_weight
            real_pro = real_pro.cuda()
            fake_pro = fake_pro.cuda()
            softmax_input = torch.cat((real_pro, fake_pro), dim=0)
            crowd_respond = F.softmax(softmax_input, dim=0)
            user_pre = torch.vstack((user_pre.cuda(), crowd_respond.cuda().resize(1, 2)))  # user_pre={Tensor:(1,2)}

            # if real_pro==0 and fake_pro==0:
            #     real_pro_standar= real_pro+0.5
            #     fake_pro_standar= fake_pro+0.5
            # else:
            #     real_pro_standar = real_pro / (real_pro + fake_pro)
            #     fake_pro_standar = fake_pro / (real_pro + fake_pro)

            # crowd_mean = fake_pro_standar
            # crowd_var = fake_pro_standar * real_pro_standar
            user_theta = crowd_respond * response_num
            user_theta_list = torch.vstack((user_theta_list.cuda(), user_theta.cuda().resize(1, 2)))
            # beta_a=crowd_respond[1]*response_num
            # beta_b=crowd_respond[0]*response_num
            beta_a = user_theta[1]
            beta_b = user_theta[0]
            beta_dist = td.beta.Beta(beta_a, beta_b)
            crowd_mean = beta_dist.mean
            crowd_var = beta_dist.variance
            crowd_std = torch.sqrt(crowd_var)
            # crowd_mean = crowd_respond[1]
            # crowd_var = crowd_respond[0] * crowd_respond[1]
            crowd_mean = crowd_mean.unsqueeze(-1)
            crowd_std = crowd_std.unsqueeze(-1)
            # crowd_ab_pairs=torch.cat((crowd_respond[0], crowd_respond[1]), dim=0)
            crowd_distribution = torch.cat((crowd_mean, crowd_std), dim=0)
            user_distribution = torch.vstack((user_distribution.cuda(), crowd_distribution.cuda().resize(1, 2)))
            # user_ab_pairs = torch.vstack((user_ab_pairs.cuda(), crowd_ab_pairs.cuda().resize(1, 2)))

            # user_pre=torch.tensor(user_pre)
            # user_distribution=torch.tensor(user_distribution)
        return user_pre, user_distribution, user_theta_list


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activate = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, machine_mean, machine_variance, user_mean, user_variance):  # tendor(32,)
        # input_mean=torch.stack([machine_mean,user_mean], dim=1)
        # input_var=torch.stack([machine_variance,user_variance], dim=1)
        # all_inputs=torch.stack([input_mean,input_var], dim=1)
        all_inputs = torch.stack([machine_mean, machine_variance, user_mean, user_variance], dim=1)
        # machine_variance=torch.sqrt(machine_variance)
        # all_inputs = torch.stack([machine_mean, machine_variance], dim=1)
        all_inputs = all_inputs.float()
        out = self.layer1(all_inputs)
        out = self.activate(out)
        out = self.layer2(out)
        encoder_mu = out[:, 0]
        encoder_mu = self.sigmoid(encoder_mu)
        encoder_logstd = out[:, 1]  # encoder_logSD表示的是encoder输出的标准差的log，sd为标准差
        # encoder_logstd = torch.tensor([-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8,-1.8]).cuda()
        return encoder_mu, encoder_logstd


class ELL(nn.Module):
    def __init__(self):
        super(ELL, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, samples, y_true, machine_mu, machine_var, user_theta, encoder_mu,
                batch_size):  # x: input data # mu: mean of the Gaussian distribution # logvar: log variance of the Gaussian distribution
        eps = 1e-8
        all_loss = torch.empty(0, 1).cuda()
        gaussi_all_loss = torch.empty(0, 1).cuda()
        beta_all_loss = torch.empty(0, 1).cuda()
        pre_all_loss = torch.empty(0, 1).cuda()
        for i in range(batch_size):  # 一个样本一个样本进行损失计算
            each_samples = samples[i, :]
            each_y_true = y_true[i]
            each_machine_mu = machine_mu[i]
            each_machine_var = machine_var[i]
            each_user_theta = user_theta[i]
            each_encoder_mu = encoder_mu[i]

            # machine_dist = td.normal.Normal(each_machine_mu, torch.sqrt(each_machine_var))  #得到正太分布
            machine_dist = td.normal.Normal(each_machine_mu, torch.sqrt(each_machine_var))  # 得到正太分布
            # print('each_samples:',each_samples)
            gaussian_log_likelihood = machine_dist.log_prob(each_samples)  # 对数似然
            gaussian_prob = torch.exp(gaussian_log_likelihood)  # 将对数概率转换为概率值
            gaussian_loss = -torch.mean(gaussian_prob)
            gaussian_loss = torch.exp(gaussian_loss)

            # logvar=machine_var.log()
            # gaussian_loss = 0.5 * ((samples - machine_mu) ** 2 / torch.exp(logvar) + logvar).mean()
            beta_a = each_user_theta[1]
            beta_b = each_user_theta[0]
            beta_dist = td.beta.Beta(beta_a, beta_b)
            beta_log_likelihood = beta_dist.log_prob(each_samples)  # 对数似然
            beta_prob = torch.exp(beta_log_likelihood)  # 将对数概率转换为概率值
            beta_loss = -torch.mean(beta_prob)
            beta_loss = torch.exp(beta_loss)

            precision_loss = F.binary_cross_entropy_with_logits(each_encoder_mu, each_y_true.float())
            # precision_loss = torch.nn.CrossEntropyLoss(y_pred, y_true, reduction='none')
            loss = gaussian_loss + beta_loss + precision_loss
            # loss = beta_loss
            # loss = gaussian_loss+beta_loss
            # loss = precision_loss
            # print('gaussian_loss:',gaussian_loss)
            # print('beta_loss:',beta_loss)
            # print('precision_loss:',precision_loss)
            all_loss = torch.vstack((all_loss, loss))
            gaussi_all_loss = torch.vstack((gaussi_all_loss, gaussian_loss))
            beta_all_loss = torch.vstack((beta_all_loss, beta_loss))
            pre_all_loss = torch.vstack((pre_all_loss, precision_loss))
        final_loss = torch.mean(all_loss)
        final_gaussi_all_loss = torch.mean(gaussi_all_loss)
        final_beta_all_loss = torch.mean(beta_all_loss)
        final_pre_all_loss = torch.mean(pre_all_loss)
        # print('-->final_loss_grad_value:', final_loss.grad)
        return final_loss, final_gaussi_all_loss, final_beta_all_loss, final_pre_all_loss


# 模型训练
def train(encoder_model,  user_model, train_data, ver_data, forward_passes, epoch, batch_size,
          test_batch_size_value, result_save_path):
    print('train model')
    best_inte_acc = 0.01
    encoder_model = encoder_model.cuda()
    user_model = user_model.cuda()
    for name2, parameter2 in user_model.named_parameters():
        parameter2.requires_grad = False

    encoder_model.train()
    user_model.eval()

    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    ver_avg_acc_list = []

    train_gaussian_loss_list = []
    train_beta_loss_list = []
    train_pre_loss_list = []
    # ver_all_acc_list = []
    # ver_machine_acc_list = []
    # ver_user_acc_list = []
    # ver_inte_acc_list = []
    # 定义损失函数和优化器
    criterion = ELL()
    #encoder_optimizer = torch.optim.SGD(encoder_model.parameters(), lr=5e-2)
    #encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=9e-3)
    encoder_optimizer = torch.optim.Adagrad(encoder_model.parameters(), lr=1e-2, lr_decay=0.001, weight_decay=0.001,
                                            initial_accumulator_value=0)
    # machine_optimizer = torch.optim.Adagrad(machine_model.parameters(), lr=5e-3, lr_decay=0, weight_decay=0,
    #                                         initial_accumulator_value=0)
    # user_optimizer = torch.optim.Adagrad(user_model.parameters(), lr=8e-2, lr_decay=0, weight_decay=0,initial_accumulator_value=0)
    # out_train = open(r'dataset/CNN_train_score_weigth_acc_out_test.txt', 'w', encoding='UTF-8-sig')
    # out_ver = open(r'dataset/CNN_ver_score_weigth_acc_out_test.txt', 'w', encoding='UTF-8-sig')
    mean_epoch_gap = np.empty((0, 1))
    std_epoch_gap = np.empty((0, 1))
    epoch_size = epoch
    for epoch in tqdm(range(epoch)):
        machine_variance_list = []
        encoder_std_list = []

        avg_loss = 0  # 平均损失
        avg_acc = 0  # 平均准确率

        avg_gaussian_loss = 0
        avg_beta_loss = 0
        avg_pre_loss = 0

        mean_all_gap = np.empty((0, 1))
        std_all_gap = np.empty((0, 1))
        for idx, (text, label, user,machine_mean,machine_variance) in enumerate(tqdm(train_data)):
            #train_x = text_transform(text, vocab).cuda()
            y_true = label.cuda()
            machine_mean=machine_mean.cuda()
            machine_variance=machine_variance.cuda()
            encoder_optimizer.zero_grad()
            # machine_optimizer.zero_grad()
            # user_optimizer.zero_grad()

            user_pre, user_distribution, user_theta = user_model(user)
            user_mean = user_distribution[:, 0]
            user_variance = user_distribution[:, 1]
            encoder_mu, encoder_logstd = encoder_model(machine_mean, machine_variance, user_mean,
                                                       user_variance)  # pred是mlp输出的二维向量
            samples = reparameterize(encoder_mu, encoder_logstd, batch_size=batch_size)  # tensor(128,10)
            encoder_std = torch.exp(encoder_logstd)  # 得到标准差
            encoder_mu_bool_tensor = torch.gt(encoder_mu,
                                              0.5)  # 使用torch.gt()函数对Tensor的每个元素进行逐元素比较，生成一个布尔型的Tensor，其中值为True的元素表示对应的原Tensor元素大于0.5。
            encoder_mu_label = torch.where(encoder_mu_bool_tensor, torch.ones_like(encoder_mu),
                                           torch.zeros_like(encoder_mu))  # 逐元素操作，将 True 替换为 1，False 替换为 0

            loss, gaussian_loss, beta_loss, pre_loss = criterion(samples=samples, y_true=y_true,
                                                                 machine_mu=machine_mean, machine_var=machine_variance,
                                                                 user_theta=user_theta, encoder_mu=encoder_mu,
                                                                 batch_size=batch_size)
            loss.requires_grad_(True)
            # #打印梯度
            # for name, parms in user_model.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")

            # Inte_label_test = pred.max(dim=1)[1]
            # # machine_label_test = machine_out.max(dim=1)[1]
            # # user_label_test = user_out.max(dim=1)[1]
            #
            # loss = criterion(pred.log(), train_y)
            loss.backward()
            # machine_optimizer.step()
            # user_optimizer.step()
            encoder_optimizer.step()

            avg_loss += loss.item()
            avg_acc += accuracy(encoder_mu_label, y_true)

            avg_gaussian_loss += gaussian_loss.item()
            avg_beta_loss += beta_loss.item()
            avg_pre_loss += pre_loss.item()

            machine_variance_list.append(machine_variance)
            encoder_std_list.append(encoder_std)

            # 计算encoder和machine的分布差距
            machine_mean2 = machine_mean.reshape(batch_size, 1)
            encoder_mu2 = encoder_mu.reshape(batch_size, 1)
            machine_variance2 = machine_variance.reshape(batch_size, 1)
            encoder_std2 = encoder_std.reshape(batch_size, 1)
            mean_gap = abs(machine_mean2 - encoder_mu2)
            std_gap = abs(machine_variance2 - encoder_std2)
            mean_all_gap = np.vstack((mean_all_gap, mean_gap.cpu().detach().numpy()))
            std_all_gap = np.vstack((std_all_gap, std_gap.cpu().detach().numpy()))
            mean_all_gap2 = np.mean(mean_all_gap)
            std_all_gap2 = np.mean(std_all_gap)
        mean_epoch_gap = np.vstack((mean_epoch_gap,
                                    mean_all_gap2))  # 插入新维度的意思，比如将一维数组变成二维数组，二维变成三维等
        std_epoch_gap = np.vstack((std_epoch_gap,
                                   std_all_gap2))  # 插入新维度的意思，比如将一维数组变成二维数组，二维变成三维等
        # 一个epoch结束后，计算平均loss和评平均acc
        avg_loss = avg_loss / len(train_data)
        avg_acc = avg_acc / len(train_data)

        avg_gaussian_loss = avg_gaussian_loss / len(train_data)
        avg_beta_loss = avg_beta_loss / len(train_data)
        avg_pre_loss = avg_pre_loss / len(train_data)

        epoch_list.append(epoch)
        train_loss_list.append(avg_loss)
        train_acc_list.append(avg_acc)

        train_gaussian_loss_list.append(avg_gaussian_loss)
        train_beta_loss_list.append(avg_beta_loss)
        train_pre_loss_list.append(avg_pre_loss)

        print("avg_loss:", avg_loss, " train_avg_acc:,", avg_acc)
        # print("train_encoder_mu", encoder_mu)
        # print("train_encoder_mu_label", encoder_mu_label)
        # print("train_encoder_std", std)
        # write_list1.append(epoch)
        # write_list1.append(avg_loss)
        # write_list1.append(avg_acc)

        # 开始验证
        print("开始验证++++++++++++++++++++++++++")
        evaluations = test( user_model, encoder_model, ver_data, test_batch_size_value, forward_passes, result_save_path)
        #evaluations = ceshi(inte_model, machine_model, user_model, ver_data,vocab, forward_passes=forward_passes,
        #                    batch_size=test_batch_size_value, result_save_path=result_save_path)

        # ver_machine_acc = evaluations[0]
        # ver_user_acc = evaluations[1]
        ver_inte_acc = evaluations[2]

        ver_avg_acc_list.append(ver_inte_acc)

        # if ver_inte_acc > ver_machine_acc and ver_inte_acc > best_inte_acc:
        if ver_inte_acc > best_inte_acc:
            best_machine_variance_list = machine_variance_list
            best_encoder_std_list = encoder_std_list
            # 保存训练完成后的模型参数
            # print(f'Save model! Best validation accuracy is {val_acc:.5f}
            best_inte_acc = ver_inte_acc
            print('保存最好的模型==========================')
            print(f'Save model! Best validation accuracy is {ver_inte_acc}')
            # 保存训练完成后的模型参数
            # machine_model.save_pretrained(result_save_path)
            # 保存训练完成后的模型参数
            torch.save(encoder_model.state_dict(),
                       result_save_path + 'bert_dropout_score_weigt_encoder_model_parameter_bata.pkl')
            # torch.save(machine_model.state_dict(),
            #            result_save_path + 'CNN_news_parameter_seed_datasetscore.pkl')
            # torch.save(user_model.state_dict(),
            #            result_save_path + 'CNN_dropout_score_weigth_user_model_parameter_test1.pkl')

        # print('ver_machine_acc:', ver_machine_acc, 'ver_user_pre:', ver_user_acc, 'ver_inte_rec:', ver_inte_acc)
    #     ver_machine_acc_list.append(ver_machine_acc)
    #     ver_user_acc_list.append(ver_user_acc)
    #     ver_inte_acc_list.append(ver_inte_acc)
    # ver_all_acc_list.append([ver_machine_acc_list, ver_user_acc_list, ver_inte_acc_list])
    # data = np.array(ver_all_acc_list[0]).T
    # data_pd = pd.DataFrame(data, columns=['machine_result', 'user_result', 'encoder_result'])
    # data_pd.to_csv(result_save_path + 'ver_CNN_dropout_user_score_weight_acc.csv')
    temp_list = []
    for i in range(1, epoch_size + 1):
        temp_list.append(i)

    # for b in range(3):
    #     plt.plot(temp_list, mean_epoch_gap[b], label=b)
    plt.plot(temp_list, mean_epoch_gap)
    plt.title("input_mean VS output_mean")
    # 设置 x,y 轴取值范围
    plt.xlabel("epoch")
    plt.ylabel("△")
    # plt.legend()
    plt.show()

    plt.plot(temp_list, std_epoch_gap)
    plt.title("input_std VS output_std")
    # 设置 x,y 轴取值范围
    plt.xlabel("epoch")
    plt.ylabel("△")
    # plt.legend()
    plt.show()

    plt.plot(epoch_list, train_loss_list)

    # 添加标题和轴标签。
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # 显示图形。
    plt.show()

    # print("best_machine_variance_list:", best_machine_variance_list)
    # print("best_encoder_std_list:", best_encoder_std_list)
    plt.plot(epoch_list, train_gaussian_loss_list)

    # 添加标题和轴标签。
    plt.title("Training Gaussian Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # 显示图形。
    plt.show()

    plt.plot(epoch_list, train_beta_loss_list)

    # 添加标题和轴标签。
    plt.title("Training Beta Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # 显示图形。
    plt.show()

    plt.plot(epoch_list, train_pre_loss_list)

    # 添加标题和轴标签。
    plt.title("Training Precision Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # 显示图形。
    plt.show()

    plt.plot(epoch_list, train_acc_list)

    # 添加标题和轴标签。
    plt.title("Train acc over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Train acc")

    # 显示图形。
    plt.show()

    plt.plot(epoch_list, ver_avg_acc_list)

    # 添加标题和轴标签。
    plt.title("Verification acc over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Verification acc")

    # 显示图形。
    plt.show()

    #     ver_avg_acc = 0
    #     for idx, (text, label, user) in enumerate(tqdm(ver_data)):
    #         train_x = text_transform(text, vocab).cuda()
    #         train_y = label.cuda()
    #
    #         machine_out = machine_model(train_x)
    #         user_out = user_model(user)
    #         pred = inte_model(machine_out, user_out)
    #         # print(pred)
    #         label_test = pred.max(dim=1)[1]
    #         print(label_test)
    #
    #         ver_avg_acc += accuracy(pred, train_y)
    #     # 一个epoch结束后，计算平均loss和评平均acc
    #     ver_avg_acc = ver_avg_acc / len(ver_data)
    #     print("ver_avg_acc:,", ver_avg_acc)
    #     write_list.append(epoch)
    #     write_list.append(ver_avg_acc)
    # out_train.write(str(write_list1))
    # out_train.close()
    # out_ver.write(str(write_list))
    # out_ver.close()


# 不打开dropout测试
def test(user_model, encoder_model, test_data, batch_size, forward_passes,  result_save_path):
    print('test model')
    encoder_model = encoder_model.cuda()
    user_model = user_model.cuda()
    encoder_model.eval()
    user_model.eval()
    avg_acc = 0
    avg_pre = 0
    avg_rec = 0
    avg_f = 0
    avg_auc = 0
    machine_avg_acc = 0
    machine_avg_pre = 0
    machine_avg_rec = 0
    machine_avg_f = 0
    machine_avg_auc = 0
    user_avg_acc = 0
    user_avg_pre = 0
    user_avg_rec = 0
    user_avg_f = 0
    user_avg_auc = 0
    all_distribution_list=[]
    for idx, (text, label, user,machine_mean,machine_variance) in enumerate(tqdm(test_data)):
        #train_x = text_transform(text, vocab).cuda()
        y_true = label.cuda()
        machine_mean=machine_mean.cuda()
        machine_variance=machine_variance.cuda()
        print(y_true)
        print(machine_mean,'machine_mean')
        user_pre, user_distribution, user_theta = user_model(user)
        user_mean = user_distribution[:, 0]

        user_variance = user_distribution[:, 1]
        encoder_mu, encoder_logstd = encoder_model(machine_mean, machine_variance, user_mean,
                                                   user_variance)  # pred是mlp输出的二维向量
        print(encoder_mu,'encoder_mu')
        enocder_std = torch.exp(encoder_logstd)
        encoder_mu_bool_tensor = torch.gt(encoder_mu,
                                          0.5)  # 使用torch.gt()函数对Tensor的每个元素进行逐元素比较，生成一个布尔型的Tensor，其中值为True的元素表示对应的原Tensor元素大于0.5。
        encoder_mu_label = torch.where(encoder_mu_bool_tensor, torch.ones_like(encoder_mu),
                                       torch.zeros_like(encoder_mu))  # 逐元素操作，将 True 替换为 1，False 替换为 0

        # test_x = text_transform(text, vocab).cuda()
        # test_y = label.cuda()
        # machine_out = machine_model(test_x)
        # user_out = user_model(user)
        # pred = inte_model(machine_out, user_out)
        # Inte_label_test = pred.max(dim=1)[1]
        machine_mean_bool_tensor = torch.gt(machine_mean,
                                          0.5)  # 使用torch.gt()函数对Tensor的每个元素进行逐元素比较，生成一个布尔型的Tensor，其中值为True的元素表示对应的原Tensor元素大于0.5。
        machine_label_test = torch.where(machine_mean_bool_tensor, torch.ones_like(machine_mean),
                                       torch.zeros_like(machine_mean))
        user_label_test = user_pre.max(dim=1)[1]

        machine_avg_acc += accuracy(machine_label_test, y_true)



        user_avg_acc += accuracy(user_label_test, y_true)
        user_avg_pre += precision(user_label_test, y_true)
        user_avg_rec += recall(user_label_test, y_true)
        user_avg_f += f1(user_label_test, y_true)
        user_avg_auc += auc_2d(user_pre, y_true)


        avg_acc += accuracy(encoder_mu_label, y_true)
        avg_pre += precision(encoder_mu_label, y_true)
        avg_rec += recall(encoder_mu_label, y_true)
        avg_f += f1(encoder_mu_label, y_true)
        avg_auc += auc(encoder_mu, y_true)
        # avg_roc += roc(encoder_mu_label, y_true)
    # machine_avg_acc = machine_avg_acc / len(test_data)
    machine_acc = machine_avg_acc / len(test_data)
    machine_pre = machine_avg_pre / len(test_data)
    machine_rec = machine_avg_rec / len(test_data)
    machine_f = machine_avg_f / len(test_data)
    machine_auc_result = machine_avg_auc / len(test_data)

    # user_avg_acc = user_avg_acc / len(test_data)
    user_acc = user_avg_acc / len(test_data)
    user_pre = user_avg_pre / len(test_data)
    user_rec = user_avg_rec / len(test_data)
    user_f = user_avg_f / len(test_data)
    user_auc_result = user_avg_auc / len(test_data)

    acc = avg_acc / len(test_data)
    pre = avg_pre / len(test_data)
    rec = avg_rec / len(test_data)
    f = avg_f / len(test_data)
    auc_result = avg_auc / len(test_data)
    # roc_result =avg_roc / len(test_data)

    # with open(result_save_path + "CNN_dropout_news_user_score_weight_roc_update_epoch5.pkl", "wb") as fo:
    #    pickle.dump(roc_result, fo)
    # print('machine_acc:', machine_acc, "machine_pre:", machine_pre, "machine_rec:", machine_rec,
    #       "machine_f:", machine_f, "machine_auc_result:", machine_auc_result,'user_acc:', user_acc, "user_pre:", user_pre, "user_rec:", user_rec,
    #       "user_f:", user_f, "user_auc
    #       _result:", user_auc_result,'acc:', acc, "pre:", pre, "rec:", rec,
    #      "f:", f, "auc_result:", auc_result)
    machine_mean=machine_mean.tolist()
    machine_variance=machine_variance.tolist()
    user_mean=user_mean.tolist()
    user_variance=user_variance.tolist()
    encoder_mu=encoder_mu.tolist()
    enocder_std=enocder_std.tolist()
    beta_a = user_theta[:,1]
    beta_b = user_theta[:,0]
    beta_a=beta_a.tolist()
    beta_b=beta_b.tolist()


    all_distribution_list.append(machine_mean)
    all_distribution_list.append(machine_variance)
    all_distribution_list.append(user_mean)
    all_distribution_list.append(user_variance)
    all_distribution_list.append(encoder_mu)
    all_distribution_list.append(enocder_std)
    all_distribution_list.append(beta_a)
    all_distribution_list.append(beta_b)
    #all_distribution_list2 = np.array(all_distribution_list[0]).T
    data_pd = pd.DataFrame(all_distribution_list)
    data_pd=data_pd.T
    data_pd.columns=['machine_mean', 'machine_std', 'user_mean', 'user_std', 'encoder_mu', 'enocder_std','beta_a','beta_b']
    #data_pd.to_csv(result_save_path + 'test_BERT_news_and_user_all_distribution3.csv')
    print('machine_acc:', machine_acc,'user_acc:', user_acc,'acc:', acc, "pre:", pre, "rec:", rec,
         "f:", f, "auc_result:", auc_result)


    evaluations = [machine_avg_acc, user_avg_acc, acc, pre, rec, f, auc_result]
    return evaluations


'''
# 打开dropout测试
def ceshi(inte_model, machine_model, user_model, test_data, vocab, forward_passes, batch_size, result_save_path):
    print('test model')
    labels = list()
    len_of_test_data = len(test_data)
    len_of_all = len_of_test_data * batch_size

    machine_dropout_predictions = np.empty((0, len_of_all, 2))
    user_dropout_predictions = np.empty((0, len_of_all, 2))
    dropout_predictions = np.empty((0, len_of_all, 2))

    for i in range(forward_passes):
        machine_predictions = np.empty((0, 2))
        user_predictions = np.empty((0, 2))
        predictions = np.empty((0, 2))

        inte_model = inte_model.cuda()
        machine_model = machine_model.cuda()
        user_model = user_model.cuda()

        inte_model.eval()
        enable_dropout(inte_model)
        fix_batch_normalization(inte_model)
        machine_model.eval()
        enable_dropout(machine_model)
        fix_batch_normalization(machine_model)
        user_model.eval()
        enable_dropout(user_model)
        fix_batch_normalization(user_model)

        for idx, (text, label, user) in enumerate(tqdm(test_data)):
            test_x = text_transform(text, vocab).cuda()
            test_y = label.cuda()
            machine_out = machine_model(test_x)
            user_out = user_model(user)
            pred = inte_model(machine_out, user_out)
            machine_predictions = np.vstack((machine_predictions, machine_out.cpu().detach().numpy()))
            user_predictions = np.vstack((user_predictions, user_out.cpu().detach().numpy()))

            # print(pred)
            # label_test = pred.max(dim=1)[1]
            predictions = np.vstack((predictions, pred.cpu().detach().numpy()))

            if i == forward_passes - 1:
                labels.extend(test_y.cpu().numpy())
        machine_dropout_predictions = np.vstack((machine_dropout_predictions,
                                                 machine_predictions[np.newaxis, :, :]))
        user_dropout_predictions = np.vstack((user_dropout_predictions,
                                              user_predictions[np.newaxis, :, :]))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout_predictions -> shape -> (forward_passes, n_samples, n_classes)
        # 计算均值
    machine_mean = np.mean(machine_dropout_predictions, axis=0)
    user_mean = np.mean(user_dropout_predictions, axis=0)

    mean = np.mean(dropout_predictions, axis=0)
    # mean -> shape -> (n_samples, n_classes)
    # print("mean:", mean)
    # 计算方差
    variance = np.var(dropout_predictions, axis=0)
    # variance -> shape -> (n_samples, n_classes)
    # print("variance:", variance)
    machine_mean = torch.from_numpy(machine_mean).cuda()
    user_mean = torch.from_numpy(user_mean).cuda()

    mean = torch.from_numpy(mean).cuda()
    labels = torch.from_numpy(np.array([labels]).T).cuda()
    machine_prediction = machine_mean.data.max(dim=1, keepdim=True)[1]


    # machine_correction = machine_prediction.eq(labels.data.view_as(machine_prediction)).sum()
    # print('\tTest machine_Accuracy:{:.2f}%'.format(100. * machine_correction / len_of_all))
    user_prediction = user_mean.data.max(dim=1, keepdim=True)[1]
    user_acc = accuracy(user_prediction, labels)
    user_p1_r1_f1_result = p1_r1_f1(user_prediction, labels)
    user_pre = user_p1_r1_f1_result[0]
    user_rec = user_p1_r1_f1_result[1]
    user_f = user_p1_r1_f1_result[2]
    user_auc_result = auc(user_mean, labels)
    machine_roc_result = roc(user_mean, labels)
    with open(result_save_path + "CNN_dropout_user_score_weight_update(用户包含反应、积分和索引)_roc_epoch5.pkl", "wb") as fo:
        pickle.dump(machine_roc_result, fo)
    print('user_acc:', user_acc, "user_pre:", user_pre, "user_rec:", user_rec, "user_f:", user_f, 'user_auc_result:',
          user_auc_result)

    machine_acc = accuracy(machine_prediction, labels)
    machine_p1_r1_f1_result = p1_r1_f1(machine_prediction, labels)
    machine_pre = machine_p1_r1_f1_result[0]
    machine_rec = machine_p1_r1_f1_result[1]
    machine_f = machine_p1_r1_f1_result[2]
    machine_auc_result = auc(machine_mean, labels)
    machine_roc_result = roc(machine_mean, labels)
    with open(result_save_path + "CNN_dropout_machine_score_weight_update(用户包含反应、积分和索引)_roc_epoch5.pkl", "wb") as fo:
        pickle.dump(machine_roc_result, fo)
    print('machine_acc:', machine_acc, "machine_pre:", machine_pre, "machine_rec:", machine_rec, "machine_f:",
          machine_f, "machine_auc:", machine_auc_result)

    prediction = mean.data.max(dim=1, keepdim=True)[1]
    file_result = pd.ExcelWriter(result_save_path + "所有结果的对比CNN_dropout_原始数据集_score_weight_update_epoch5.xlsx")
    result_duibi = torch.stack([torch.tensor(labels), machine_prediction, user_prediction, prediction], dim=1)
    # data = {"正确标签": torch.tensor(labels), "机器标签": machine_prediction, "用户标签": user_prediction,'人机结合标签':prediction}
    result_duibi = result_duibi.squeeze()
    result_duibi = result_duibi.cpu().numpy()
    result_duibi = pd.DataFrame(result_duibi)
    result_duibi.columns = ["正确标签", "机器标签", "用户标签", "人机结合标签"]
    result_duibi.to_excel(file_result)
    file_result.save()
    file_result.close()

    acc = accuracy(prediction, labels)
    p1_r1_f1_result = p1_r1_f1(prediction, labels)
    pre = p1_r1_f1_result[0]
    rec = p1_r1_f1_result[1]
    f = p1_r1_f1_result[2]
    auc_result = auc(mean, labels)
    roc_result = roc(mean, labels)
    # torch.save(roc_result, '../roc.pkl')
    with open(result_save_path + "CNN_dropout_news_user_score_weight_roc_update_epoch5.pkl", "wb") as fo:
        pickle.dump(roc_result, fo)
    print('acc:', acc, "pre:", pre, "rec:", rec, "f:", f, "auc:", auc_result)

    # print('\tTest Accuracy:{:.2f}%'.format(100. * correction / len_of_all))
    # print(result_duibi)
    # print('machine_avg_acc:', machine_avg_acc, 'user_avg_acc:', user_avg_acc, 'acc:', acc, "pre:", pre, "rec:", rec,
    #       "f:", f, "auc:", auc_result)
    evaluations = [machine_acc, user_acc, acc, pre, rec, f, auc_result]
    return evaluations
'''


# # 计算预测准确性
# def accuracy(y_pred, y_true):
#     label_pred = y_pred.max(dim=1)[1]
#     acc = len(y_pred) - torch.sum(torch.abs(label_pred - y_true))  # 正确的个数
#     return acc.detach().cpu().numpy() / len(y_pred)

def accuracy(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    # label_pred = y_pred.max(dim=1)[1]
    acc = accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())
    return acc

def precision(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    # label_pred = y_pred.max(dim=1)[1]
    pre = metrics.precision_score(y_true, y_pred,  average='macro')
    return pre

def recall(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    # label_pred = y_pred.max(dim=1)[1]
    rec = metrics.recall_score(y_true, y_pred,  average='macro')
    return rec

def f1(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    # label_pred = y_pred.max(dim=1)[1]
    # label_pred = []
    # label_pred.append(pred1.detach().cpu().numpy())
    f = metrics.f1_score(y_true, y_pred,  average='macro')
    return f

def auc(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    # auc = roc_auc_score(y_true, y_pred[:, 1].detach().numpy())
    auc = roc_auc_score(y_true.detach().numpy(), y_pred.detach().numpy())
    return auc

def auc_2d(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    auc = roc_auc_score(y_true, y_pred[:, 1].detach().numpy())
    # auc = roc_auc_score(y_true.detach().numpy(), y_pred.detach().numpy())
    return auc

def p1_r1_f1(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    # label_pred = y_pred.max(dim=1)[1]
    p1, r1, f1, _ = precision_recall_fscore_support(y_true.detach().numpy(), y_pred.detach().numpy(), pos_label=1,
                                                    average='macro')
    p1_r1_f1 = [p1, r1, f1]
    return p1_r1_f1

def roc(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    # roc = roc_curve(y_true.detach().numpy(), y_pred[:, 1].detach().numpy())
    roc = roc_curve(y_true.detach().numpy(), y_pred.detach().numpy())
    return roc

'''
#打开dropout测试
def ceshi(inte_model, machine_model, user_model, test_data, vocab, forward_passes, batch_size):
    print('test model')
    labels = list()
    len_of_test_data = len(test_data)
    len_of_all = len_of_test_data * batch_size

    machine_dropout_predictions = np.empty((0, len_of_all, 2))
    user_dropout_predictions = np.empty((0, len_of_all, 2))
    dropout_predictions = np.empty((0, len_of_all, 2))

    for i in range(forward_passes):
        machine_predictions = np.empty((0, 2))
        user_predictions = np.empty((0, 2))
        predictions = np.empty((0, 2))

        inte_model = inte_model.cuda()
        machine_model = machine_model.cuda()
        user_model = user_model.cuda()

        inte_model.eval()
        enable_dropout(inte_model)
        fix_batch_normalization(inte_model)
        machine_model.eval()
        enable_dropout(machine_model)
        fix_batch_normalization(machine_model)
        user_model.eval()
        enable_dropout(user_model)
        fix_batch_normalization(user_model)

        for idx, (text, label, user) in enumerate(tqdm(test_data)):
            test_x = text_transform(text, vocab).cuda()
            test_y = label.cuda()
            machine_out = machine_model(test_x)
            user_out = user_model(user)
            pred = inte_model(machine_out, user_out)
            machine_predictions = np.vstack((machine_predictions, machine_out.cpu().detach().numpy()))
            user_predictions = np.vstack((user_predictions, user_out.cpu().detach().numpy()))

            # print(pred)
            # label_test = pred.max(dim=1)[1]
            predictions = np.vstack((predictions, pred.cpu().detach().numpy()))
            if i == forward_passes - 1:
                labels.extend(test_y.cpu().numpy())
        machine_dropout_predictions = np.vstack((machine_dropout_predictions,
                                                 machine_predictions[np.newaxis, :, :]))
        user_dropout_predictions = np.vstack((user_dropout_predictions,
                                              user_predictions[np.newaxis, :, :]))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout_predictions -> shape -> (forward_passes, n_samples, n_classes)
        # 计算均值
    machine_mean = np.mean(machine_dropout_predictions, axis=0)
    user_mean = np.mean(user_dropout_predictions, axis=0)

    mean = np.mean(dropout_predictions, axis=0)
    # mean -> shape -> (n_samples, n_classes)
    # print("mean:", mean)
    # 计算方差
    variance = np.var(dropout_predictions, axis=0)
    # variance -> shape -> (n_samples, n_classes)
    # print("variance:", variance)
    machine_mean = torch.from_numpy(machine_mean).cuda()
    user_mean = torch.from_numpy(user_mean).cuda()

    mean = torch.from_numpy(mean).cuda()
    labels = torch.from_numpy(np.array([labels]).T).cuda()
    machine_prediction = machine_mean.data.max(dim=1, keepdim=True)[1]
    # 存储pickle
    # file1 = open(r'dataset/delete_without_realRespond/机器的预测结果.pkl', 'wb')
    # pickle.dump(machine_prediction, file1)
    # file1.close()
    machine_acc = accuracy(machine_prediction, labels)
    print('machine_acc:', machine_acc)
    # machine_correction = machine_prediction.eq(labels.data.view_as(machine_prediction)).sum()
    # print('\tTest machine_Accuracy:{:.2f}%'.format(100. * machine_correction / len_of_all))
    user_prediction = user_mean.data.max(dim=1, keepdim=True)[1]
    # file2 = open(r'dataset/delete_without_realRespond/用户的预测结果.pkl', 'wb')
    # pickle.dump(user_prediction, file2)
    # file2.close()

    user_acc = accuracy(user_prediction, labels)
    print('user_acc:', user_acc)
    # user_correction = user_prediction.eq(labels.data.view_as(user_prediction)).sum()
    # print('\tTest user_Accuracy:{:.2f}%'.format(100. * user_correction / len_of_all))
    prediction = mean.data.max(dim=1, keepdim=True)[1]
    # file3 = open(r'dataset/delete_without_realRespond/人机结合的预测结果.pkl', 'wb')
    # pickle.dump(prediction, file3)
    # file3.close()
    file_result = pd.ExcelWriter("dataset/所有结果的对比（积分）.xlsx")
    result_duibi = torch.stack([torch.tensor(labels), machine_prediction, user_prediction, prediction], dim=1)
    # data = {"正确标签": torch.tensor(labels), "机器标签": machine_prediction, "用户标签": user_prediction,'人机结合标签':prediction}
    result_duibi = result_duibi.squeeze()
    result_duibi = result_duibi.numpy()
    result_duibi = pd.DataFrame(result_duibi)
    result_duibi.columns = ["正确标签", "机器标签", "用户标签", "人机结合标签"]
    result_duibi.to_excel(file_result)
    file_result.save()
    file_result.close()
    # np.savetxt("dataset/delete_without_realRespond/所有结果的对比lll.csv", result_duibi)
    # f = open('dataset/delete_without_realRespond/所有结果的对比lll.txt', 'w', encoding='utf-8')
    # for i in result_duibi:
    #     f.write(str(i.T).replace('tensor([[', '').replace(']])', '') + '\n')
    # file4 = open(r'dataset/delete_without_realRespond/正确的预测结果.pkl', 'wb')
    # pickle.dump(labels, file4)
    # file4.close()

    # correction = prediction.eq(labels.data.view_as(prediction)).sum()

    acc = accuracy(prediction, labels)
    p1_r1_f1_result = p1_r1_f1(prediction, labels)
    pre = p1_r1_f1_result[0]
    rec = p1_r1_f1_result[1]
    f = p1_r1_f1_result[2]
    auc_result = auc(mean, labels)
    roc_result = roc(mean, labels)
    torch.save(roc_result, 'roc.pkl')
    with open("CNN_news_user_ALL_MLP_roc.pkl", "wb") as fo:
        pickle.dump(roc_result, fo)
    print('acc:', acc, "pre:", pre, "rec:", rec, "f:", f, "auc:", auc_result)

    # print('\tTest Accuracy:{:.2f}%'.format(100. * correction / len_of_all))
    print(result_duibi)
    return mean, variance


# 计算预测准确性
# def accuracy(y_pred, y_true):
#      label_pred = y_pred.max(dim=1)[1]
#      acc = len(y_pred) - torch.sum(torch.abs(label_pred-y_true)) # 正确的个数
#      return acc.detach().cpu().numpy() / len(y_pred)

def accuracy(y_pred, y_true):
    # label_pred = y_pred.max(dim=1)[1]
    acc = accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())
    return acc


def p1_r1_f1(y_pred, y_true):
    # label_pred = y_pred.max(dim=1)[1]
    p1, r1, f1, _ = precision_recall_fscore_support(y_true.detach().numpy(), y_pred.detach().numpy(), pos_label=1,
                                                    average='macro')
    p1_r1_f1 = [p1, r1, f1]
    return p1_r1_f1


def auc(y_pred, y_true):
    auc = roc_auc_score(y_true.detach().numpy(), y_pred[:, 1].detach().numpy())
    # auc = roc_auc_score(y_true, y_pred)
    return auc


def roc(y_pred, y_true):
    roc = roc_curve(y_true.detach().numpy(), y_pred[:, 1].detach().numpy())
    return roc
'''


def main():
    warnings.filterwarnings("ignore")
    #print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  # 现在时间
    #filename = ''  # 当前路径
    #t = os.path.getmtime(filename)
    #print(datetime.datetime.fromtimestamp(t))
    set_seed(23711)

    data_path_title = "../../../../correct_score_(true_rate_to_num_for_diff)/changeLabel_fakeNews_is_1(20230328)/"

    #train_path = open('../bert_dataset/mini_val_data_with_bert_machine_distribution.pkl', 'rb')
    #train_file = pickle.load(train_path)

    test_path = open('../bert_dataset/mini_test_data_with_bert_machine_distribution.pkl', 'rb')
    test_file = pickle.load(test_path)

    # model_save_path = 'dataset/cnn/20220717fenci/20220902_glove/cnn_with_dropout/seed666_test/bin'

    #result_save_path = r'../results/'
    encoder_result_save_path = r'../results/bert_news_user_weight_beta/gausi_beta_pre_loss/round2/'
    # encoder_result_save_path = r'../results/cnn_news_user_weight_beta/7_2_1/gausi_beta_pre_loss/round7_230711/'
    # test_result_save_path = r'dataset/cnn/20220717fenci/20220902_glove/cnn_inte/all_data/'

    #f2 = open('../../../correct_score_(true_rate_to_num_for_diff)/glove_50d_vocab.pkl', 'rb')
    #vocab = pickle.load(f2)  # 加载本地已经存储的vocab

    # 构建MyDataset实例
    #train_data = MyDataset(text_file=train_file)
    test_data = MyDataset(text_file=test_file)
    # ver_data = MyDataset(text_file=test_file)

    train_batch_size_value = 64
    test_batch_size_value = 128
    # 构建DataLoder
    #train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size_value, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size_value, shuffle=False, drop_last=True)
    # ver_loader = DataLoader(dataset=ver_data, batch_size=test_batch_size_value, shuffle=False, drop_last=False)

    # 生成模型
    # machine_model = WordCNNForClassification.from_pretrained(model_save_path)
    # machine_model.load_state_dict(torch.load(model_save_path + 'CNN_news_seed_datasetscore_model_parameter.pkl',
    #                map_location=torch.device('cpu')))
    df = pd.read_excel(data_path_title + 'user_score_index(true_rate_to_num_for_diff).xlsx', usecols=[1])  # 指定读取第2列的积分
    temp = torch.tensor(data=df.values)
    user_score = temp.squeeze()  # 减少一个tensor维度
    user_model = User(user_score)

    # encoder_model = Encoder(input_size=4, hidden_size=8, output_size=2)
    encoder_model = Encoder(input_size=4, hidden_size=8, output_size=2)
    forward_passes = 10

    # user_model.load_state_dict(
    #     torch.load('../../user_score230327/user_weight_model_parameter20230404.pkl',
    #                map_location=torch.device('cpu')))
    #train(encoder_model=encoder_model, user_model=user_model, train_data=train_loader,
    #       ver_data=test_loader, forward_passes=forward_passes, epoch=30, batch_size=train_batch_size_value,
    #       test_batch_size_value=test_batch_size_value, result_save_path=encoder_result_save_path)

    # 加载训练好的模型

    user_model.load_state_dict(
        torch.load('../../../user_score230327/user_weight_model_parameter20230404.pkl',
                   map_location=torch.device('cpu')))
    # user_model.load_state_dict(
    #     torch.load('/home/zhoulujuan/encoder230408/encoder/user_score230327/0515/round1/user_weight_model_parameter(with_update).pkl',
    #                map_location=torch.device('cpu')))
    encoder_model.load_state_dict(
        torch.load(encoder_result_save_path + 'bert_dropout_score_weigt_encoder_model_parameter_bata.pkl',
                   map_location=torch.device('cpu')))
    print("开始测试————————————————————————————————")
    # 测试结果
    # ceshi(inte_model=inte_model, machine_model=machine_model, user_model=user_model,
    #       test_data=test_loader, vocab=vocab, forward_passes=forward_passes, batch_size=test_batch_size_value,
    #       result_save_path=test_result_save_path)

    test(user_model=user_model, encoder_model=encoder_model, test_data=test_loader,
         batch_size=test_batch_size_value, forward_passes=forward_passes,result_save_path=encoder_result_save_path)
    # print('acc:', evaluations[0], 'pre:', evaluations[1], 'rec:', evaluations[2], 'f:', evaluations[3], 'auc:',
    #       evaluations[4])
    # print('acc:', evaluations[0], 'pre:', evaluations[1], 'rec:', evaluations[2], 'f:', evaluations[3])


if __name__ == '__main__':
    main()
