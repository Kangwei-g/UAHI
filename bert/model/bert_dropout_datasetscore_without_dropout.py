import datetime
import pickle
import random
#import sys
import time
#import textattack

import pandas as pd
#from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForSequenceClassification
#import numpy as np
#import torch
#import torch.nn as nn
#from pytorch_pretrained_bert import BertModel, BertTokenizer
#from pytorch_pretrained_bert.modeling import BertModel as BertModel_Embedding
#from torch.utils.data import DataLoader

#from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForSequenceClassification
#import numpy as np
#import torch
#import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
#from pytorch_pretrained_bert.modeling import BertModel as BertModel_Embedding

#import torch  # torch==1.7.1
#import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
#import re
#import numpy as np
from tqdm import tqdm

#from sklearn import metrics
# from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import numpy as np
#from torch.autograd import Variable
#import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, roc_curve
#import json
#from textattack.model_args import TEXTATTACK_MODELS
from textattack.models.helpers.utils import load_cached_state_dict
#from textattack.shared import utils

#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# # 加载bert的分词器
# tokenizer = BertTokenizer.from_pretrained('vocab_chinese.txt')

# MAX_WORD = 10000  # 只保留最高频的5000词
MAX_LEN = 30  # 句子统一长度为200
# word_count = {}  # 词-词出现的词数 词典
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
    # ^^ safe to call this function even if cuda is not available



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
    :param model:
    :return:
    """
    for m in model.modules():
        if m.__class__.__name__.find('BatchNorm') != -1:
            m.eval()


#  数据预处理过程
def text_transform(sentence_list):
    # print('text_transform_input_text:', sentence_list)
    # print("text_transform")

    sentence_index_list = []
    for sentence in sentence_list:
        tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/vocab_chinese.txt')
        tokens = tokenizer.tokenize(sentence)
        # tokens = tokenizer(sentence)  # 分词统计词数
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).cuda()
        sentence_idx = tokenizer.convert_tokens_to_ids(tokens)
        if len(sentence_idx) < MAX_LEN:
            for i in range(MAX_LEN - len(sentence_idx)):  # 对长度不够的句子进行PAD填充
                # sentence_idx.append(int(1))
                sentence_idx.append(0)
        sentence_idx = sentence_idx[:MAX_LEN]  # 取前MAX_LEN长度
        sentence_index_list.append(sentence_idx)
    # print('sentence_index_list:', sentence_index_list)
    return torch.LongTensor(sentence_index_list)  # 将转为idx的词转为tensor


# 加载 train_path = r'data_clean_20211020/train_clean'  txt文件的版本
# class MyDataset(Dataset):
#     def __init__(self, text_path):
#         file = open(text_path, 'r', encoding='UTF-8-sig')
#         self.text_with_tag = file.readlines()  # 文本标签与内容
#         file.close()
#
#     def __getitem__(self, index):  # 重写getitem
#         line = self.text_with_tag[index]  # 获取一个样本的标签和文本信息
#         label = int(line[0].encode('utf-8'))  # 标签信息
#         # label = int(str(line[0].encode('utf-8')))
#         # label = int(line[0])
#         text = line[2:-1]  # 文本信息
#         return text, label
#
#     def __len__(self):
#         return len(self.text_with_tag)


# 加载含有news user 的pkl文件版本，但是只保留News进行文本分类
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
            # user = line[1]
            each_list.append(label)
            each_list.append(text)
            # each_list.append(user)
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
        # user = line[2]  # 文本信息
        # user = torch.tensor(user)
        # print('user', user)
        # print('dataset_user:', user)
        # print('dataset_user_type:', type(user))

        return text, label

    def __len__(self):
        return len(self.text_with_tag)


class Bert(nn.Module):
    def __init__(self, bert_path, num_hiddens,dropout):
        super(Bert, self).__init__()
        # self._config = {
        #     "architectures": "Bert",
        #     "bert_path": bert_path,
        #     "num_hiddens": num_hiddens,
        #     "dropout": dropout,
        #     # #下面的这些参数配置都是从源码自带的里面抄来的,报错了，已删
        # }
        # bert_path = 'bert-base-chinese/'
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.decoder = nn.Linear(num_hiddens, 2)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, seqs):
        encoded_layers, pooled_output = self.bert(seqs, output_all_encoded_layers=False)
        # encoding = pooled_output  # [batch_size, num_dim]
        # pooled_output = self.drop(pooled_output)
        encoding = self.decoder(pooled_output)
        outs = self.softmax(encoding)
        return outs

    def save_pretrained(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))
        # with open(os.path.join(output_path, "config.json"), "w") as f:
        #     json.dump(self._config, f)

    @classmethod
    def from_pretrained(cls, name_or_path,model):
        """Load trained LSTM model by name or from path.

        Args:
            name_or_path (:obj:`str`): Name of the model (e.g. "lstm-imdb") or model saved via :meth:`save_pretrained`.
        Returns:
            :class:`~textattack.models.helpers.LSTMForClassification` model
        """
        # if name_or_path in TEXTATTACK_MODELS:
        #     # path = utils.download_if_needed(TEXTATTACK_MODELS[name_or_path])
        #     path = utils.download_from_s3(TEXTATTACK_MODELS[name_or_path])
        # else:
        #     path = name_or_path

        # config_path = os.path.join(path, "config.json")
        #
        # if os.path.exists(config_path):
        #     with open(config_path, "r") as f:
        #         config = json.load(f)
        # else:
        #     # Default config
        #     config = {
        #         "architectures": "Bert",
        #         "bert_path": 'bert-base-chinese/',
        #         "num_hiddens": 768,
        #         "dropout": 0,
        #         # 下面的这些参数配置都是从源码自带的里面抄来的,报错了，已删
        #     }
        # del config["architectures"]
        # model = cls(**config)
        state_dict = load_cached_state_dict(name_or_path)
        model.load_state_dict(state_dict)
        return model


# 模型训练
def train(model, train_data, val_data, val_size, epoch, forward_passes, model_save_path,result_save_path):
    print('train model')
    best_loss, best_acc = 10, 0.5

    model = model.cuda()
    loss_sigma = 0.0
    correct = 0.0
    ver_acc_list = []
    # 定义损失函数和优化器
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=3e-5, lr_decay=0, weight_decay=0,
                                    initial_accumulator_value=0)

    for epoch in tqdm(range(epoch)):
        print('//epoch//:',epoch)
        model.train()
        avg_loss = 0  # 平均损失
        avg_acc = 0  # 平均准确率
        for idx, (text, label) in enumerate(tqdm(train_data)):
            train_x = text_transform(text).cuda()
            # train_x = text_transform(text)
            train_y = label.cuda()
            optimizer.zero_grad()
            pred = model(train_x)
            # print("train_pred:", pred)
            label_test = pred.max(dim=1)[1]
            # print('label_test:', label_test)
            loss = criterion(pred.log(), train_y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += accuracy(pred, train_y)
        # 一个epoch结束后，计算平均loss和评平均acc
        avg_loss = avg_loss / len(train_data)
        avg_acc = avg_acc / len(train_data)

        print("train_loss:", avg_loss, " train_acc:", avg_acc)
        # print("avg_loss:", avg_loss)
        # 保存训练完成后的模型参数

        # torch.save(model.state_dict(), 'BERT_LSTM_news_parameter_seed_datasetscore.pkl')

        # 开始验证
        print('开始验证####################')
        evaluations = test(model, val_data, val_size, forward_passes,result_save_path)
        val_acc = evaluations[0]
        val_loss = evaluations[5]
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            # 保存训练完成后的模型参数
            best_acc = val_acc
            best_loss = val_loss
            # print(f'Save model! Best validation accuracy is {val_acc:.5f}
            print('保存最好的模型%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'Save model! Best validation accuracy is {val_acc}')
            model.save_pretrained(model_save_path)
        #
        # ver_avg_acc = 0
        # for idx, (text, label) in enumerate(tqdm(ver_data)):
        #     train_x = text_transform(text).cuda()
        #     train_y = label.cuda()
        #
        #     pred = model(train_x)
        #     # print(pred)
        #     label_test = pred.max(dim=1)[1]
        #     # print(label_test)
        #
        #     ver_avg_acc += accuracy(pred, train_y)
        # # 一个epoch结束后，计算平均loss和评平均acc
        # ver_avg_acc = ver_avg_acc / len(ver_data)
        # print("ver_avg_acc:,", ver_avg_acc)
        ver_acc_list.append(val_acc)
    data = np.array(ver_acc_list).T
    data_pd = pd.DataFrame(data, columns=['ver_model_acc'])
    data_pd.to_csv(result_save_path+'val_bert_dropout_news_seed_datasetscore_acc.csv')


# 模型测试
def test(model, test_data, test_size, forward_passes):
    labels = list()
    dropout_predictions = np.empty((0, test_size, 2))
    criterion = torch.nn.NLLLoss()
    all_batch_test_loss = 0
    for i in range(forward_passes):
        predictions = np.empty((0, 2))
        model = model.cuda()
        model.eval()
        enable_dropout(model)
        fix_batch_normalization(model)

        for idx, (text, label) in enumerate(tqdm(test_data)):
            train_x = text_transform(text).cuda()
            train_y = label.cuda()
            pred = model(train_x)
            # print(pred)
            each_batch_test_loss = criterion(pred.log(), train_y)
            all_batch_test_loss += each_batch_test_loss.item()

            predictions = np.vstack((predictions, pred.cpu().detach().numpy()))
            if i == forward_passes - 1:
                labels.extend(train_y.cpu().numpy())

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))

        # dropout_predictions -> shape -> (forward_passes, n_samples, n_classes)
        # 计算均值
    avg_test_loss = all_batch_test_loss/(forward_passes*len(test_data))
    mean = np.mean(dropout_predictions, axis=0)
    # mean -> shape -> (n_samples, n_classes)
    # print("mean:", mean)
    # 计算方差
    std = np.std(dropout_predictions, axis=0)
    # variance -> shape -> (n_samples, n_classes)
    # print("variance:", variance)
    file = open(r'../dataset/mini_test_bert_mean230716.pkl', 'wb')
    pickle.dump(mean, file)
    file.close()

    file2 = open(r'../dataset/mini_test_bert_std230716.pkl', 'wb')
    pickle.dump(std, file2)
    file2.close()

    mean = torch.from_numpy(mean).cuda()
    labels = torch.from_numpy(np.array([labels]).T).cuda()
    # prediction = mean.data.max(dim=1, keepdim=True)[1]
    prediction = mean.max(dim=1)[1]
    # correction = prediction.eq(labels.data.view_as(prediction)).sum()

    acc = accuracy(mean, labels)
    p1_r1_f1_result = p1_r1_f1(prediction, labels)
    pre = p1_r1_f1_result[0]
    rec = p1_r1_f1_result[1]
    f = p1_r1_f1_result[2]
    auc_result = auc(mean, labels)
    # roc_result = roc(mean, labels)
    # test_loss = criterion(mean.log(), labels)
    evaluations = [acc, pre, rec, f, auc_result, avg_test_loss]
    # torch.save(roc_result, 'roc.pkl')
    # with open(result_save_path+'roc_bert_news_seed_datasetscore.pkl', "wb") as fo:
    #     pickle.dump(roc_result, fo)
    print('test_acc:', acc, "test_pre:", pre, "test_rec:", rec, "test_f:", f, "test_auc:", auc_result, 'test_loss:',
          avg_test_loss)

    # print('\tTest Accuracy:{:.2f}%'.format(100. * correction / len(test_data.dataset)))

    return evaluations
# 模型测试
# 没打开dropout
def test_without_dropout(model, test_data):
    print('test model')
    model = model.cuda()
    model.eval()
    acc = 0
    pre = 0
    rec = 0
    f = 0
    auc_result = 0

    for idx, (text, label) in enumerate(tqdm(test_data)):
        train_x = text_transform(text).cuda()
        train_y = label.cuda()
        pred = model(train_x)
        prediction = pred.max(dim=1)[1]
        acc += accuracy(pred, train_y)
        p1_r1_f1_result = p1_r1_f1(prediction, train_y)
        pre += p1_r1_f1_result[0]
        rec += p1_r1_f1_result[1]
        f += p1_r1_f1_result[2]
        auc_result += auc(pred, train_y)
    avg_acc = acc / len(test_data)
    avg_pre = pre / len(test_data)
    avg_rec = rec / len(test_data)
    avg_f = f / len(test_data)
    avg_auc = auc_result / len(test_data)
    print("acc:", avg_acc, "pre:", avg_pre, "rec:", avg_rec, "f:", avg_f, "auc:", avg_auc)


# 计算预测准确性
# def accuracy(y_pred, y_true):
#      label_pred = y_pred.max(dim=1)[1]
#      acc = len(y_pred) - torch.sum(torch.abs(label_pred-y_true)) # 正确的个数
#      return acc.detach().cpu().numpy() / len(y_pred)

def accuracy(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    label_pred = y_pred.max(dim=1)[1]
    acc = accuracy_score(y_true.detach().numpy(), label_pred.detach().numpy())
    return acc


def p1_r1_f1(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    # label_pred = y_pred.max(dim=1)[1]
    p1, r1, f1, _ = precision_recall_fscore_support(y_true.detach().numpy(), y_pred.detach().numpy(), average='macro')
    p1_r1_f1 = [p1, r1, f1]
    return p1_r1_f1


def auc(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    auc = roc_auc_score(y_true.detach().numpy(), y_pred[:, 1].detach().numpy())
    # auc = roc_auc_score(y_true, y_pred)
    return auc


def roc(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    roc = roc_curve(y_true.detach().numpy(), y_pred[:, 1].detach().numpy())
    return roc


def main():
    #print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  # 现在时间
    #filename = 'bert_dropout_datasetscore_without_dropout.py'  # 当前路径
    #t = os.path.getmtime(filename)
    #print(datetime.datetime.fromtimestamp(t))
    set_seed(2021110744)
    data_path_title = "../../../correct_score_(true_rate_to_num_for_diff)/changeLabel_fakeNews_is_1(20230328)/"

    # train_path = r'data_clean_20211020/train_clean'  # 预处理后的训练集文件地址
    # test_path = r'data_clean_20211020/test_clean'  # 预处理后的训练集文件地址
    # train_path = open(data_path_title + 'train_data(true_rate_to_num_for_diff).pkl', 'rb')
    # train_file = pickle.load(train_path)

    test_path = open(data_path_title + 'mini_test_data(true_rate_to_num_for_diff).pkl', 'rb')
    test_file = pickle.load(test_path)

    model_save_path = r'../results/bert_dropout/round1_230716/bin'
    result_save_path = r'../results/bert_dropout/round1_230716/'

    bert_path = 'bert-base-chinese/'
    # bert_path = 'dataset/bert/20220717fenci/bert_dropout/model/'

    # 构建MyDataset实例
    # train_data = MyDataset(text_file=train_file)
    test_data = MyDataset(text_file=test_file)
    test_size = len(test_data)
    # 构建DataLoder
    # train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False)

    # 生成模型
    model = Bert(bert_path=bert_path,num_hiddens=768,dropout=0.1)  # 定义模型
    forward_passes = 10
    # train(model=model, train_data=train_loader, val_data=test_loader, val_size=test_size, epoch=30,
    #       forward_passes=forward_passes,
    #       model_save_path=model_save_path,result_save_path=result_save_path)

    # 加载训练好的模型
    # model.load_state_dict(torch.load('BERT_LSTM_news_parameter_seed_datasetscore.pkl', map_location=torch.device('cuda:0')))
    # model = model.from_pretrained(model_save_path,model).cuda()
    model.load_state_dict(load_cached_state_dict(model_save_path))


    # 测试结果
    print('开始测试@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #test(model=model, test_data=test_loader, test_size=test_size, forward_passes=forward_passes)
    test_without_dropout(model=model,test_data=test_loader)
    # print("mean:", mean, "variance", variance)


if __name__ == '__main__':
    main()


