import pickle

import torch  # torch==1.7.1
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

import numpy as np
from tqdm import tqdm

# from lstm import tokenizer
import random
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, roc_curve

import json
from torch.nn import functional as F

# import textattack
from textattack.model_args import TEXTATTACK_MODELS
# from textattack.models.helpers import GloveEmbeddingLayer
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.shared import utils
import datetime
import time
import matplotlib.pyplot as plt

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# seed = 1
# torch.manual_seed(seed)            # 为CPU设置随机种子
# torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
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


MAX_WORD = 10000  # 只保留最高频的10000词   文档总共有17387个词  前3000多个词之后开始每个词只出现2词
MAX_LEN = 30  # 句子统一长度为30   文档的平均长度是27
word_count = {}  # 词-词出现的词数 词典
'''
#清理文本，去标点符号，转小写
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()

# 分词方法
def tokenizer(sentence):
    return sentence.split()

#  数据预处理过程
def data_process(text_path, text_dir): # 根据文本路径生成文本的标签

    print("data preprocess")
    file_pro = open(text_path,'w',encoding='UTF-8-sig')
    for root, s_dirs, _ in os.walk(text_dir): # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取train和test文件夹下所有的路径
            text_list = os.listdir(i_dir)
            tag = i_dir.split("\\")[-1] # 获取标签
            if tag == 'pos':
                label = '1'
            if tag == 'neg':
                label = '0'
            if tag =='unsup':
                continue
            for i in range(len(text_list)):
                if not text_list[i].endswith('txt'): # 判断若不是txt,则跳过
                    continue
                f = open(os.path.join(i_dir, text_list[i]),'r',encoding='utf-8') # 打开文本
                raw_line = f.readline()
                pro_line = clean_str(raw_line)
                tokens = tokenizer(pro_line) # 分词统计词数
                for token in tokens:
                    if token in word_count.keys():
                        word_count[token] = word_count[token] + 1
                    else:
                        word_count[token] = 0
                file_pro.write(label + ' ' + pro_line +'\n')
                f.close()
                file_pro.flush()
    file_pro.close()

    print("build vocabulary")

    vocab = {"<UNK>": 0, "<PAD>": 1}

    word_count_sort = sorted(word_count.items(), key=lambda item : item[1], reverse=True) # 对词进行排序，过滤低频词，只取前MAX_WORD个高频词
    word_number = 1
    for word in word_count_sort:
        if word[0] not in vocab.keys():
            vocab[word[0]] = len(vocab)
            word_number += 1
        if word_number > MAX_WORD:
            break
    return vocab
'''


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


# 分词方法
def tokenizer(sentence):
    return sentence.split()


def data_process(text_file):
    print("data preprocess")
    # f = open(text_path, 'r', encoding='UTF-8-sig')  # 打开文本
    # raw_lines = text_file.readlines()
    for i in range(len(text_file)):
        raw_line = text_file[i]
        tokens = tokenizer(raw_line[0][2])  # 分词统计词数
        for token in tokens:
            if token in word_count.keys():
                word_count[token] = word_count[token] + 1
            else:
                word_count[token] = 0
    print("build vocabulary")

    vocab = {"<UNK>": 0, "<PAD>": 1}

    word_count_sort = sorted(word_count.items(), key=lambda item: item[1], reverse=True)  # 对词进行排序，过滤低频词，只取前MAX_WORD个高频词
    word_number = 1
    for word in word_count_sort:
        if word[0] not in vocab.keys():
            vocab[word[0]] = len(vocab)
            word_number += 1
        if word_number > MAX_WORD:
            break
    return vocab


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


# 根据vocab将句子转为定长MAX_LEN的tensor
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


# 定义LSTM模型
class GRU(nn.Module):
    def __init__(self, embed_size=50, num_hiddens=300, num_layers=2,dropout=0.1):
        super(GRU, self).__init__()
        self._config = {
            "architectures": "GRU",
            "embed_size": embed_size,
            "num_hiddens": num_hiddens,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        f2 = open('../../../correct_score_(true_rate_to_num_for_diff)/glove_50d_vocab.pkl', 'rb')
        vocab = pickle.load(f2)
        self.emb_layer = nn.Embedding(len(vocab), embed_size)  # embedding层
        embeding_vector = np.load('../../../correct_score_(true_rate_to_num_for_diff)/glove_50d_wordlist.npy', allow_pickle=True)
        self.emb_layer.weight.data.copy_(torch.from_numpy(embeding_vector))
        self.emb_layer.weight.requires_grad = True
        self.encoder = nn.GRU(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False,
                               batch_first=False,
                               dropout=dropout,
                               )
        self.decoder = nn.Linear(num_hiddens, 2)
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs的形状是（批量大小，词数），因此LSTM需要将序列长度（Seq_len）作为第一维，所以将输入转置后 再提取词特征
        embeddings = self.emb_layer(inputs.permute(1, 0))  # permute(1,0)交换维度
        # LSTM只传入输入embeddings,因此只返回最后一层的隐藏层再各时间步的隐藏状态
        # outputs的形状是（词数，批量大小， 隐藏单元个数）
        embeddings = self.dropout(embeddings)
        outputs, _ = self.encoder(embeddings)
        # 连接初时间步和最终时间步的隐藏状态作为全连接层的输入。形状为(批量大小， 隐藏单元个数)
        encoding = outputs[-1]  # 取LSTM最后一层结果
        # encoding = outputs[:, -1,:]
        deep1 = self.decoder(encoding)
        deep2 = self.dropout(deep1)
        outs = self.softmax(deep2)  # 输出层为二维概率[a,b]
        return outs

    def save_pretrained(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(
            state_dict,
            os.path.join(output_path, "model.bin"),
        )
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self._config, f)

    @classmethod
    def from_pretrained(cls, name_or_path):
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
        path = name_or_path
        config_path = os.path.join(path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                "architectures": "GRU",
                "embed_size": 300,
                "num_hiddens": 300,
                "num_layers": 2,
                "dropout": 0.1,
            }
        del config["architectures"]
        model = cls(**config)
        state_dict = load_cached_state_dict(path)
        model.load_state_dict(state_dict)
        return model



# 模型训练
def train(model, train_data, val_data, epoch, result_save_path,model_save_path, vocab, val_size, forward_passes):
    print('train model')
    model = model.cuda()
    loss_sigma = 0.0
    correct = 0.0
    best_loss, best_acc = 10, 0.5

    # 定义损失函数和优化器
    criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=2e-3, lr_decay=0.0001, weight_decay=0,
                                    initial_accumulator_value=0)
    val_avg_acc_list = []
    epoch_list = []
    for epoch in tqdm(range(epoch)):
        model.train()
        avg_loss = 0  # 平均损失
        avg_acc = 0  # 平均准确率
        for idx, (text, label) in enumerate(tqdm(train_data)):
            train_x = text_transform(text, vocab).cuda()
            train_y = label.cuda()

            optimizer.zero_grad()
            pred = model(train_x)
            # sm = nn.Softmax(dim=1)
            # pred = sm(pred)
            # print(pred)
            label_test = pred.max(dim=1)[1]
            # print(label_test)
            # loss = criterion(pred, train_y)

            loss = criterion(pred.log(), train_y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            avg_acc += accuracy(pred, train_y)
        # 一个epoch结束后，计算平均loss和评平均acc
        avg_loss = avg_loss / len(train_data)
        avg_acc = avg_acc / len(train_data)

        epoch_list.append(epoch)
        print("avg_loss:", avg_loss, " train_avg_acc:,", avg_acc)

        print('开始验证####################')
        evaluations = test(model=model, test_data=val_data, vocab=vocab, test_size=val_size, forward_passes=forward_passes,
             result_save_path=result_save_path)
        val_acc = evaluations[0]
        val_loss = evaluations[5]

        val_avg_acc_list.append(val_acc)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            # 保存训练完成后的模型参数
            best_acc = val_acc
            best_loss = val_loss
            # print(f'Save model! Best validation accuracy is {val_acc:.5f}
            print('保存最好的模型%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'Save model! Best validation accuracy is {val_acc}')
            model.save_pretrained(model_save_path)
            # 保存训练完成后的模型参数
            torch.save(model.state_dict(), result_save_path + 'GRU_news_user_parameter_seed_datasetscore.pkl')

    plt.plot(epoch_list, val_avg_acc_list)

    # 添加标题和轴标签。
    plt.title("Verification acc over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Verification acc")

    # 显示图形。
    plt.show()

# 模型测试
# 打开dropout
def test(model, test_data, vocab, test_size, forward_passes, result_save_path):
    labels = list()
    dropout_predictions = np.empty((0, test_size, 2))
    criterion = torch.nn.NLLLoss()
    all_batch_test_loss = 0
    for i in range(forward_passes):
        predictions = np.empty((0, 2))
        model = model.cuda()
        model.eval()
        # enable_dropout(model)
        # fix_batch_normalization(model)

        for idx, (text, label) in enumerate(tqdm(test_data)):
            train_x = text_transform(text, vocab).cuda()
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
    avg_test_loss = all_batch_test_loss / (forward_passes * len(test_data))
    mean = np.mean(dropout_predictions, axis=0)
    # mean -> shape -> (n_samples, n_classes)
    # print("mean:", mean)
    # 计算方差
    std = np.std(dropout_predictions, axis=0)
    # variance -> shape -> (n_samples, n_classes)
    # print("variance:", variance)
    # file = open(r'../dataset/mini_val_gru_glove_mean230714.pkl', 'wb')
    # pickle.dump(mean, file)
    # file.close()
    #
    # file2 = open(r'../dataset/mini_val_gru_glove_std230714.pkl', 'wb')
    # pickle.dump(std, file2)
    # file2.close()

    mean = torch.from_numpy(mean).cuda()
    labels = torch.from_numpy(np.array([labels]).T).cuda()
    prediction = mean.data.max(dim=1, keepdim=True)[1]
    # correction = prediction.eq(labels.data.view_as(prediction)).sum()

    acc = accuracy(mean, labels)
    p1_r1_f1_result = p1_r1_f1(prediction, labels)
    pre = p1_r1_f1_result[0]
    rec = p1_r1_f1_result[1]
    f = p1_r1_f1_result[2]
    auc_result = auc(mean, labels)
    roc_result = roc(mean, labels)
    # test_loss = criterion(mean.log(), labels)
    evaluations = [acc, pre, rec, f, auc_result, avg_test_loss]
    # torch.save(roc_result, 'roc.pkl')
    # with open(result_save_path + 'roc_lstm_news_seed_datasetscore.pkl', "wb") as fo:
    #     pickle.dump(roc_result, fo)
    print('test_acc:', acc, "test_pre:", pre, "test_rec:", rec, "test_f:", f, "test_auc:", auc_result, 'test_loss:',
          avg_test_loss)

    # print('\tTest Accuracy:{:.2f}%'.format(100. * correction / len(test_data.dataset)))

    return evaluations


# def test(model, test_data, vocab, forward_passes):
#     print('test model')
#     labels = list()
#     dropout_predictions = np.empty((0, 969, 2))
#     dropout_predictions = np.empty((0, test_size, 2))
#
#     for i in range(forward_passes):
#         predictions = np.empty((0, 2))
#         model = model.cuda()
#         model.eval()
#         enable_dropout(model)
#         fix_batch_normalization(model)
#
#         for idx, (text, label) in enumerate(tqdm(test_data)):
#             train_x = text_transform(text, vocab).cuda()
#             train_y = label.cuda()
#             pred = model(train_x)
#             # print(pred)
#             predictions = np.vstack((predictions, pred.cpu().detach().numpy()))
#             if i == forward_passes - 1:
#                 labels.extend(train_y.cpu().numpy())
#
#         dropout_predictions = np.vstack((dropout_predictions,
#                                          predictions[np.newaxis, :, :]))
#         # dropout_predictions -> shape -> (forward_passes, n_samples, n_classes)
#         # 计算均值
#     mean = np.mean(dropout_predictions, axis=0)
#     # mean -> shape -> (n_samples, n_classes)
#     # print("mean:", mean)
#     # 计算方差
#     variance = np.var(dropout_predictions, axis=0)
#     # variance -> shape -> (n_samples, n_classes)
#     # print("variance:", variance)
#
#     mean = torch.from_numpy(mean).cuda()
#     labels = torch.from_numpy(np.array([labels]).T).cuda()
#     prediction = mean.data.max(dim=1, keepdim=True)[1]
#     # correction = prediction.eq(labels.data.view_as(prediction)).sum()
#
#     acc = accuracy(mean, labels)
#     p1_r1_f1_result = p1_r1_f1(prediction, labels)
#     pre = p1_r1_f1_result[0]
#     rec = p1_r1_f1_result[1]
#     f = p1_r1_f1_result[2]
#     auc_result = auc(mean, labels)
#     # roc_result = roc(mean, labels)
#     # torch.save(roc_result, 'roc_15.pkl')
#     # with open("bert_roc_15.pkl", "wb") as fo:
#     #     pickle.dump(roc_result, fo)
#     print("acc:", acc, "pre:", pre, "rec:", rec, "f:", f, "auc:", auc_result)
#
#     # print('\tTest Accuracy:{:.2f}%'.format(100. * correction / len(test_data.dataset)))
#
#     return mean, variance


'''


# 模型测试
# 没打开dropout
def test(model, test_data, vocab):
    print('test model')
    model = model.cuda()
    model.eval()
    avg_acc = 0
    avg_pre = 0
    avg_rec = 0
    avg_f = 0
    avg_auc = 0
    avg_loss = 0
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    for idx, (text, label) in enumerate(tqdm(test_data)):
        train_x = text_transform(text, vocab).cuda()
        train_y = label.cuda()
        pred = model(train_x)
        # print(pred)
        label_test = pred.max(dim=1)[1]
        # print(label_test)
        # sm = nn.Softmax(dim=1)
        # pred = sm(pred)
        loss = criterion(pred, train_y)

        # loss = criterion(pred.log(), train_y)
        loss.backward()
        avg_loss += loss.item()
        # sm = nn.Softmax(dim=1)
        # pred = sm(pred)
        avg_acc += accuracy(pred, train_y)
        avg_pre += precision(pred, train_y)
        avg_rec += recall(pred, train_y)
        avg_f += f1(pred, train_y)
        avg_auc += auc(pred, train_y)
        # print(avg_acc, avg_pre, avg_rec, avg_f, avg_auc)
    avg_acc = avg_acc / len(test_data)
    avg_pre = avg_pre / len(test_data)
    avg_rec = avg_rec / len(test_data)
    avg_f = avg_f / len(test_data)
    avg_auc = avg_auc / len(test_data)
    avg_loss = avg_loss / len(test_data)
    print("acc:", avg_acc, "pre:", avg_pre, "rec:", avg_rec, "f:", avg_f, "auc:", avg_auc, "test_loss:", avg_loss)
'''


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
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  # 现在时间
    filename = 'GRU_glove_with_distribution.py'  # 当前路径
    t = os.path.getmtime(filename)
    print(datetime.datetime.fromtimestamp(t))
    set_seed(2021110744)
    data_path_title = "../../../correct_score_(true_rate_to_num_for_diff)/changeLabel_fakeNews_is_1(20230328)/"

    # train_path = open(data_path_title + 'train_data(true_rate_to_num_for_diff).pkl', 'rb')
    # train_file = pickle.load(train_path)

    test_path = open(data_path_title + 'mini_test_data(true_rate_to_num_for_diff).pkl', 'rb')
    test_file = pickle.load(test_path)

    f2 = open('../../../correct_score_(true_rate_to_num_for_diff)/glove_50d_vocab.pkl', 'rb')
    vocab = pickle.load(f2)  # 加载本地已经存储的vocab

    # 构建MyDataset实例
    # train_data = MyDataset(text_file=train_file)
    test_data = MyDataset(text_file=test_file)
    test_size = len(test_data)
    # 构建DataLoder
    forward_passes = 1

    # train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False)

    # 生成模型
    model = GRU()  # 定义模型

    # result_save_path = 'dataset/cnn/20220717fenci/glove_50d_word_vec/'
    result_save_path = '../results/gru_glove/round2_230714/'
    model_save_path = '../results/gru_glove/round2_230714/bin'


    # train(model=model, train_data=train_loader, val_data=test_loader, epoch=22, result_save_path=result_save_path,model_save_path=model_save_path,
    #       vocab=vocab, val_size=test_size, forward_passes=forward_passes)
    # 加载训练好的模型
    # model = model.from_pretrained(model_save_path).cuda()
    model.load_state_dict(
        torch.load(result_save_path + 'GRU_news_user_parameter_seed_datasetscore.pkl', map_location=torch.device('cpu')))

    # 测试结果
    print('开始测试——————————————')
    # test(model=model, test_data=test_loader, vocab=vocab, forward_passes=10)
    test(model=model, test_data=test_loader, vocab=vocab,test_size=test_size, forward_passes=forward_passes,result_save_path=result_save_path)
    # print('acc:',evaluations[0],'pre:',evaluations[1],'rec:',evaluations[2],'f:',evaluations[3], 'auc:', evaluations[4])


if __name__ == '__main__':
    main()
