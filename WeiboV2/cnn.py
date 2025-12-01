# 文件名：text_classifier_train.py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score
from collections import Counter
import numpy as np
import jieba
import os

# ==================== 1. 加载已划分的数据 ====================
print("正在加载训练/验证/测试集...")
train_df = pd.read_csv('splits/news_basic_train.csv')
val_df   = pd.read_csv('splits/news_basic_val.csv')
test_df  = pd.read_csv('splits/news_basic_test.csv')

# 合并训练+验证构建词汇表（推荐做法）
full_train_df = pd.concat([train_df, val_df], ignore_index=True)

texts = full_train_df['news_text_raw'].astype(str).tolist()
labels = full_train_df['isFake'].tolist()

print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
print(f"假新闻比例: {full_train_df['isFake'].mean():.3f}")

# ==================== 2. Jieba 分词 + 构建词汇表 ====================
print("正在进行中文分词...")
jieba.setLogLevel(20)  # 减少 jieba 警告
tokenized_texts = [list(jieba.cut(text)) for text in texts]

all_words = [word for words in tokenized_texts for word in words]
word_count = Counter(all_words)

# 词汇表大小 10000（根据实际数据可调整）
VOCAB_SIZE_LIMIT = 10000
vocab = ['<pad>', '<unk>'] + [w for w, c in word_count.most_common(VOCAB_SIZE_LIMIT - 2)]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
max_len = 200  # 适当增长，中文新闻较长

print(f"词汇表大小: {vocab_size}，最大序列长度: {max_len}")

def text_to_indices(text):
    words = list(jieba.cut(text))
    indices = [word_to_idx.get(w, 1) for w in words[:max_len]]  # <unk>=1
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))  # <pad>=0
    return indices

# ==================== 3. 自定义 Dataset ====================
class NewsDataset(Dataset):
    def __init__ (self, df):
        self.texts = df['news_text_raw'].astype(str).tolist()
        self.labels = df['isFake'].values.astype(int)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = torch.tensor(text_to_indices(self.texts[idx]), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label

batch_size = 64
train_dataset = NewsDataset(train_df)
val_dataset   = NewsDataset(val_df)
test_dataset  = NewsDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# ==================== 4. 设备选择 ====================
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"使用设备: {device}")

# ==================== 5. 模型定义 ====================
embed_dim = 128
dropout_rate = 0.3
hidden_dim = 256

class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, kernel_size=k) for k in [3,4,5]
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128*3, 2)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (B, C, L)
        x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, h_n = self.gru(x)
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ==================== 6. 训练函数（早停）===================
def train_model(model, name):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 7
    counter = 0
    best_path = f"best_{name}.pth"

    for epoch in range(1, 101):
        model.train()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_true, val_preds)

        print(f"{name.upper()} Epoch {epoch:02d} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"{name.upper()} Early stopping!")
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model

# ==================== 7. MC Dropout 预测 ====================
def mc_predict(model, loader, T=15):
    model.train()  # 开启 dropout
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for _ in range(T):
            probs_batch = []
            labels_batch = []
            for seqs, labels in loader:
                seqs = seqs.to(device)
                outputs = model(seqs)
                probs = F.softmax(outputs, dim=1)
                probs_batch.append(probs.cpu().numpy())
                labels_batch.append(labels.numpy())
            all_probs.append(np.concatenate(probs_batch))
            all_labels = np.concatenate(labels_batch)

    avg_probs = np.mean(all_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    prob_fake = avg_probs[:, 1]  # 假新闻概率，用于 AUC
    return preds, prob_fake, all_labels

# ==================== 8. 训练 & 测试 ====================
os.makedirs("models", exist_ok=True)

for name, model_cls in [("cnn", TextCNN), ("lstm", TextLSTM), ("gru", TextGRU)]:
    print(f"\n{'='*20} 训练 {name.upper()} 模型 {'='*20}")
    model = model_cls()
    model = train_model(model, name)

    # 测试集预测（MC Dropout）
    preds, probs, true = mc_predict(model, test_loader, T=15)

    # Macro 指标
    acc = accuracy_score(true, preds)
    p, r, f1, _ = precision_recall_fscore_support(true, preds, average='macro')
    auc = roc_auc_score(true, probs)

    print(f"\n{name.upper()} 测试集结果（Macro）")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {p:.4f}")
    print(f"Recall     : {r:.4f}")
    print(f"F1         : {f1:.4f}")
    print(f"AUC        : {auc:.4f}")
    print("\n详细报告：")
    print(classification_report(true, preds, target_names=['真实新闻', '假新闻'], digits=4))

    # 保存预测结果
    test_df_copy = test_df.copy()
    test_df_copy['pred'] = preds
    test_df_copy['prob_fake'] = probs
    test_df_copy.to_csv(f"splits/test_text_{name}_results.csv", index=False, encoding='utf-8-sig')