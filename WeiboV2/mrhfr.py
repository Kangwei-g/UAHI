# 文件名：mrhfr_final.py
# 复现 MRHFR：新闻与评论一致性 + BERT + 分类

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm

# ==================== 1. 防丢标签加载 ====================
print("正在加载数据...")
news_basic = pd.read_csv('news_with_comments_fixed.csv')
comments_df = pd.read_csv('news_with_comments_fixed.csv', dtype={'user_id': str})

folder_to_label = dict(zip(news_basic['news_folder'], news_basic['isFake']))
train_folders = set(pd.read_csv('splits/news_basic_train.csv')['news_folder'])
test_folders = set(pd.read_csv('splits/news_basic_test.csv')['news_folder'])

# 评论分组
news_to_comments = comments_df.groupby('news_folder')['comment_text_raw'].apply(list).to_dict()

# ==================== 2. BERT 加载 ====================
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert = BertModel.from_pretrained('bert-base-chinese')
bert.eval()  # 只做编码，不训练
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
bert = bert.to(device)


@torch.no_grad()
def get_bert_embedding(text):
    if not text.strip():
        text = "[PAD]"
    inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]


# ==================== 3. 计算一致性特征 ====================
print("正在计算新闻-评论一致性特征...")
consistency_scores = []

for folder in tqdm(news_basic['news_folder']):
    news_text = str(news_basic[news_basic['news_folder'] == folder]['news_text_raw'].iloc[0])
    comments = news_to_comments.get(folder, [])

    news_emb = get_bert_embedding(news_text)

    if not comments:
        consistency_scores.append(0.0)
        continue

    comment_embs = []
    for c in comments[:20]:  # 最多20条
        comment_embs.append(get_bert_embedding(str(c)))

    comment_embs = np.vstack(comment_embs)
    # 余弦相似度
    sims = np.sum(news_emb * comment_embs, axis=1) / (
            np.linalg.norm(news_emb) * np.linalg.norm(comment_embs, axis=1) + 1e-8
    )
    consistency = np.mean(sims)
    consistency_scores.append(float(consistency))

news_basic['consistency'] = consistency_scores


# ==================== 4. Dataset ====================
class MRHFRDataset(Dataset):
    def __init__(self, folders):
        df = news_basic[news_basic['news_folder'].isin(folders)].copy()
        self.texts = df['news_text_raw'].astype(str).tolist()
        self.consistency = torch.tensor(df['consistency'].values, dtype=torch.float32)
        self.labels = torch.tensor(df['isFake'].values, dtype=torch.long)
        self.folders = df['news_folder'].tolist()

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return self.texts[i], self.consistency[i], self.labels[i]


def collate_fn(batch):
    texts = [b[0] for b in batch]
    cons = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return texts, cons, labels


train_dataset = MRHFRDataset(train_folders)
test_dataset = MRHFRDataset(test_folders)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


# ==================== 5. MRHFR 模型 ====================
class MRHFR(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768 + 1, 2)  # BERT [CLS] + 一致性分数

    def forward(self, texts, consistency):
        inputs = tokenizer(texts, return_tensors='pt', max_length=256, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        text_feat = outputs.last_hidden_state[:, 0, :]  # [CLS]
        feat = torch.cat([text_feat, consistency.unsqueeze(1)], dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)


model = MRHFR().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ==================== 6. 训练 ====================
print("开始训练 MRHFR...")
best_auc = 0.0
for epoch in range(1, 11):
    model.train()
    for texts, cons, y in train_loader:
        cons, y = cons.to(device), y.to(device)
        out = model(texts, cons)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        model.eval()
        all_prob, all_true = [], []
        with torch.no_grad():
            for texts, cons, y in test_loader:
                cons = cons.to(device)
                out = model(texts, cons)
                prob = F.softmax(out, dim=1)[:, 1]
                all_prob.append(prob.cpu().numpy())
                all_true.append(y.numpy())
        prob = np.concatenate(all_prob)
        true = np.concatenate(all_true)
        auc = roc_auc_score(true, prob)
        print(f"Epoch {epoch} | Test AUC: {auc:.4f}", end="")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_mrhfr.pth')
            print("  ← Best!")
        else:
            print()

# ==================== 7. 最终5个指标 ====================
model.load_state_dict(torch.load('best_mrhfr.pth', map_location=device))
model.eval()
all_prob, all_pred, all_true = [], [], []
with torch.no_grad():
    for texts, cons, y in test_loader:
        cons = cons.to(device)
        out = model(texts, cons)
        prob = F.softmax(out, dim=1)[:, 1]
        pred = out.argmax(dim=1)
        all_prob.append(prob.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.numpy())

prob = np.concatenate(all_prob).flatten()
pred = np.concatenate(all_pred).flatten()
true = np.concatenate(all_true).flatten()

acc = accuracy_score(true, pred)
p, r, f1, _ = precision_recall_fscore_support(true, pred, average='macro')
auc = roc_auc_score(true, prob)

print("\n" + "=" * 80)
print("               MRHFR 最终结果（一致性特征）")
print("=" * 80)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print(f"AUC             : {auc:.4f}")
print("=" * 80)