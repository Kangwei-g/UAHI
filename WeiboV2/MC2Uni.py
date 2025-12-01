# 文件名：final_uniform_forever_work.py
# 这真的是最后一次了！我发誓！

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import os

# ==================== 1. 强制读取并打印列名（找出真相！）===================
print("正在强制加载并修复所有数据...")

# 打印所有文件的列名，找出真相
print("\n=== 文件列名诊断 ===")
for path in ['news_basic.csv', 'splits/trainval_bert_mc.csv', 'splits/test_bert_mc.csv']:
    df = pd.read_csv(path)
    print(f"{path} 列名: {list(df.columns)}")

# 读取原始数据（可能叫 '标签' 或 'isFake' 或 'label'）
raw = pd.read_csv('news_basic.csv')
print(f"\n原始标签列名: {raw.columns}")

# 找出真正的标签列名
label_col = None
for col in ['isFake', '标签', 'label', 'Label', 'fake']:
    if col in raw.columns:
        label_col = col
        break
if label_col is None:
    raise ValueError("找不到标签列！请检查 news_basic.csv")

print(f"检测到标签列名为: '{label_col}'")

# 强制创建标准标签映射
folder_to_label = dict(zip(raw['news_folder'], raw[label_col]))

# ==================== 2. 读取预测文件（不依赖任何标签）===================
trainval_bert = pd.read_csv('splits/trainval_bert_mc.csv')
test_bert     = pd.read_csv('splits/test_bert_mc.csv')
trainval_crowd = pd.read_csv('splits/trainval_crowd.csv')
test_crowd     = pd.read_csv('splits/test_crowd.csv')

# 合并
trainval_df = trainval_bert.merge(trainval_crowd, on='news_folder', how='left')
test_df     = test_bert.merge(test_crowd,      on='news_folder', how='left')

# 强制添加 isFake 列（从字典映射）
trainval_df['isFake'] = trainval_df['news_folder'].map(folder_to_label)
test_df['isFake']     = test_df['news_folder'].map(folder_to_label)

# 检查是否成功
print(f"\n最终检查：")
print(f"训练集合数: {len(trainval_df)}, 标签非空: {trainval_df['isFake'].notna().all()}")
print(f"测试集合数: {len(test_df)}, 标签非空: {test_df['isFake'].notna().all()}")

# 划分 train / val
train_folders = set(pd.read_csv('splits/news_basic_train.csv')['news_folder'])
val_folders   = set(pd.read_csv('splits/news_basic_val.csv')['news_folder'])

train_df = trainval_df[trainval_df['news_folder'].isin(train_folders)].copy()
val_df   = trainval_df[trainval_df['news_folder'].isin(val_folders)].copy()

print(f"训练集: {len(train_df)}，验证集: {len(val_df)}，测试集: {len(test_df)}")

# ==================== 3. Dataset ====================
class FusionDataset(Dataset):
    def __init__(self, df):
        logvar = np.log(df['prob_fake_std'].values**2 + 1e-12)
        self.x = torch.tensor(np.column_stack([
            df['prob_fake_mean'].values,
            logvar,
            df['crowd_alpha'].values,
            df['crowd_beta'].values
        ]), dtype=torch.float32)
        self.y = torch.tensor(df['isFake'].values.astype(int), dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_loader = DataLoader(FusionDataset(train_df), batch_size=32, shuffle=True)
val_loader   = DataLoader(FusionDataset(val_df),   batch_size=64, shuffle=False)
test_loader  = DataLoader(FusionDataset(test_df),  batch_size=64, shuffle=False)

# ==================== 4. Uniform 模型（正确可学习版）===================
class UniformLMFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )
    def forward(self, x):
        out = self.net(x)
        lower = torch.sigmoid(out[:, 0])
        upper = torch.sigmoid(out[:, 1])
        lower, upper = torch.min(lower, upper), torch.max(lower, upper)
        return lower, upper

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UniformLMFM().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.BCELoss()

print(f"\n开始训练 Uniform LMFM（最终版）...")

best_auc = 0.0
for epoch in range(1, 501):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        l, u = model(x)
        p = (l + u) / 2
        loss = criterion(p, y) + 0.05 * (u - l).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            probs = []
            for x, _ in val_loader:
                x = x.to(device)
                l, u = model(x)
                probs.append(((l + u)/2).cpu().numpy())
            prob = np.concatenate(probs)
            true = val_df['isFake'].values
            auc = roc_auc_score(true, prob)
            print(f"Epoch {epoch:3d} | Val AUC: {auc:.4f}", end="")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), 'best_uniform.pth')
                print("  ← Best!")
            else:
                print()

print(f"\n训练完成！最佳验证 AUC: {best_auc:.4f}")

# ==================== 5. 测试集最终结果 ====================
model.load_state_dict(torch.load('best_uniform.pth', map_location=device))
model.eval()
with torch.no_grad():
    l_all, u_all = [], []
    for x, _ in test_loader:
        x = x.to(device)
        l, u = model(x)
        l_all.append(l.cpu().numpy())
        u_all.append(u.cpu().numpy())
    lower = np.concatenate(l_all).flatten()
    upper = np.concatenate(u_all).flatten()
    prob = (lower + upper) / 2
    pred = (prob >= 0.5).astype(int)
    true = test_df['isFake'].values

acc = accuracy_score(true, pred)
p, r, f1, _ = precision_recall_fscore_support(true, pred, average='macro')
auc = roc_auc_score(true, prob)

print("\n" + "="*70)
print("       RAHI + Uniform(u,v) 最终结果")
print("="*70)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print(f"AUC             : {auc:.4f}")
print(f"平均区间宽度    : {(upper-lower).mean():.4f}")
print("="*70)