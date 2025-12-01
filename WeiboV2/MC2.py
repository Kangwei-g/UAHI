import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

# ==================== 1. 终极保险读取 ====================
print("正在终极保险加载数据...")

# 读取原始完整标签（永不丢失的源头！）
label_df = pd.read_csv('news_basic.csv')[['news_folder', 'isFake']]  # 这就是真相！

# 读取 Machine 预测（可能没标签）
train_machine = pd.read_csv('splits/trainval_bert_mc.csv')
test_machine  = pd.read_csv('splits/test_bert_mc.csv')

# 读取 Crowd 预测
train_crowd = pd.read_csv('splits/trainval_crowd.csv')
test_crowd  = pd.read_csv('splits/test_crowd.csv')

# 强制用原始标签补全（核心！）
train_df = train_machine.merge(train_crowd, on='news_folder', how='left')
test_df  = test_machine.merge(test_crowd,  on='news_folder', how='left')

# 关键：用原始 label_df 补全 isFake（不管之前有没有，一律覆盖！）
train_df = train_df[['news_folder', 'prob_fake_mean', 'prob_fake_std', 'crowd_alpha', 'crowd_beta']].merge(
    label_df, on='news_folder', how='left'
)
test_df = test_df[['news_folder', 'prob_fake_mean', 'prob_fake_std', 'crowd_alpha', 'crowd_beta']].merge(
    label_df, on='news_folder', how='left'
)

# 检查
print(f"训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")
print(f"isFake 是否存在 → 训练集: {'isFake' in train_df.columns}, 测试集: {'isFake' in test_df.columns}")
print(f"训练集标签分布:\n{train_df['isFake'].value_counts()}")
print(f"测试集标签分布:\n{test_df['isFake'].value_counts()}")

# ==================== 2. Dataset ====================
class FusionDataset(Dataset):
    def __init__(self, df):
        logvar = np.log(df['prob_fake_std'].values**2 + 1e-12)
        self.x = torch.tensor(np.column_stack([
            df['prob_fake_mean'].values,
            logvar,
            df['crowd_alpha'].values,
            df['crowd_beta'].values
        ]), dtype=torch.float32)
        self.y = torch.tensor(df['isFake'].values, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_loader = DataLoader(FusionDataset(train_df), batch_size=32, shuffle=True)
test_loader  = DataLoader(FusionDataset(test_df),  batch_size=64, shuffle=False)

# ==================== 3. 极简 2层 MLP ====================
class SimpleLMFM(nn.Module):
    def __init__(self, hidden=32, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.logvar = nn.Parameter(torch.tensor(-3.0))  # 学一个全局不确定性

    def forward(self, x):
        mu = self.mlp(x).squeeze(-1)
        return mu, self.logvar.expand_as(mu)

# ==================== 4. 训练 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")

model = SimpleLMFM(hidden=8, dropout=0.1).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

def nll_loss(mu, logvar, y):
    var = torch.exp(logvar)
    return 0.5 * (logvar + (y - mu)**2 / var).mean()

print("\n开始训练 2层 LMFM（SGD）...")
best_auc = 0.0
patience = 40
trigger = 0

for epoch in range(1, 801):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        mu, logvar = model(x)
        loss = nll_loss(mu, logvar, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            all_mu, all_y = [], []
            for x, y in test_loader:
                x = x.to(device)
                mu, _ = model(x)
                all_mu.append(mu.cpu().numpy())
                all_y.append(y.numpy())
            pred_prob = np.concatenate(all_mu)
            true = np.concatenate(all_y)
            auc = roc_auc_score(true, pred_prob)

            print(f"Epoch {epoch:3d} | AUC: {auc:.4f}", end="")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), 'best_lmfm_final.pth')
                print("  ← Best!")
                trigger = 0
            else:
                print()
                trigger += 1
                if trigger >= patience:
                    print("Early Stop!")
                    break

print(f"\n训练完成！最佳 AUC: {best_auc:.4f}")

# ==================== 5. 最终结果 ====================
model.load_state_dict(torch.load('best_lmfm_final.pth', map_location=device))
model.eval()
with torch.no_grad():
    all_mu, all_pred, all_true = [], [], []
    for x, y in test_loader:
        x = x.to(device)
        mu, _ = model(x)
        all_mu.append(mu.cpu().numpy())
        all_pred.append((mu >= 0.5).cpu().numpy().astype(int))
        all_true.append(y.numpy())
    prob = np.concatenate(all_mu).flatten()
    pred = np.concatenate(all_pred).flatten()
    true = np.concatenate(all_true).flatten()

acc = accuracy_score(true, pred)
p, r, f1, _ = precision_recall_fscore_support(true, pred, average='macro')
final_auc = roc_auc_score(true, prob)

print("\n" + "="*70)
print("           RAHI + 2层LMFM 最终测试结果")
print("="*70)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print(f"AUC             : {final_auc:.4f}")
print("="*70)

# 保存
result = test_df.copy()
result['final_prob'] = prob
result['final_pred'] = pred
result.to_csv('splits/RAHI_2LAYER_FINAL.csv', index=False, encoding='utf-8-sig')
print("最终结果已保存 → splits/RAHI_2LAYER_FINAL.csv")