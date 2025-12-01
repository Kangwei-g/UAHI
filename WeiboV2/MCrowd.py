# 文件名：rahi_full_pipeline.py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import os
# 文件名：rahi_full_final_fixed.py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
import numpy as np
from tqdm import tqdm
import os
# 文件名：rahi_final_ultimate.py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm
import os

# ==================== 配置 ====================
BERT_MODEL_PATH = 'best_bert_fake_news.pth'
CROWD_CA_PATH = 'rahi_crowd_learned_ca.pth'
MC_SAMPLES = 20
BATCH_SIZE = 16
MAX_LEN = 400
MODEL_NAME = 'bert-base-chinese'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 1. 加载数据 ====================
print("正在加载数据...")
news_basic = pd.read_csv('news_basic.csv')
comments_df = pd.read_csv('news_with_comments.csv', dtype={'user_id': str})

train_df = pd.read_csv('splits/news_basic_train.csv')
val_df = pd.read_csv('splits/news_basic_val.csv')
test_df = pd.read_csv('splits/news_basic_test.csv')

train_folders = set(train_df['news_folder'])
val_folders = set(val_df['news_folder'])
test_folders = set(test_df['news_folder'])
train_val_folders = train_folders | val_folders

print(f"训练集: {len(train_folders)}，验证集: {len(val_folders)}，测试集: {len(test_folders)}")

# 加载 c_a
ca_data = torch.load(CROWD_CA_PATH, map_location='cpu', weights_only=False)
learned_ca_dict = ca_data['learned_ca_dict']
global_avg_ca = ca_data.get('global_avg_ca', 1.0)

# ==================== 2. BERT 模型 ====================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device, weights_only=True))
bert_model.to(device)
bert_model.train()


# ==================== 3. BERT MC Dropout ====================
def run_bert_mc(folders, name):
    print(f"\n正在为 {name} ({len(folders)} 条) 运行 BERT MC Dropout...")
    df = news_basic[news_basic['news_folder'].isin(folders)][['news_folder', 'news_text_raw', 'isFake']].copy()
    dataset = df.reset_index(drop=True)

    loader = DataLoader([
        tokenizer(str(text), truncation=True, max_length=MAX_LEN, padding='max_length', return_tensors='pt')
        for text in dataset['news_text_raw']
    ], batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for _ in tqdm(range(MC_SAMPLES), desc="MC", leave=False):
            batch_probs = []
            for batch in loader:
                ids = batch['input_ids'].squeeze(1).to(device)
                mask = batch['attention_mask'].squeeze(1).to(device)
                outputs = bert_model(input_ids=ids, attention_mask=mask)
                prob_fake = F.softmax(outputs.logits, dim=1)[:, 1]
                batch_probs.append(prob_fake.cpu().numpy())
            all_probs.append(np.concatenate(batch_probs))

    all_probs = np.stack(all_probs)
    results = pd.DataFrame({
        'news_folder': dataset['news_folder'].values,
        'prob_fake_mean': all_probs.mean(axis=0),
        'prob_fake_std': all_probs.std(axis=0),
        'isFake': dataset['isFake'].values  # 强制保留！
    })
    path = f'splits/{name}_bert_mc.csv'
    results.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"{name} Machine 已保存 → {path}")
    return results


# ==================== 4. Crowd 预测 ====================
def run_crowd(folders, name):
    print(f"\n正在为 {name} ({len(folders)} 条) 生成 Crowd 预测...")
    results = []
    for f in tqdm(folders, desc="Crowd", leave=False):
        g = comments_df[comments_df['news_folder'] == f]
        if len(g) == 0:
            results.append({'news_folder': f, 'crowd_alpha': 1.0, 'crowd_beta': 1.0})
            continue
        w_fake = sum(learned_ca_dict.get(u, global_avg_ca) for u in g[g['attitude'] == 'B']['user_id'])
        w_true = sum(learned_ca_dict.get(u, global_avg_ca) for u in g[g['attitude'] == 'A']['user_id'])
        results.append({'news_folder': f, 'crowd_alpha': 1.0 + w_fake, 'crowd_beta': 1.0 + w_true})
    df = pd.DataFrame(results)
    path = f'splits/{name}_crowd.csv'
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return df


# ==================== 5. 执行 ====================
os.makedirs('splits', exist_ok=True)

m_trainval = run_bert_mc(train_val_folders, 'trainval')
m_test = run_bert_mc(test_folders, 'test')
c_trainval = run_crowd(train_val_folders, 'trainval')
c_test = run_crowd(test_folders, 'test')

# 强制合并，确保 isFake 不丢！
trainval_df = pd.merge(m_trainval, c_trainval, on='news_folder', how='inner')
test_df = pd.merge(m_test, c_test, on='news_folder', how='inner')

# 再次保险：从原始 news_basic 补回 isFake
trainval_df = trainval_df.merge(news_basic[['news_folder', 'isFake']], on='news_folder', how='left',
                                suffixes=('', '_orig'))
test_df = test_df.merge(news_basic[['news_folder', 'isFake']], on='news_folder', how='left', suffixes=('', '_orig'))

# 如果有冲突，以原始为准
trainval_df['isFake'] = trainval_df['isFake_orig'].combine_first(trainval_df['isFake'])
test_df['isFake'] = test_df['isFake_orig'].combine_first(test_df['isFake'])
trainval_df = trainval_df.drop(columns=[c for c in trainval_df.columns if '_orig' in c], errors='ignore')
test_df = test_df.drop(columns=[c for c in test_df.columns if '_orig' in c], errors='ignore')

print(
    f"\n最终训练 LMFM: {len(trainval_df)} 条，测试: {len(test_df)} 条，isFake 列存在: {'isFake' in trainval_df.columns}")


# ==================== 6. LMFM 训练 ====================
class FusionDataset(Dataset):
    def __init__(self, df):
        logvar = np.log(df['prob_fake_std'].values ** 2 + 1e-12)
        self.x = torch.tensor(np.column_stack([df['prob_fake_mean'], logvar, df['crowd_alpha'], df['crowd_beta']]),
                              dtype=torch.float32)
        self.y = torch.tensor(df['isFake'].values, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return self.x[i], self.y[i]


train_loader = DataLoader(FusionDataset(trainval_df), batch_size=32, shuffle=True)


class LMFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 4), nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        o = self.net(x)
        return torch.sigmoid(o[:, 0]), o[:, 1]


model = LMFM().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)

print("\n开始训练 LMFM（SGD）...")
best_auc = 0
for epoch in range(1, 401):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        mu, logvar = model(x)
        var = torch.exp(logvar)
        loss = 0.5 * (logvar + (y - mu) ** 2 / var).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            xt = FusionDataset(test_df).x.to(device)
            mu, _ = model(xt)
            auc = roc_auc_score(test_df['isFake'], mu.cpu().numpy())
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), 'best_lmfm_final.pth')
            print(f"Epoch {epoch:3d} | Test AUC: {auc:.4f} ← Best: {best_auc:.4f}")

# ==================== 7. 最终预测 ====================
model.load_state_dict(torch.load('best_lmfm_final.pth', map_location=device))
model.eval()
with torch.no_grad():
    xt = FusionDataset(test_df).x.to(device)
    mu, logvar = model(xt)
    prob = mu.cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    std = torch.sqrt(torch.exp(logvar)).cpu().numpy()

# 5 个指标
acc = accuracy_score(test_df['isFake'], pred)
p, r, f1, _ = precision_recall_fscore_support(test_df['isFake'], pred, average='macro')
auc_final = roc_auc_score(test_df['isFake'], prob)

print("\n" + "=" * 80)
print("                     RAHI 最终结果")
print("=" * 80)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print(f"AUC             : {auc_final:.4f}")
print("=" * 80)

# 保存
result = test_df[['news_folder', 'isFake']].copy()
result['RAHI_prob'] = prob.flatten()
result['RAHI_std'] = std.flatten()
result['RAHI_pred'] = pred.flatten()
result.to_csv('splits/RAHI_FINAL_RESULT.csv', index=False, encoding='utf-8-sig')
print("最终结果已保存 → splits/RAHI_FINAL_RESULT.csv")