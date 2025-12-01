# 文件名：bert_test_with_uncertainty.py
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score
import numpy as np
from tqdm import tqdm
import os

# ==================== 1. 配置 ====================
model_path = 'best_bert_fake_news.pth'   # 你训练时保存的路径
model_name = 'bert-base-chinese'
max_len = 400
batch_size = 16
mc_samples = 10   # MC Dropout 采样次数（越大标准差越准，建议 15~30）

# ==================== 2. 数据集定义（和训练时完全一致）===================
tokenizer = BertTokenizer.from_pretrained(model_name)

class NewsDataset(Dataset):
    def __init__(self, df):
        self.texts = df['news_text_raw'].astype(str).values
        self.labels = df['isFake'].values.astype(int)

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 加载测试集
test_df = pd.read_csv('splits/news_basic_test.csv')
test_dataset = NewsDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== 3. 加载模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.train()   # 必须开启 train() 才能激活 dropout → MC Dropout

print(f"已加载模型: {model_path}")

# ==================== 4. MC Dropout 多轮前向传播（带进度条）===================
print(f"正在进行 MC Dropout 预测（采样 {mc_samples} 次）...")
all_prob_fake = []   # 每条新闻每次采样的“假新闻”概率

pbar = tqdm(range(mc_samples), desc="MC Sampling")
with torch.no_grad():
    for _ in pbar:
        probs_per_sample = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            probs_per_sample.append(probs[:, 1].cpu().numpy())  # 假新闻概率

        all_prob_fake.append(np.concatenate(probs_per_sample))

# 转换为 (n_samples, n_news)
all_prob_fake = np.stack(all_prob_fake)  # shape: (mc_samples, N)

# ==================== 5. 计算预测结果 & 不确定性 ====================
pred_prob_mean = all_prob_fake.mean(axis=0)      # 最终预测概率（均值）
pred_prob_std  = all_prob_fake.std(axis=0)       # 预测不确定性（标准差）
final_pred     = (pred_prob_mean >= 0.5).astype(int)
true_labels    = test_df['isFake'].values

# ==================== 6. 计算 5 个指标并打印 ====================
acc = accuracy_score(true_labels, final_pred)
p, r, f1, _ = precision_recall_fscore_support(true_labels, final_pred, average='macro')
auc = roc_auc_score(true_labels, pred_prob_mean)

print("\n" + "="*60)
print("          BERT 测试集最终结果（MC Dropout）")
print("="*60)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro_precision : {p:.4f}")
print(f"Macro_recall    : {r:.4f}")
print(f"Macro_f1        : {f1:.4f}")
print(f"AUC             : {auc:.4f}")
print("="*60)

# ==================== 7. 保存结果（包含均值和标准差）===================
result_df = test_df.copy()
result_df['pred_label']      = final_pred
result_df['prob_fake_mean']  = pred_prob_mean
result_df['prob_fake_std']   = pred_prob_std     # ← 不确定性指标
result_df['correct']         = (final_pred == true_labels)

save_path = 'splits/test_bert_with_uncertainty.csv'
result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"\n预测结果（含均值 & 标准差）已保存 → {save_path}")

# 顺便也保存一份只含关键列的轻量版（方便画不确定性图）
uncertainty_df = result_df[['news_folder', 'isFake', 'pred_label', 'correct',
                            'prob_fake_mean', 'prob_fake_std']].copy()
uncertainty_df.to_csv('splits/test_uncertainty_summary.csv', index=False, encoding='utf-8-sig')
print("不确定性摘要已保存 → splits/test_uncertainty_summary.csv")