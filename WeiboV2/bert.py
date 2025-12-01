# 文件名：bert_finetune_with_progress.py
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score
import numpy as np
import os
from tqdm import tqdm   # ← 新增：进度条神器
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # Moved here from transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# ==================== 1. 数据加载 ====================
print("正在加载数据...")
train_df = pd.read_csv('splits/news_basic_train.csv')
val_df   = pd.read_csv('splits/news_basic_val.csv')
test_df  = pd.read_csv('splits/news_basic_test.csv')

print(f"训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

# ==================== 2. Dataset & Tokenizer ====================
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
max_len = 400

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

batch_size = 16
train_loader = DataLoader(NewsDataset(train_df), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(NewsDataset(val_df),   batch_size=batch_size)
test_loader  = DataLoader(NewsDataset(test_df),  batch_size=batch_size)

# ==================== 3. 设备 & 模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# ==================== 4. 带进度条的训练函数 ====================
accumulation_steps = 4
epochs = 30
lr = 2e-5

def train_bert():
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    best_path = 'best_bert_fake_news.pth'

    # 主循环进度条
    epoch_bar = tqdm(range(epochs), desc="BERT Fine-tuning", unit="epoch", leave=True)

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        # 训练批次进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for step, batch in enumerate(train_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 实时更新 loss
            train_bar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})

        # ==================== 验证阶段 ====================
        model.eval()
        val_preds, val_true = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val  ]", leave=False)

        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        avg_loss = total_loss / len(train_loader)

        # 更新主进度条
        epoch_bar.set_postfix({
            'val_acc': f'{val_acc:.4f}',
            'loss': f'{avg_loss:.4f}',
            'best': f'{best_val_acc:.4f}'
        })

        print(f" → Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f} | Loss: {avg_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print("   → 新最佳模型已保存！")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping！")
                break

    print(f"\n训练完成！最佳验证 Acc: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model

# ==================== 5. MC Dropout 预测（也带进度条）===================
def mc_dropout_predict(model, loader, T=10):
    model.train()
    all_probs_list = []
    labels_list = None

    mc_bar = tqdm(range(T), desc="MC Dropout 预测", leave=False)

    with torch.no_grad():
        for _ in mc_bar:
            probs_batch = []
            labels_batch = []
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)
                probs_batch.append(probs.cpu().numpy())
                labels_batch.append(labels.numpy())

            all_probs_list.append(np.concatenate(probs_batch))
            labels_list = np.concatenate(labels_batch)

    avg_probs = np.mean(all_probs_list, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    prob_fake = avg_probs[:, 1]
    return preds, prob_fake, labels_list

# ==================== 6. 开始训练！===================
os.makedirs("models", exist_ok=True)

model = train_bert()
model.eval()

print("\n" + "="*70)
print("          BERT 测试集最终结果（MC Dropout T=10）")
print("="*70)

preds, probs, true = mc_dropout_predict(model, test_loader, T=10)

acc = accuracy_score(true, preds)
p, r, f1, _ = precision_recall_fscore_support(true, preds, average='macro')
auc = roc_auc_score(true, probs)

print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print(f"AUC             : {auc:.4f}")
print("="*70)
print(classification_report(true, preds, target_names=['真实新闻', '假新闻'], digits=4))

# 保存结果
result_df = test_df.copy()
result_df['pred'] = preds
result_df['prob_fake'] = probs
result_df['correct'] = (preds == true)
result_df.to_csv('splits/test_bert_results.csv', index=False, encoding='utf-8-sig')
print("\n预测结果已保存 → splits/test_bert_results.csv")