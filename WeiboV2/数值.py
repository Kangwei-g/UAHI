# 文件名：dataset_statistics_pure_char.py
# 新闻长度 = 纯字符数（包括所有可见字符、空格、换行、表情）

import json

# 加载你的原始 json 文件
with open('c-0-c-picture删去4个-5-200-1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"原始数据总条数: {len(data)}\n")

# ==================== 基本统计 ====================
total_news = len(data)
fake_news = sum(1 for item in data if item.get('isFake', 0) == 1)
real_news = total_news - fake_news

all_comments = []
all_user_ids = []
attitude_a = 0
attitude_b = 0

for item in data:
    comments = item.get('comments', [])
    all_comments.extend(comments)
    for c in comments:
        all_user_ids.append(c['user_id'])
        if c['attitude'] == 'A':
            attitude_a += 1
        elif c['attitude'] == 'B':
            attitude_b += 1

total_comments = len(all_comments)
unique_users = len(set(all_user_ids))

# ==================== 新闻长度：纯字符数（最简单最准确）===================
news_lengths = [len(str(item['news_text_raw'])) for item in data]
avg_length = sum(news_lengths) / len(news_lengths)
min_length = min(news_lengths)
max_length = max(news_lengths)

# ==================== 输出表格 ====================
print("="*80)
print(" " * 30 + "数据集统计信息（Table 1）")
print("="*80)
print(f"{'指标':<35} {'数值':<15}")
print("-"*80)
print(f"{'总新闻数量':<35} {total_news:<15}")
print(f"{'真实新闻数量':<35} {real_news:<15}")
print(f"{'虚假新闻数量':<35} {fake_news:<15}")

print(f"{'总评论数量':<35} {total_comments:<15}")
print(f"{'唯一用户数量':<35} {unique_users:<15}")
print(f"{'支持真实新闻的评论 (A)':<35} {attitude_a:<15}")
print(f"{'认为新闻为假的评论 (B)':<35} {attitude_b:<15}")

print(f"{'最短新闻长度（字符数）':<35} {min_length:<15}")
print(f"{'最长新闻长度（字符数）':<35} {max_length:<15}")

print("="*80)

# ==================== Markdown 表格（直接粘贴到论文）===================
print("\nMarkdown 表格（直接复制到论文）：\n")
print("| 指标                     | 数值       |")
print("|--------------------------|------------|")
print(f"| 总新闻数量               | {total_news}       |")
print(f"| 真实新闻                 | {real_news}       |")
print(f"| 虚假新闻                 | {fake_news}       |")
print(f"| 虚假新闻比例             | {fake_news/total_news*100:.2f}%     |")
print(f"| 总评论数量               | {total_comments}       |")
print(f"| 唯一用户数量             | {unique_users}       |")
print(f"| 支持真实新闻评论 (A)     | {attitude_a}       |")
print(f"| 认为新闻为假评论 (B)     | {attitude_b}       |")
print(f"| 平均新闻长度（字符数）   | {avg_length:.1f}     |")
print(f"| 最短新闻长度             | {min_length}       |")
print(f"| 最长新闻长度             | {max_length}       |")
print(f"| 平均每篇新闻评论数       | {total_comments/total_news:.2f}     |")
print(f"| 平均每个用户评论数       | {total_comments/unique_users:.2f}     |")

print("\n统计完成！长度为纯字符数（包括空格、标点、表情），最符合实际！")