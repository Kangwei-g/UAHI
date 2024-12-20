import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
import torch.distributions as td
device = torch.device('cpu')

plt.rcParams['font.sans-serif'] = ['SimSun']  # SimSun是宋体的英文名称
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# ab_pairs = torch.Tensor([0.6, 0.4], [6, 4])
# ab_pairs = ((0.2,0.8),(0.4,1.6),(1,4),(20,80),(40,160))
# a=ab_pairs[0]
# b=ab_pairs[1]
# print("a",a)
# print("b",b)
# beta_dist = td.beta.Beta(a, b)
# crowd_mean = beta_dist.mean
# crowd_var = beta_dist.variance
# print("crowd_mean",crowd_mean)
# print("crowd_var",crowd_var)

x = np.linspace(0, 1, 1002)[1:-1]
mu, sigma = 0.252591322362422, 0.059105787483636 # 均值和标准差

x2 = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
#y2 = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
y2= norm.pdf(x, mu, sigma)


'''
for a, b in ab_pairs:
    print(a, b)
    dist = beta(a, b)
    y = dist.pdf(x)
    plt.plot(x, y, label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))
'''

dist = beta(8, 92)
y = dist.pdf(x)
encoder_mu=0.264414101839065
encoder_std=0.178727880120277
encoder_mu_up=encoder_mu+encoder_std
encoder_mu_down=encoder_mu-encoder_std

plt.axvline(encoder_mu,label="UAHI预测均值",c='red',ls='--',)
plt.axvline(encoder_mu_up,label="UAHI预测方差",c='green',ls='--')
plt.axvline(encoder_mu_down,c='green',ls='--')
plt.plot(x, y,label="人群智能-Beta分布")
plt.plot(x2, y2,label="机器智能-高斯分布")
# 设置标题
#plt.title(u'UAHI模型各模块输出分布对比图')
# 设置 x,y 轴取值范围
# plt.xlim(0, 1)
# plt.ylim(0, 10)
plt.legend()
plt.show()
# plt.savefig("./beta.svg", format="svg")


# beta_dist = td.beta.Beta(60, 40)
# crowd_mean = beta_dist.mean
# crowd_var = beta_dist.variance
# print('mean:', crowd_mean)
# print('variance:', crowd_var)
