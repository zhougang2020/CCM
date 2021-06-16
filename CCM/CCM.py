import numpy as np
import math
import matplotlib.pyplot as plt   # 导入模块 matplotlib.pyplot，并简写成 plt
from scipy.stats import pearsonr
import sys
import FNN
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

def distance( p1, p2):
    '''计算两点间距'''
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)

#股价时间序列
x = [13.6,14.12,13.31,13.31,13.31,12.72,13.85,13.85,14.83,15.63,15.63,
     15.63, 15.53,19.459999,20.57,20.15,20.15,20.15,20.15,20.99,19.379999,
     19.26,18.84,18.84,18.84,18.84,17.25,17.370001,18.360001,18.08,17.690001,
     17.690001,17.690001,19.940001,19.950001,31.4,39.91,35.5,35.5,35.5,35.5,
     39.360001,39.119999,43.029999,65.010002,65.010002,65.010002,76.790001,
     147.979996,347.51001,193.600006,325,325,325,225,90,92.410004]
#发帖量时间序列
y = [99524, 94835, 84619, 64246, 41946, 63609, 72329, 78416, 76054, 80304,
     65834, 46872, 75153, 85471, 90157, 77797, 52463, 42719, 41034, 69179,
     79319, 75461, 79015, 60783, 43368, 45177, 59947, 74863, 89597, 99150,
     93408, 79525, 52481, 79725, 90981, 94626, 126559, 159105, 103659, 55873,
     67960, 111194, 140057, 129423, 179115, 239148, 75091, 75091, 75091, 92726,
     1237856, 1213702, 754480, 410245, 629727, 799227, 421188]


class CCM:

    def __init__(self, tau, E, Cause, result):
        self.tao = tau
        self.E = E
        self.Cause = Cause
        self.result = result
        self.causal_strength = []

    def main(self,cause,result,l,tao,E):
        x = cause
        y = result
        cause_predit = []
        Mx = np.zeros((l - (E - 1) * tao, E), dtype=float)  # 创建吸引子矩阵
        My = np.zeros((l - (E - 1) * tao, E), dtype=float)  # 创建吸引子矩阵
        Dx = np.zeros((l - (E - 1) * tao, l - (E - 1) * tao), dtype=float)  # 创建距离矩阵
        index = np.zeros((l - (E - 1) * tao, E + 1), dtype=int)  # 创建最近邻索引矩阵
        for j in range((E - 1) * tao, l):  # 填充矩阵Mx
            for k in range(0, E):
                Mx[j - (E - 1) * tao][k] = x[j - k * tao]

        for m in range((E - 1) * tao, l):  # 填充矩阵My
            for n in range(0, E):
                My[m - (E - 1) * tao][n] = y[m - n * tao]

        for k in range(0, l - (E - 1) * tao):  # 填充距离矩阵Dx
            my_now = My[k, :]
            for i in range(0, l - (E - 1) * tao):
                my = My[i, :]
                if k == i or (my_now == my).all():
                    Dx[k][i] = sys.maxsize
                    continue
                Dx[k][i] = distance(my_now, my)

        for k in range(0, l - (E - 1) * tao):  # 填充最近邻索引矩阵
            dis = Dx[k, :]
            arr = np.array(dis)
            dis = np.argsort(arr)
            for i in range(0, E + 1):
                index[k][i] = dis[i]

        for k in range(0, l - (E - 1) * tao):  # 填充预测的序列
            sita = 0
            for i in range(0, E + 1):  # 这个循环求出sita值
                mi = math.exp(-distance(My[k, :], My[index[k][i], :]) / distance(My[k, :], My[index[k][0], :]))
                sita += mi
            tem = 0
            for h in range(0, E + 1):
                wh = math.exp(-distance(My[k, :], My[index[k][h], :]) / distance(My[k, :], My[index[k][0], :])) / sita
                tem += wh * x[index[k][h] + (E - 1) * tao]
            cause_predit.append(tem)
        return cause_predit




    def getcausality(self):
        for lag in range(-12, 5):
            if lag < 0:
                result = self.result[-lag:]
                cause = self.Cause[:len(self.Cause) + lag]
            else:
                result = self.result[:len(self.Cause) - lag]
                cause = self.Cause[lag:]
            cause_predit = self.main(cause, result, len(cause), self.tao, self.E)
            cause = cause[(self.E - 1) * self.tao:]
            self.causal_strength.append(math.fabs(pearsonr(cause, cause_predit)[0]))
        return self.causal_strength


'''
tao = int(input('τ='))#输入嵌入时延
E = int(input('E='))#输入嵌入维度
taud = int(input('τ='))


x = [0.2, 0.6048, 0.9034841088000001, 0.3296181695153873]
y= [0.4, 0.8984000000000001, 0.3006477631999997, 0.7709448069124968]
z= [0.6, 0.8495999999999999, 0.3083572224000004, 0.6999595973498268]
for t in range(3, 200):
    x_temp = x[t] * (3.78 - 3.78 * x[t])# -0.07*y[t])
    x.append(x_temp)
    y_temp = y[t] * (3.77 - 3.77 * y[t] - 0.7 * x[t-taud])
    y.append(y_temp)
    z_temp = z[t] * (3.7 - 3.7 * z[t] - 0.32 * x[t])
    z.append(z_temp)

fnn = FNN.FNN(tao, y)
E = fnn.getdimension()
print('E=', E)


ccm = CCM(tao, E, x, y)
reverseccm =CCM(tao, E, y, x)
casual_strength = ccm.getcausality()
reverse_casual_strength = reverseccm.getcausality()
print(casual_strength)
print(reverse_casual_strength)
x_axis = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
plt.ylim(0, 1, 0.1)
plt.plot(x_axis, casual_strength, label = 'reddit_comments cross map Stock price')
plt.plot(x_axis, reverse_casual_strength, label = 'Stock price cross map reddit_comments')
plt.xlabel('lag')
plt.ylabel('cross map skill')
plt.title(u'时延-因果图', fontproperties=font)
plt.legend()
plt.show()'''
