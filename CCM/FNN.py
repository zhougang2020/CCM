import numpy as np
import math
import sys


def distance( p1, p2):
    '''计算两点间距'''
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)

def get_attractor_size(timeseries):
        time_num = 0
        for i in range(0, len(timeseries)):
            time_num += timeseries[i]
        average = time_num /len(timeseries)
        temp = 0
        for i in range(0, len(timeseries)):
            temp += pow((timeseries[i]-average), 2)
        return pow(temp /len(timeseries), 0.5)

class FNN:
    def __init__(self, tau, timeseries):
        self.tau = tau
        self.timeseries = timeseries
        self.timeseries_length = len(self.timeseries)
        self.false_percentage = 0.05
        self.RA = get_attractor_size(self.timeseries)

    def generate_attracter(self, timeseries, E, tau):
        l = len(timeseries)
        M = np.zeros((l - (E - 1) * tau, E), dtype=float)
        for j in range((E - 1) * tau, l):  # 填充矩阵Mx
            for k in range(0, E):
                M[j - (E - 1) * tau][k] = timeseries[j - k * tau]
        return M


    def generate_nearstneighbour(self, M):
        nearstneighbour = []
        for i in range(0 , M.shape[0]):
            min_distance = sys.maxsize
            position = -1
            for j in range(0, M.shape[0]):
                if (M[i, :] == M[j, :]).all():
                    continue
                if distance(M[i, :], M[j, :]) < min_distance:
                    min_distance = distance(M[i, :], M[j, :])
                    position = j
            nearstneighbour.append(position)
        return nearstneighbour


    def getdimension(self):
        dimension = 1
        FNN_percentage = [sys.maxsize]
        while True:
            dimension += 1
            false_count = 0
            M = self.generate_attracter(self.timeseries, dimension, self.tau)
            for i in range(0,self.tau):
                M = np.delete(M, 0, axis=0)
            nearstneighbour = self.generate_nearstneighbour(M)
            M_high_dim = self.generate_attracter(self.timeseries, dimension + 1, self.tau)
            for i in range(0, M.shape[0]):
                judge_first = math.fabs(M_high_dim[i][dimension]-M_high_dim[nearstneighbour[i]][dimension])/distance(M[i, :], M[nearstneighbour[i], :]) > 15
                judge_second = distance(M_high_dim[i, :], M_high_dim[nearstneighbour[i], :]) / self.RA > 2
                if judge_first == True or judge_second == True:
                    false_count += 1
            FNN_percentage.append(false_count / M.shape[0])
            #print('dimension=',dimension, 'false_count=', false_count, 'FNN_percentage=', FNN_percentage[dimension-1])
            if FNN_percentage[dimension-1]-FNN_percentage[dimension-2] >= 0 or dimension > 15:
                break
        return dimension-1
