'''
个人认为svd也是一种函数拟合器，本质上和神经网络相同
'''
from time import sleep

import numpy as np
import torch
from torch import nn
import mv100
import random
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class SVD(nn.Module):


    def batch_generator(self,batchi_size):
        '''
        :param batchi_size:
        :return: 一个生成器返回所有数据信息
        '''
        counter_total=0

        while counter_total<len(self.mvlist):
            batch_date_u = []
            batch_date_i = []
            batch_date_s = []
            for i in range(batchi_size):
                if counter_total+i>=len(self.mvlist):
                    break
                batch_date_u.append(self.mvlist[counter_total+i][0])
                batch_date_i.append(self.mvlist[counter_total+i][1])
                batch_date_s.append(self.mvlist[counter_total+i][2])
                counter_total+=1
            yield batch_date_u,batch_date_i,batch_date_s







    def get_random_batch(self,batch_size=32):
         temp1=[]
         temp2=[]
         temp3=[]
         for i in random.sample(self.mvlist,batch_size):
             temp1.append(i[0])
             temp2.append(i[1])
             temp3.append(i[2])

         return  temp1,temp2,temp3

    def __init__(self,batch=16,kernel_size=100,filename='/home/iiap/桌面/资料/ml-100k/u.data',device='cuda'):
        super().__init__()
        self.k=kernel_size
        self.mvlist=mv100.mv1002list(filename)
        self.U=mv100.get_U_from_list(self.mvlist)
        self.mvarray=mv100.creat_matrix(self.mvlist)

        '''
        初始化物品向量
        '''

        self.bi=nn.Parameter(torch.full((1,self.mvarray.shape[1]),0.0)[0])
        '''
        初始化用户向量
        '''
        self.bu=nn.Parameter(torch.full((1,self.mvarray.shape[0]),0.0)[0])
        '''
        初始化物品矩阵,这里提前转制
        '''
        self.qi=nn.Parameter(torch.full((self.k,self.mvarray.shape[1]),self.U))
        '''
        初始化用户矩阵
        '''
        self.pu=nn.Parameter(torch.full((self.mvarray.shape[0],self.k),self.U))



    def forward(self,u,i):
        '''
        这个函数我需要设计成支持list所以需要全部使用矩阵表示
        :param u:
        :param i:
        :return:
        '''
        # print('前向传播')
        # print(self.U)
        # print(self.pu[u,:].shape)
        #print(self.bi[i]+self.bu[u])\

        r_hat=self.U+self.bi[i]+self.bu[u]+torch.diag(torch.mm(self.pu[u,:],self.qi[:,i]))

        return r_hat

class SVD_plus_plus(nn.Module):
    '''
    因为python继承用的不熟悉，直接复制了
    这次需要新增用户评分历史矩阵，全是01,以及修改前向推理方法
    '''


    def batch_generator(self,batchi_size):
        '''
        :param batchi_size:
        :return: 一个生成器返回所有数据信息
        '''
        counter_total=0

        while counter_total<len(self.mvlist):
            batch_date_u = []
            batch_date_i = []
            batch_date_s = []
            for i in range(batchi_size):
                if counter_total+i>=len(self.mvlist):
                    break
                batch_date_u.append(self.mvlist[counter_total+i][0])
                batch_date_i.append(self.mvlist[counter_total+i][1])
                batch_date_s.append(self.mvlist[counter_total+i][2])
                counter_total+=1
            yield batch_date_u,batch_date_i,batch_date_s


    def get_random_batch(self,batch_size=32):
         temp1=[]
         temp2=[]
         temp3=[]
         for i in random.sample(self.mvlist,batch_size):
             temp1.append(i[0])
             temp2.append(i[1])
             temp3.append(i[2])

         return  temp1,temp2,temp3

    def __init__(self,batch=16,kernel_size=100,filename='/home/iiap/桌面/资料/ml-100k/u.data',device='cuda'):
        super().__init__()
        self.k=kernel_size
        self.mvlist=mv100.mv1002list(filename)
        self.U=mv100.get_U_from_list(self.mvlist)
        self.mvarray=mv100.creat_matrix(self.mvlist)
        self.device=device
        '''
        初始化物品向量
        '''
        print("????????")
        print(self.mvarray.shape)

        self.bi=nn.Parameter(torch.full((1,self.mvarray.shape[1]),0.0)[0])
        '''
        初始化用户向量
        '''
        self.bu=nn.Parameter(torch.full((1,self.mvarray.shape[0]),0.0)[0])
        '''
        初始化物品矩阵,这里提前转制
        '''
        self.qi=nn.Parameter(torch.full((self.k,self.mvarray.shape[1]),self.U))
        '''
        初始化用户矩阵
        '''
        self.pu=nn.Parameter(torch.full((self.mvarray.shape[0],self.k),self.U))
        '''
        初始化用户评分历史矩阵
        '''
        self.Ru=self.mvarray.copy()
        self.Ru[self.Ru>=0]=1
        self.Ru[self.Ru<0]=0
        RU_width=self.mvarray.shape[1]
        sum_vector=np.ones((RU_width,1))
        # print(RU_width**0.5)
        # print(np.dot(self.Ru,sum_vector))
        # print((np.dot(self.Ru,sum_vector)/(RU_width**0.5)))
        self.Ru=nn.Parameter(torch.tensor((np.dot(self.Ru,sum_vector)/(RU_width**0.5)).tolist()))
        # sleep(100)








    def forward(self,u,i):
        '''
        这个函数我需要设计成支持list所以需要全部使用矩阵表示
        :param u:
        :param i:
        :return:
        '''
        # print('前向传播')
        # print(self.U)
        # print(self.pu[u,:].shape)
        #print(self.bi[i]+self.bu[u])\
        if self.device=='cpu':

            r_hat=self.U+self.bi[i]+self.bu[u]+torch.diag(torch.mm(self.pu[u,:]+self.Ru[u],self.qi[:,i]))

        else:
            R=self.Ru[u].cuda(0)

            # print(i)
            # print(u)
            r_hat = self.U + self.bi[i] + self.bu[u] + torch.diag(torch.mm(self.pu[u, :]+R , self.qi[:, i]))
        return r_hat