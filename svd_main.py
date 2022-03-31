import time

import torch
import mv100
import svd
import matplotlib.pyplot as plt

'''
思路，使用5折交叉验证，采用官方划分，每次训练2000epoch，采用最低testloss作为当前的loss，最后手动计算平均值

'''
start_time=time.time()


epoch=1000
batch_size=512
net=svd.SVD_plus_plus(kernel_size=50,filename="/home/iiap/桌面/资料/ml-100k/u5.base")
net_test=svd.SVD_plus_plus(filename="/home/iiap/桌面/资料/ml-100k/u5.test")


optim=torch.optim.Adam(net.parameters(),lr=0.01,weight_decay=1e-6)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20, eta_min=0)
Loss=svd.RMSELoss()
MAE_loss=torch.nn.L1Loss()
net.to('cuda:0')
net_test.to('cuda:0')

x_list=[]
train_loss_list=[]
test_loss_list=[]
test_MAE_loss_list=[]

for i in range(epoch):
    train_loss=0.0
    test_loss=0.0
    test_mae_loss=0.0
    train_loss_couter=0
    test_loss_counter=0

    for xu, xi, y in net.batch_generator(batch_size):
        train_loss_couter+=1
        net.train()
        optim.zero_grad()
        y_hat = net(xu, xi)
        LOSS = Loss(y_hat, torch.tensor(y).cuda(0))
        train_loss+=LOSS
        LOSS.backward()
        optim.step()
        scheduler.step()
        # print(y_hat)
        # print(y)

    with torch.no_grad():
        test_time=time.time()
        for xu_test, xi_test, y_test in net_test.batch_generator(batch_size):

            test_loss_counter+=1
            y_hat_test = net(xu_test, xi_test)
            LOSS_test = Loss(y_hat_test, torch.tensor(y_test).cuda(0))

            MAE_LOSS=MAE_loss(y_hat_test,torch.tensor(y_test).cuda(0))

            test_mae_loss+= MAE_LOSS
            test_loss += LOSS_test
        print(f"test_time:{time.time()-test_time}s")
    '''
    写一下计算的逻辑防止出错：最后的平均loss=每一个batch的loss求和处以每一个eopch的batch数量
    '''
    print(f"epoch :{i}  train_loss :{train_loss/float(train_loss_couter)} test_loss :{test_loss/float(test_loss_counter)} mae_loss :{test_mae_loss/float(test_loss_counter)}")

    '''
    绘图阶段,每20个epoch绘制一张图片,0不画，因为loss太大
    '''
    if i%20==0 and i!=0:
        x_list.append(i)

        train_loss_list.append(float(train_loss/float(train_loss_couter)))

        test_loss_list.append(float(test_loss/float(test_loss_counter)))

        test_MAE_loss_list.append(float(test_mae_loss/float(test_loss_counter)))

        plt.figure(dpi=800)
        plt.plot(x_list,train_loss_list,label='Train_RMSE_Loss')
        plt.plot(x_list,test_loss_list,label='Test_RMSE_Loss')
        plt.plot(x_list,test_MAE_loss_list,label="Test_MAE_Loss")
        plt.title('Adam Learning Curve 5')
        plt.xlabel('Epoch Num')
        plt.ylabel('Loss')
        plt.legend();
        plt.grid(True)
        plt.savefig('test.png')


'''
寻找最低loss
'''


print(test_loss_list)
result=100000.0
result2=1000000.0
for i in test_loss_list:
    if i<result:
        result=i
for i in test_MAE_loss_list:
    if i<result2:
        result2=i

print(f"当前最低rmse loss为：{result} 最低mae loss 为{result2}")



end_time=time.time()

print(f"最终耗时:{end_time-start_time}s")