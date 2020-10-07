#利用梯度法求函数最小值
import numpy as np 
import matplotlib.pylab as plt
def function(x):
    return x[0]**2+x[1]**2
def numerical_gradent(f,x):
    #求函数f(x)在x处的梯度
    h=1e-4
    grad=np.zeros_like(x)
    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)
        x[idx]=tmp_val-h
        fxh2=f(x)
        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
    return grad
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    #迭代step_num次求在init_x附近的最值点
    x=init_x
    for i in range(step_num):
        grad=numerical_gradent(f,x)
        x-=lr*grad
        plt.scatter(x[0],x[1])
    plt.xlim((-3,3))
    plt.ylim((-4,4))
    plt.show()
    return x
init_x=np.array([-3.0,4.0])
print(gradient_descent(function,init_x,0.1,100))