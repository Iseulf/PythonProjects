import numpy as np


def get_value(data):
    return np.dot(data, data.reshape(data.shape[0], 1))


def get_max_index(total_data, thea=0, window=1):
    # 获取数据段data窗口大小为window,阈值为thea的最大窗口的开始表位
    temp = []  # 保存每段数据段的比较值
    for i in range(int(len(total_data)/window)):
        if i+window <= len(total_data):
            t = np.array(total_data[i:i+window])
            temp.append(np.dot(t-thea, t.reshape(t.shape[0], 1)))
        else:
            t = np.array(total_data[i, -1])
            temp.append(np.dot(t-thea, t.reshape(t.shape[0], 1)))
    return temp.index(max(temp))


def get_pulse(front_data, second_to_number):
    # 获取脉冲轮廓的函数，front_data是只有脉冲影响的信号
    # second_to_number是秒到数组元素个数的转换率，相当于信号的收集频率
    tp = front_data.index(max(front_data))
    return {"pulse_data": front_data[tp-0.2*second_to_number:tp+0.4*second_to_number], "thea": max(front_data)}


