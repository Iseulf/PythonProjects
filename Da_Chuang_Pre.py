import numpy as np
import scipy.signal as ss
WINDOW = 1  # 窗口大小
FRE = 1  # 采样频率
CON_TIME = 4.5  # 手势最长持续时间4.5s


def shorttime_energy(data, window=np.ones(WINDOW)):
    # 获取数据data的短时能量
    data = np.array(data)
    data = data*data
    window = np.array(window)
    window = window*window
    return ss.convolve(data, window)


def get_possible(data):
    # 求出手势相关信号的大致范围
    data_energy = list(shorttime_energy(data))
    index = data_energy.index(max(data_energy))
    return data[index:index+CON_TIME*FRE]


def get_end_index(useful_data, pulse, start_index, window=WINDOW):
    # 求手势相关的终止点
    temp = []
    for i in range(len(useful_data)-WINDOW-start_index-1):
        foobar = useful_data[(start_index+i):(start_index+i+WINDOW)]
        temp.append(dtw_distance(foobar, pulse))
    return start_index+temp.index(min(temp))


def get_start_index(useful_data, thea=0, window=WINDOW):
    # 获取数据段data窗口大小为window,阈值为thea的最大窗口的开始表位
    # useful_data=shorttime_energy(useful_data)
    temp = []  # 保存每段数据段的比较值
    for i in range(int(len(useful_data)/window)):
        if i+window <= len(useful_data):
            t = np.array(useful_data[i:i+window])
            temp.append(np.dot(t-thea, t.reshape(t.shape[0], 1)))
        else:
            t = np.array(useful_data[i, -1])
            temp.append(np.dot(t-thea, t.reshape(t.shape[0], 1)))
    return temp.index(max(temp))


def get_pulse(front_data, second_to_number=FRE):
    # 获取脉冲轮廓的函数，front_data是只有脉冲影响的信号
    # second_to_number是秒到数组元素个数的转换率，相当于信号的收集频率
    # front_data=list(shorttime_energy(front_data))
    tp = front_data.index(max(front_data))
    front_data = np.array(front_data)
    return {"pulse_data": front_data[int(tp-0.2*second_to_number):
                                     int(tp+0.4*second_to_number)], "thea": max(front_data),
            "mean": np.mean(front_data), "std": np.std(front_data, ddof=1)}


def dtw_distance(ts_a, ts_b, d=lambda x, y: abs(x-y), mww=np.inf):
    # 计算dtw距离
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]
