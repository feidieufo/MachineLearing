import numpy as np
def loadData():
    data1 = np.random.multivariate_normal([-1, -1, 1], [[1, 0, 0], [0, 1, 0], [0, 0, 0]], 1000)
    data2 = np.random.multivariate_normal([1, 1, -1], [[1, 0, 0], [0, 1, 0], [0, 0, 0]], 1000)
    data = np.concatenate([data1, data2], 0)
    return data

def nearest(data):
    np.random.shuffle(data)
    train = data[0:50, :]
    np.random.shuffle(data[50:, :])
    test = data[50:550, :]

    accuracy = 0
    for i in range(500):
        dist = train[:,0:2]-test[i, 0:2]
        predict = np.argmin(np.sum(dist**2, 1))
        accuracy += (test[i, 2]==train[predict, 2])

    accuracy /= 500
    print(accuracy)
if __name__ == '__main__':
    data = loadData()
    nearest(data)
