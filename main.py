import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class Perceptron:

    def __init__(self):

        self.n_num = [2, 10, 1]
                
        self.w1 = np.random.rand(self.n_num[0], self.n_num[1])
        self.b1 = np.random.rand(self.n_num[1], 1)
        self.w2 = np.random.rand(self.n_num[1], self.n_num[2])
        self.b2 = np.random.rand()

        self.H = np.zeros(self.n_num[1])

    def forward(self, X):
        self.H = np.dot(X, self.w1) + np.repeat(self.b1.T, X.shape[0], axis = 0)
        self.H = self.sigmoid(self.H)
        output = np.dot(self.H, self.w2) + self.b2
        return output

    def backward(self, X, Y0, Y, lr):
        shape1 = self.w1.shape
        grad1 = np.zeros(shape1)
        for i in range(shape1[0]):
            for j in range(shape1[1]):
                grad1[i][j] = np.sum(X[:, i:i+1] * self.H[:, j:j+1] * (1 - self.H[:, j:j+1]) * self.w2[j][0] * (Y0 - Y) * 2)
        grad1 /= X.shape[0]

        shape2 = self.b1.shape
        grad2 = np.zeros(shape2)
        for j in range(shape2[0]):
            grad2[j] = np.sum(self.H[:, j:j+1] * (1 - self.H[:, j:j+1]) * self.w2[j][0] * (Y0 - Y) * 2)
        grad2 /= X.shape[0]

        shape3 = self.w2.shape
        grad3 = np.zeros(shape3)
        for j in range(shape3[0]):            
            grad3[j] = np.sum(self.H[:, j:j+1] * (Y0 - Y) * 2)
        grad3 /= X.shape[0]
        grad4 = np.sum(2 * (Y0 - Y)) / X.shape[0]

        self.w1 = self.w1 - lr * grad1
        self.b1 = self.b1 - lr * grad2
        self.w2 = self.w2 - lr * grad3
        self.b2 = self.b2 - lr * grad4

    def loss(self, Y0, Y):
        return np.sum((Y0 - Y) * (Y0 - Y)) / Y0.shape[0]

    def sigmoid(self, data):
        return 1 / (1 + np.exp(-1 * data))

    def save_model(self):
        np.savez('./perceptron', w1 = self.w1, b1 = self.b1, w2 = self.w2, b2 = self.b2)

    def load_model(self):
        data = np.load('./perceptron.npz')
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']

def train(per, dataset, epoches, batch_size, lr):
    n = dataset.shape[0]
    best_loss = float("inf")
    for i in range(epoches):
        # if (i > 0 and i % 100 == 0):
            # lr /= 2
        for j in range(int(np.ceil(n / batch_size))):
            X = dataset[j*batch_size:min((j+1)*batch_size, n), 0:2]
            Y = dataset[j*batch_size:min((j+1)*batch_size, n), 2:3]
            Y0 = per.forward(X)
            E = per.loss(Y0, Y)
            # print(X, self.H, Y0, E)
            per.backward(X, Y0, Y, lr)
            # input("Press Enter to continue...")
        Yi = per.forward(dataset[:, 0:2])
        Ei = per.loss(Yi, dataset[:, 2:3])
        if (Ei < best_loss):
            best_loss = Ei
            per.save_model()
        print(i, lr, Ei)
    ax = plt.gca(projection='3d')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_zlabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    z = per.forward(dataset[:, 0:2])
    ax.plot_trisurf(dataset[:, 0], dataset[:, 1], z[:, 0], cmap='Blues', edgecolor='none')
    plt.savefig('trained.png')
    plt.show()

def main():
    per = Perceptron()

    epoches = 400
    batch_size = 64
    lr = 0.04

    n = 10201
    dataset = np.zeros((n, 3))

    for i in range(101):
        dataset[101*i:101*(i+1), 0] = -5 + 0.1 * i
        dataset[101*i:101*(i+1), 1] = np.arange(-5, 5.1, 0.1)
    dataset[:, 2] = np.sin(dataset[:, 0]) - np.cos(dataset[:, 1])
    np.random.shuffle(dataset)
    
    train(per, dataset, epoches, batch_size, lr)

if __name__=='__main__':
    main()
