import numpy as np

def batch_generator(train_data, batch_size):
    # train_data consists in the tuple (X,y)
    # In this function, we read X and y from the memory.
    # To read from disk, we would use something like :
    # pandas.read_csv('large_dataset.csv', iterator=True, chunksize=batch_size)
    i=0
    while (i+1)*batch_size<X.shape[0]:
        yield (train_data[0][i*batch_size:(i+1)*batch_size,:], train_data[1][i*batch_size:(i+1)*batch_size])
        i += 1

def relu(x, deriv=False):
    if deriv:
        return (x>0).astype(float)
    return np.maximum(0,x)

def tanh(x, deriv=False):
    if deriv:
        return 1-np.tanh(x)**2
    return np.tanh(x)


class MLP(object):

    def __init__(self, dimensions, learning_rate, learning_rate_decay=0.9, activation=relu):
        self.n_layers = len(dimensions)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.activation = activation
        self.W = [None]*self.n_layers
        self.B = [None]*self.n_layers
        for i in range(1,self.n_layers):
            weight_matrix = np.random.randn(dimensions[i-1], dimensions[i])
            self.W[i] = weight_matrix
            self.B[i] = np.random.randn(dimensions[i])
        # Deltas
        self.D = [None]*self.n_layers
        # Activations
        self.A = [None]*self.n_layers
        # Derivative of the activation
        self.Ap = [None]*self.n_layers

        self.trained = False

    def fit(self, X, y, batch_size, nb_epoch):
        assert X.shape[1] == self.W[1].shape[0]
        assert y.shape[1] == self.W[-1].shape[1]

        shuffle_indices = np.random.permutation(X.shape[0])

        loss = []
        lr = self.learning_rate
        for i in range(nb_epoch):
            for (X_, y_) in batch_generator((X[shuffle_indices,:], y[shuffle_indices]), batch_size=batch_size):
                loss.append(self.forward_propagation(X_, y_))
                self.backward_propagation(y_)
                self.update_weights()
                lr *= self.learning_rate_decay
        self.trained = True
        return loss

    def predict(self, X):
        assert self.trained
        assert X.shape[1] == self.W[1].shape[0]
        return self.forward_propagation(X, phase='predict')

    def forward_propagation(self, X, y=[], phase='training'):
        if phase=='training':
            assert len(y)!=0

        self.A[0] = X
        for i in range(1, self.n_layers):
            # Sum of weighted inputs
            Z = self.A[i-1].dot(self.W[i]) + self.B[i]
            if i < self.n_layers-1:
                # Activation for hidden layers
                self.A[i] = self.activation(Z)
                self.Ap[i] = self.activation(Z, deriv=True)
            else:
                # Linear activation for last layer
                self.A[i] = Z
                self.Ap[i] = np.ones(Z.shape)
        # Return loss if training
        if phase=='training':
            loss = np.sum(np.linalg.norm(y-self.A[-1], ord=2, axis=1))
            return loss
        elif phase=='predict':
            return self.A[-1]

    def backward_propagation(self, y):
        self.D[-1] =  (self.A[-1] - y) * self.Ap[-1]
        for i in range(1, self.n_layers-1)[::-1]:
            self.D[i] = self.D[i+1].dot(self.W[i+1].T) * self.Ap[i]

    def update_weights(self):
        for i in range(1, self.n_layers):
            self.W[i] -= self.learning_rate * self.A[i-1].T.dot(self.D[i])
            self.B[i] -= self.learning_rate * np.sum(self.D[i])
