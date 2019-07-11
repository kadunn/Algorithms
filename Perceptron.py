class Perceptron:
    def __init__(self, len_inputs, epochs,threshold=0, learning_rate=1, bias=0):
        self.w = [0 for i in range(len_inputs)]
        self.thresh = threshold
        self.epochs = epochs
        self.lr = learning_rate
        self.b = bias

    def predict(self, x):
        summation = sum( [ xi * wi for xi, wi in zip( x, self.w ) ] ) + self.b
        return 1 if summation > self.thresh else 0

    def train(self, inputs, targets):
        for _ in range(self.epochs):
            for x,y in zip(inputs,targets):
                prediction = self.predict(x)
                weight_change = [self.lr*(y-prediction)*xi for xi in x]
                adjust_weights = map(lambda i,j: i+j, weight_change, self.w)
                self.w = list(adjust_weights)
                self.b += self.lr*(y-prediction)

def main():
    perc = Perceptron(2,5)
    # The Perceptron can learn and:
    # 1  1  | 1
    # 1  0  | 0
    # 0  1  | 0
    # 0  0  | 0
    observations = [[1,1],[1,0],[0,1],[0,0]]
    labels = [1,0,0,0]
    perc.train(observations, labels)
    print('bias: ', perc.b)
    print('final weights: ', perc.w)
    print(perc.predict([1,1]))
    print(perc.predict([0,1]))
    print(perc.predict([1,0]))
    print(perc.predict([0,0]))

main()
