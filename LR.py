import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv")
print("Train shape:", train.shape)

class LogisticRegressionScrtach:
    def forward(self, x):
        Z= x@ self.W+self.b 
        return Z
    
    def compute_loss(self, prob, y):
        m= len(y)
        correct_prob=prob[np.arange(m), y]
        loss= -np.mean(np.log(correct_prob+1e-12))
        return loss
    
    def softmax(self, z):
        exp_z= np.exp(z-np.max(z, axis=1, keepdims=True))
        return exp_z/np.sum(exp_z, axis=1, keepdims=True)
    
    def __init__(self, n_features, n_classes):
        self.W=np.zeros((n_features, n_classes))
        self.b=np.zeros((n_classes))

    def backward(self, x, prob, y):
        m =len(y)
        y= np.eye(self.W.shape[1])[y]
        dz = prob -y 
        dw= (x.T@dz)/m
        db =np.sum(dz, axis=0)/m
        return dw, db

    def update_parameters(self, dw, db, learning_rate):
        self.W -= learning_rate*dw
        self.b -= learning_rate*db

    def fit(self, x, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            z= self.forward(x)
            score = self.forward(x)
            prob = self.softmax(score)
            loss = self.compute_loss(prob, y)
            dw, db = self.backward(x, prob, y)
            self.update_parameters(dw, db, learning_rate)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    def predict(self, x):
        z = self.forward(x)
        prob = self.softmax(z)
        return np.argmax(prob,axis=1)


model = LogisticRegressionScrtach(
    n_features=5000,
    n_classes=4
)
if __name__ == "__main__":
    X_test = np.array([[1, 0, 1],[0, 1, 1]])
    test_model = LogisticRegressionScrtach(n_features=3, n_classes=3)

    score = test_model.forward(X_test)
    prob= test_model.softmax(score)
    
    print("Probabilities:" )
    print(prob)

    y_test= np.array([0, 2])
    loss= test_model.compute_loss(prob, y_test)
    print("Loss:", loss)
    print("row's sum:", prob.sum(axis=1))
    print("Test model score: ", score)
    print("score shape", score.shape)
    print("Weight shape:", model.W.shape)
    print("Bias shape:", model.b.shape)
    test_model.fit(X_test, y_test, learning_rate=0.1, epochs=100)
    pred = test_model.predict(X_test)
    print("Predictions:", pred)