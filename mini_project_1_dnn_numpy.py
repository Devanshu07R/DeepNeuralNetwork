#To implement a simple Deep Neural Network (DNN) from scratch using Numpy, starting from dataset creation to model training, from dataset creation to model training, accuracy eavaluation, and evaluation, and visualization of results.

import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0, 2, (100, 2))
y = np.logical_xor(x[:, 0], x[:,1]).astype(int).reshape(-1,1)#XOR ouput

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm')
plt.title("Binary Dataset Visualization (XOR Problem)")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.savefig("graph.jpg")
plt.show()

# To convert the dataset into the numpy array.....
x = np.array(x)
y = np.array(y)
print("Shape of X:", x.shape)
print("Shape of y:", y.shape)

# STEP-3 To define the architecture and activation functions....
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def relu(z):
    return np.maximum(0, z)
def sigmoid_derivative(a):
    return a * (1-a)
def relu_derivative(a):
    return np.where(a > 0, 1, 0)

#STEP 4: Initialize weights and biases
input_dim = 2 # no.of input features
hidden_dim = 5 # no.of neurons in hidden layers
output_dim = 1 # no. of output layers

np.random.seed(42) #for reproducibility

W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1,hidden_dim))

W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1,output_dim))

print("Initial weights and Biases initialized successfully!")

# Now lets the train the model...
lr = 0.05 #learning rate
epochs = 5000 # no.of continous training iterations

losses = []
accuracies = []

for epoch in range(epochs):
    #Forward propagation
    Z1 = np.dot(x, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Compute loss(Binary Cross Entropy)
    m = x.shape[0]
    loss = -np.mean(y * np.log(A2+1e-8)+(1-y)*np.log(1-A2+1e-8))
    losses.append(loss)
    
    #Compute accuracy
    predictions = (A2>0.5).astype(int)
    accuracy = np.mean(predictions==y)
    accuracies.append(accuracy)
    
    #Backward propagation
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis = 0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(x.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims = True) / m
    
    # to update the weights and biases
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%")

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.plot(losses, label="Loss", color="red")
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


plt.tight_layout()
plt.savefig("loss_epochs_graph.jpg")
plt.show()

#STEP 7: Final Evaluation -------------------------------------------------
print("\nFinal Training Accuracy:", round(accuracies[-1] * 100, 2), "%")

# Test on new input
test_input = np.array([[0,0], [0,1], [1,0], [1,1]])
Z1 = np.dot(test_input, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)
predictions = (A2 > 0.5).astype(int)

print("\nTest Inputs:\n", test_input)
print("Predicted Outputs:\n",predictions)