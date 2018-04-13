# MNIST
This project bases on the Kaggle digital recognition competition.  
Data: From the TensorFlow’s package.  
Input: 28*28=784  
Output: 10  
Five files:  


## 1.    Autoencoder
    Key Words: Xaiver distribution


## 2.	Softmax
    Use SoftMax function to train the data.
    Key Words: GradientDescentOptimizer
    Result: 92.05%


## 3.	NN
    Use ANN to train the data.
    Key Words: epoch(5000), dropout(0.8), hidden_units(300), batch(100), ReLU
    Result: 97.99%


## 4.	Double_NN
    Use two hidden layers to train the data
    Key Words:  hidden_units(300, 300)
    Result: 98.11%


## 5.	CNN
    Use CNN to train the data
    Key Words: (conv[5,5]-pool-conv[5,5]-pool-fc[1024]-fc)
    Results: 99.19%
