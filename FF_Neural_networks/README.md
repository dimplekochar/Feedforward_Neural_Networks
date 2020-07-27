# Feedforward_Neural_Networks
https://drive.google.com/open?id=1iehr0YSVPKCTe3DgVGsC_ogKdiFlr_x4 : for train_class1.csv
https://drive.google.com/open?id=1rrHwbxQlSSAZii01J-Nm4aQlsQ65Mc7Y : for train_reg2.csv


Training and evaluating feedforward neural networks is the aim. 
First work- Classification task where I have predicted the release decade of a song from a set of timbre-based audio features extracted from the song. This consists of a total of 9 labels marking decades between 1922 to 2011. This neural network was designed to solve a classification task by optimizing cross-entropy loss. 
Second work- Instead of predicting the decade in which the song is released, we could pose the problem as a regression task where we directly try to predict the year of release (between 1922 to 2011). So, I changed the neural network such that it is now minimizing mean squared error. (can refer to the commented code) -results.txt

TenserFlow library is utilized.
Used the "Adam" optimizer to optimize cross-entropy loss and ReLU activation function with no regularization. -results_A.txt
Also, tried out SGD optimization. -results_B.txt
Also, added dropout to the hidden layer. 
Tried three different dropout keep probabilities (i.e. the probability of retaining a neuron). -results_C.txt 
