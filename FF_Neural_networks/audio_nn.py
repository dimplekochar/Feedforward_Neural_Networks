import numpy as np
import pandas as pd        
import tensorflow as tf

df_train = pd.read_csv('train_class.csv')
df_dev = pd.read_csv('dev_class.csv')

y_train = df_train['label'].as_matrix()
df_train = df_train.drop('label', axis=1)

y_dev = df_dev['label'].as_matrix()
df_dev = df_dev.drop('label', axis=1)

features=['TimbreAvg1', 'TimbreAvg2', 'TimbreAvg3', 'TimbreAvg4', 'TimbreAvg5', 'TimbreAvg6', 'TimbreAvg7', 'TimbreAvg8', 'TimbreAvg9', 'TimbreAvg10', 'TimbreAvg11', 'TimbreAvg12', 'TimbreCovariance1', 'TimbreCovariance2', 'TimbreCovariance3', 'TimbreCovariance4', 'TimbreCovariance5', 'TimbreCovariance6', 'TimbreCovariance7', 'TimbreCovariance8', 'TimbreCovariance9', 'TimbreCovariance10', 'TimbreCovariance11', 'TimbreCovariance12', 'TimbreCovariance13', 'TimbreCovariance14', 'TimbreCovariance15', 'TimbreCovariance16', 'TimbreCovariance17', 'TimbreCovariance18', 'TimbreCovariance19', 'TimbreCovariance20', 'TimbreCovariance21', 'TimbreCovariance22', 'TimbreCovariance23', 'TimbreCovariance24', 'TimbreCovariance25', 'TimbreCovariance26', 'TimbreCovariance27', 'TimbreCovariance28', 'TimbreCovariance29', 'TimbreCovariance30', 'TimbreCovariance31', 'TimbreCovariance32', 'TimbreCovariance33', 'TimbreCovariance34', 'TimbreCovariance35', 'TimbreCovariance36', 'TimbreCovariance37', 'TimbreCovariance38', 'TimbreCovariance39', 'TimbreCovariance40', 'TimbreCovariance41', 'TimbreCovariance42', 'TimbreCovariance43', 'TimbreCovariance44', 'TimbreCovariance45', 'TimbreCovariance46', 'TimbreCovariance47', 'TimbreCovariance48', 'TimbreCovariance49', 'TimbreCovariance50', 'TimbreCovariance51', 'TimbreCovariance52', 'TimbreCovariance53', 'TimbreCovariance54', 'TimbreCovariance55', 'TimbreCovariance56', 'TimbreCovariance57', 'TimbreCovariance58', 'TimbreCovariance59', 'TimbreCovariance60', 'TimbreCovariance61', 'TimbreCovariance62', 'TimbreCovariance63', 'TimbreCovariance64', 'TimbreCovariance65', 'TimbreCovariance66', 'TimbreCovariance67', 'TimbreCovariance68', 'TimbreCovariance69', 'TimbreCovariance70', 'TimbreCovariance71', 'TimbreCovariance72', 'TimbreCovariance73', 'TimbreCovariance74', 'TimbreCovariance75', 'TimbreCovariance76', 'TimbreCovariance77', 'TimbreCovariance78']

# Subtracting mean from and dividing by standard deviation
#for s in features:
#	mean_f = df_train[s].mean()
#	std_f = df_train[s].std()
#	df_train[s] = (df_train[s] - mean_f) / std_f

#for s in features:
#	mean_f = df_dev[s].mean()
#	std_f = df_dev[s].std()
#	df_dev[s] = (df_dev[s] - mean_f) / std_f

X_train = df_train.as_matrix()
X_dev = df_dev.as_matrix()

labels_train = (np.arange(9) == y_train[:,None]).astype(np.float32)
labels_dev = (np.arange(9) == y_dev[:,None]).astype(np.float32)

inputs = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='inputs')
label = tf.placeholder(tf.float32, shape=(None, 9), name='labels')

# Hidden layer 1
hid1_size = 100
w1 = tf.Variable(tf.random_normal([hid1_size, X_train.shape[1]], stddev=0.01), name='w1')
b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')
y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=0.8) #keep_prob=1 for part A as no dropout

# Hidden layer 2
#hid2_size = 256
#w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size], stddev=0.01), name='w2')
#b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')
#y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=1) #keep_prob=1 for part B as no dropout

# Output layer
#wo = tf.Variable(tf.random_normal([9, hid2_size], stddev=0.01), name='wo')
#bo = tf.Variable(tf.random_normal([9, 1]), name='bo')
#yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))

# Output layer
wo = tf.Variable(tf.random_normal([9, hid1_size], stddev=0.01), name='wo')
bo = tf.Variable(tf.random_normal([9, 1]), name='bo')
yo = tf.transpose(tf.add(tf.matmul(wo, y1), bo))

# Cross Entropy Loss function and Adam optimizer
lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels=label))
#loss = tf.reduce_mean(tf.squared_difference(yo, label))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Prediction
pred = tf.nn.softmax(yo)
pred_label = tf.argmax(pred, 1)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Referred Google for all this on how to use the Tenserflow library
# Next 6 lines are not my code
# Create operation which will initialize all variables
init = tf.global_variables_initializer()

# Configure GPU not to use all memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Start a new tensorflow session and initialize variables
sess = tf.InteractiveSession(config=config)
sess.run(init)

#Number of iterations
for epoch in range(100):

        avg_cost = 0.0
        batch_size=500
        
        total_batch = int(df_train.shape[0]/batch_size)
        for i in range(total_batch):
           
            _, c = sess.run([optimizer, loss], feed_dict={lr:0.001, inputs: X_train[i, None], label: labels_train[i, None]})
            avg_cost += c/total_batch   

acc_train = accuracy.eval(feed_dict={inputs: X_train, label: labels_train})
print("Train accuracy: {:3.2f}%".format(acc_train*100.0))

acc_dev = accuracy.eval(feed_dict={inputs: X_dev, label: labels_dev})
print("Dev accuracy:  {:3.2f}%".format(acc_dev*100.0))

df_test = pd.read_csv('test-class.csv')

df_test = df_test.drop('ids', axis=1)

#for s in features:
#	mean_f = df_test[s].mean()
#	std_f = df_test[s].std()
#	df_test[s] = (df_test[s] - mean_f) / std_f

X_test = df_test.as_matrix()

# Prediction
for i in range(X_test.shape[0]):
    df_test.loc[i, 'label'] = sess.run(pred_label, feed_dict={inputs: X_test[i, None]}).squeeze()
sess.close()
df_test1 = pd.read_csv('test-class.csv')
output = pd.DataFrame()
output['ids'] = df_test1['ids']
output['label'] = df_test['label'].astype(int)
output.to_csv('./submission.csv', index=False)
output.head()







