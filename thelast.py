import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

with open(r'.\Iris数据集\iris.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    print(headers[1:6])
    x = []
    label = []
    for row in f_csv:
        print(row)
        x1 = [float(i) for i in row[1:5]]
        x.append(x1)
        if row[5] == 'setosa':
            label.append(0)
        elif row[5] == 'versicolor':
            label.append(1)
        else:
            label.append(2)
    X1 = np.array(x)
    Y1 = np.array(label)
    permutation = np.random.permutation(Y1.shape[0])  # 记下第一维度的打乱顺序
    X1 = X1[permutation, :]  # 按照顺序索引
    Y1 = Y1[permutation]

#将一维数据转化成二维八通道数据以进行卷积
def change1to2(X1,num):
    X2=np.ones((num,4,4,8), dtype=float)
    for i in range(num):
            for j in range(3):
                for a in range(4):
                    for b in range(4):
                        # print(' ')
                        if j==0:
                            X2[i,a,b,j]=(X1[i,a]+X1[i,b])/2
                        elif j==1:
                            X2[i,a,b,j]=max([X1[i,a],X1[i,b]])
                        elif j==2:
                            X2[i, a, b, j] =min([X1[i,a],X1[i,b]])
                        elif j==2:
                            X2[i, a, b, j] =abs(2*X1[i,a]-X2[i,b])
                        elif j==2:
                            X2[i,a,b,j] =(((X1[i,a])**2+(X1[i,b])**2)/2)**0.5
                        elif j==2:
                            X2[i,a,b,j] =(((X1[i,a])**5+(X1[i,b])**5)/2)**1/5
                        elif j==2:
                            X2[i,a,b,j] =(((X1[i,a])**10+(X1[i,b])**10)/2)**1/10
                        else:
                            X2[i,a,b,j] =(((X1[i,a])**100+(X1[i,b])**100)/2)**0.01
    return X2
X2=change1to2(X1,150)
print('转换前的一个数据:',X1[0,:])
print('转换后的一个数据:',X2[0,:,:,:])

#可以看一下二维一个数据每一个通道的图像（虽然没什么实际意义）
for i in range(3):
    plt.imshow(X2[9, :, :, i])
    plt.show()

#将数据的label进行one-hot 编码
Y2 = np.zeros((150, 3), dtype=float)
for i in range(150):
    if Y1[i]==0:
        Y2[i,0]=1
    elif Y1[i]==1:
        Y2[i,1]=1
    else :
        Y2[i,2]=1

#题目里的意思应该是不分训练集和测试集，但是这样很可能过拟合，而且我就算分了也能跑到100%的准确率，所以无伤大雅
X2_train=X2[:140,:,:,:]
Y2_train=Y2[:140,:]
X2_test=X2[140:,:,:,:]
Y2_test=Y2[140:,:]

sess = tf.InteractiveSession()
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
y_ = tf.placeholder(tf.float32, [None, 3])
x_image = tf.placeholder(tf.float32, [None, 4,4,8])

#第一个卷积层+relu+池化
W_conv1 = weight_variable([2, 2, 8, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)
#第二个卷积层+relu+池化
W_conv2 = weight_variable([2, 2, 16, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)
#全连接+relu
W_fc1 = weight_variable([4 *4 * 64, 32])
b_fc1 = bias_variable([32])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 *64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#全连接+softmax
W_fc2 = weight_variable([32, 3])
b_fc2 = bias_variable([3])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
#使用Adam优化器
train_step = tf.train.AdamOptimizer(0.5e-4).minimize(cross_entropy)
#这里是取y_conv中最大的标签值和y_中的最大标签值比较，如果相等就置1
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#因为是三分类问题，准确率就等于判对的除以总数，即下面
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predictxl=tf.argmax(y_conv,1)

tf.global_variables_initializer().run()
for i in range(7000):
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x_image: X2_train, y_: Y2_train})
        print("step %d, training accuracy %g" % (i, train_accuracy),end='  ')
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x_image: X2_test, y_: Y2_test
        }))
    train_step.run(feed_dict={x_image: X2_train, y_: Y2_train})

print("the last train accuracy %g" % accuracy.eval(feed_dict={
    x_image: X2_train, y_: Y2_train
}))
print("the last test accuracy %g" % accuracy.eval(feed_dict={
    x_image: X2_test, y_: Y2_test
}))

def predicttoword(predict):
    if predict==0:
        print('Species:setosa')
    elif predict==1:
        print('Species:versicolor')
    else:
        print('Species:virginica')
list=input('请依次输入Sepal.Length, Sepal.Width, Petal.Length, Petal.Width（用空格隔开）').split()
predict_X=(np.array([float(i) for i in list])).reshape(-1,4)
predict_X2=change1to2(predict_X,1)
predict= predictxl.eval(feed_dict={x_image: predict_X2})
predicttoword(predict)