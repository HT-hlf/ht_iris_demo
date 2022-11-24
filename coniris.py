#encoding=UTF-8
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
# def tran_y(y):
#     y_ohe = np.zeros(3)
#     y_ohe[y] = 1
#     return y_ohe

def onehot(labels):
    n_sample = len(labels)
    #数据分为几类。因为编码从0开始所以要加1
    n_class = max(labels) + 1
    #建立一个batch所需要的数组，全部赋0.
    onehot_labels = np.zeros((n_sample, n_class))
    #对每一行的，对应分类赋1
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels
with open(r'.\Iris数据集\iris.csv') as f:
    f_csv=csv.reader(f)
    headers = next(f_csv)
    print(headers[1:6])
    x=[]
    label=[]
    for row in f_csv:
        # print(row)
        x1=[float(i) for i in row[1:5]]
        x.append(x1)
        if row[5]=='setosa':
            label.append(0)
        elif row[5]=='versicolor':
            label.append(1)
        else:
            label.append(2)
    X2=np.ones((150,4,4,8), dtype=float)
    print(X2.shape)
        # if row[5]=='setosa':
        #     label.append([1,0,0])
        # elif row[5]=='versicolor':
        #     label.append([0,1,0])
        # else:
        #     label.append([0,0,1])
    X1=np.array(x)
    Y1=np.array(label)
    permutation = np.random.permutation(Y1.shape[0])  # 记下第一维度的打乱顺序
    X1= X1[permutation, :]  # 按照顺序索引
    Y1 = Y1[permutation]
    print(X1[0,:])
    for i in range(150):
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
    # Y2 = np.ones((150, 3), dtype=float)
    # for i in range(150):
    #     if Y1[i]==0:
    #         Y2[i,:]=[1,0,0]
    #     elif Y1[i]==1:
    #         Y2[i,:] = [0,1,0]
    #     else:
    #         Y2[i,:] = [0,0,1]
    # Y2=onehot(Y1)
    # print(type(Y2))
    plt.imshow(X2[0, :, :, 1])
    plt.show()
    Y2 = np.zeros((150, 3), dtype=float)
    for i in range(150):
        if Y1[i]==0:
            Y2[i,0]=1
        elif Y1[i]==1:
            Y2[i,1]=1
        else :
            Y2[i,2]=1
    print(type(X2))
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


# 卷积层和池化层也是接下来要重复使用的，因此也为它们定义创建函数

# tf.nn.conv2d是TensorFlow中的2维卷积函数，参数中x是输入，W是卷积的参数，比如[5, 5, 1, 32]
# 前面两个数字代表卷积核的尺寸，第三个数字代表有多少个channel，因为我们只有灰度单色，所以是1，如果是彩色的RGB图片，这里是3
# 最后代表核的数量，也就是这个卷积层会提取多少类的特征

# Strides代表卷积模板移动的步长，都是1代表会不遗漏地划过图片的每一个点！Padding代表边界的处理方式，这里的SAME代表给
# 边界加上Padding让卷积的输出和输入保持同样SAME的尺寸
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# tf.nn.max_pool是TensorFlow中的最大池化函数，我们这里使用2*2的最大池化，即将2*2的像素块降为1*1的像素
# 最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著的特征，因为希望整体上缩小图片尺寸，因此池化层
# strides也设为横竖两个方向以2为步长。如果步长还是1，那么我们会得到一个尺寸不变的图片
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')


y_ = tf.placeholder(tf.float32, [None, 3])
x_image = tf.placeholder(tf.float32, [None, 4,4,8])

# 定义我的第一个卷积层，我们先使用前面写好的函数进行参数初始化，包括weights和bias，这里的[5, 5, 1, 32]代表卷积
# 核尺寸为5*5，1个颜色通道，32个不同的卷积核，然后使用conv2d函数进行卷积操作，并加上偏置项，接着再使用ReLU激活函数进行
# 非线性处理，最后，使用最大池化函数max_pool_2*2对卷积的输出结果进行池化操作
W_conv1 = weight_variable([2, 2, 8, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

# 第二层和第一个一样，但是卷积核变成了64
W_conv2 = weight_variable([2, 2, 16, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

# 因为前面经历了两次步长为2*2的最大池化，所以边长已经只有1/4了，图片尺寸由28*28变成了7*7
# 而第二个卷积层的卷积核数量为64，其输出的tensor尺寸即为7*7*64
# 我们使用tf.reshape函数对第二个卷积层的输出tensor进行变形，将其转成1D的向量
# 然后连接一个全连接层，隐含节点为1024，并使用ReLU激活函数
W_fc1 = weight_variable([4 *4 * 64, 32])
b_fc1 = bias_variable([32])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 *64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 防止过拟合，使用Dropout层
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 接 Softmax分类
W_fc2 = weight_variable([32, 3])
b_fc2 = bias_variable([3])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.5e-4).minimize(cross_entropy)

#这里是取y_conv中最大的标签值和y_中的最大标签值比较，如果相等就置1
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#因为是三分类问题，准确率就等于判对的除以总数，即下面
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练
tf.global_variables_initializer().run()
for i in range(9000):
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


