from keras import layers
import tensorflow as tf
import keras
# 偏置b
fc = layers.Dense(3)
fc.build(input_shape=(2,4))
print(fc.bias)

# 权重矩阵W
fc = layers.Dense(3)
fc.build(input_shape=(2,4))
print(fc.kernel)

# 三维张量
# 自动加载 IMDB 电影评价数据集
(x_train,y_train),(x_test,y_test)=keras.datasets.imdb.load_data(num_words=10000)
# 将句子填充、截断为等长 80 个单词的句子
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=80)
print(x_train.shape)

 # 创建词向量 Embedding 层类
embedding=layers.Embedding(10000, 100)
# 将数字编码的单词转换为词向量
out = embedding(x_train)
print(out.shape)

# 交换维度
x = tf.random.normal([2,32,32,3])
x = tf.transpose(x,perm=[0,3,1,2])
print(x)