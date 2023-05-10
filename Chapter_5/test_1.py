import unittest
import tensorflow as tf
import keras
class MyTestCase(unittest.TestCase):
    def test_concat(self):
        # 模拟成绩册 A
        a = tf.random.normal([4, 35, 8])
        # 模拟成绩册 B
        b = tf.random.normal([6, 35, 8])
        # 拼接合并成绩册
        c = tf.concat([a, b], axis=0)
        print(c)

    def test_stack(self):
        a = tf.random.normal([35, 8])
        b = tf.random.normal([35, 8])
        # 堆叠合并为 2 个班级，班级维度插入在最前
        c = tf.stack([a, b], axis=0)
        print(c)

    def test_split(self):
        x = tf.random.normal([10, 35, 8])
        # 等长切割为 10 份
        result = tf.split(x, num_or_size_splits=10, axis=0)
        # 返回的列表为 10 个张量的列表
        print(len(result))

    def test_reduce(self):
        # 模拟网络预测输出
        out = tf.random.normal([4,10])
        y = tf.constant([1,2,2,8])
        # one-hot 编码
        y = tf.one_hot(y, depth=10)
        # 计算每个样本的误差
        loss = keras.losses.mse(y, out)
        # 平均误差，在样本数维度上取均值
        loss = tf.reduce_mean(loss)
        # 打印
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(loss))

    def test_argmax(self):
        out = tf.random.normal([2, 10])
        # 通过 softmax 函数转换为概率值
        out = tf.nn.softmax(out, axis=1)
        # 选取概率最大的索引下标
        pred = tf.argmax(out, axis=1)
        # 打印
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(pred))

    def test_compare(self):
        out = tf.random.normal([100,10])
        # 将输出转换为概率
        out = tf.nn.softmax(out, axis=1)
        # 拿到100个样本分别概率最大的类别
        pred = tf.argmax(out, axis=1)
        # 模拟真实标签
        y = tf.random.uniform([100],dtype=tf.int64,maxval=10)
        out = tf.equal(pred,y)
        out = tf.cast(out, dtype=tf.float32)
        # 预测正确的个数
        correct = tf.reduce_sum(out)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Accuracy: %.2f%%" % (sess.run(correct/100)))

    def test_pad(self):
        # 第一个句子.
        a = tf.constant([1, 2, 3, 4, 5, 6])
        # 第二个句子
        b = tf.constant([7, 8, 1, 6])
        # 句子末尾填充 2 个 0
        b = tf.pad(b, [[0, 2]])

        c = tf.stack([a,b], axis=0)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(c))

    def test_tile(self):
        x = tf.random.normal([4,32,32,1])
        # 复制模式[2,3,3,1]，即数量复制一份，长宽复制两份，通道不复制
        out = tf.tile(x, [2,3,3,1])
        print(out)


if __name__ == '__main__':
    unittest.main()
