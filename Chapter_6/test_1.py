import unittest

import keras.losses
import tensorflow as tf
class MyTestCase(unittest.TestCase):
    def test_softmax(self):
        out = tf.random.normal([2,10])
        y = tf.constant([1,3])
        y_onehot = tf.one_hot(y,depth=10)
        # 为True代表未经过softmax函数
        # 调用softmax函数再计算交叉熵损失
        loss = keras.losses.categorical_crossentropy(y_onehot, out, from_logits=True)
        # 计算平均损失
        loss = tf.reduce_mean(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(loss))



if __name__ == '__main__':
    unittest.main()
