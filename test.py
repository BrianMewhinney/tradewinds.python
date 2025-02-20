import tensorflow as tf

# Check if GPU is available
print(tf.config.list_physical_devices())

# Simple test
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.matmul(a, b)
print(c)

