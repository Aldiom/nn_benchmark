import tensorflow as tf
import numpy as np

route = '/'

print('Getting Mobilenet...')
mobnet = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), weights='imagenet')

netin = tf.keras.Input((224,224,3), dtype='uint8')
x = tf.cast(netin, 'float32')
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = mobnet(x)
mobnet = tf.keras.Model(inputs=netin, outputs=x)

mobnet.save(route + 'mobilenet')
print('Done')

print('Getting ResNet50...')
resnet = tf.keras.applications.ResNet50(input_shape=(224,224,3), weights='imagenet')

netin = tf.keras.Input((224,224,3), dtype='uint8')
x = tf.cast(netin, 'float32')
x = tf.nn.bias_add(x, [-123.68, -116.779, -103.939]) 

# BGR →  RGB, to avoid problems with TFLite unsupported ops 
in_weights = resnet.layers[2].get_weights()
in_weights[0] = np.flip(in_weights[0], axis=-2)
resnet.layers[2].set_weights(in_weights)

x = resnet(x)
resnet = tf.keras.Model(inputs=netin, outputs=x)

resnet.save(route + 'resnet50')
print('Done')


