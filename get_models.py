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

# save in Saved Model and TFLite (32 & 16 bits) formats
mobnet.save(route + 'mobilenet')
converter = tf.lite.TFLiteConverter.from_keras_model(mobnet)
with tf.io.gfile.GFile(route + 'mobilenet.tflite', 'wb') as f:
    f.write(converter.convert())
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
with tf.io.gfile.GFile(route + 'mobilenet_f16.tflite', 'wb') as f:
    f.write(converter.convert())
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

# save in Saved Model and TFLite (32 & 16 bits) formats
resnet.save(route + 'resnet50')
converter = tf.lite.TFLiteConverter.from_keras_model(resnet)
with tf.io.gfile.GFile(route + 'resnet50.tflite', 'wb') as f:
    f.write(converter.convert())
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
with tf.io.gfile.GFile(route + 'resnet50_f16.tflite', 'wb') as f:
    f.write(converter.convert())
print('Done')

print('Getting VGG16-BN...')
import torchvision.models
vgg16_torch = torchvision.models.vgg16_bn(pretrained=True)

wb_list_torch = list(vgg16_torch.parameters())
wb_list_keras = []

stage = 0
stats_idx = []
for i in range(len(wb_list_torch) - 6):
    stage = stage % 4
	# select weights in the conv2d layers and transpose them to keras dim ordering:
    if wb_list_torch[i].dim() == 4:
        w = np.transpose(wb_list_torch[i].detach().numpy(), axes=[2, 3, 1, 0])
        wb_list_keras.append(w)
    else:
        b = wb_list_torch[i].detach().numpy()
        wb_list_keras.append(b)
        if stage == 3:
          stats_idx.append(i)
    stage += 1

# add FC layers weights
for i in range(-6, 0):
  w = wb_list_torch[i].detach().numpy()
  if len(w.shape) == 2:
    w = np.transpose(w)
  wb_list_keras.append(w)

# correct fc1 weight ordering
wcorr = wb_list_keras[-6].reshape((512,49,4096))
wcorr = wcorr.transpose((1,0,2))
wcorr = wcorr.reshape((25088,4096))
wb_list_keras[-6] = wcorr

# get net first submodule
n = 0
for i in vgg16_torch.modules():
  if n == 1:
    seq = i
    break
  n += 1
# get BN layers trained stats
n = 0
bns = [1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
bn_stats = []
for mod in seq.modules():
  if (n - 1) in bns:
    mu = mod.state_dict()['running_mean']
    var = mod.state_dict()['running_var']
    bn_stats.append((mu.detach().numpy(), var.detach().numpy()))
  n += 1
# and insert them in the weight list
for i in range(len(stats_idx) - 1, -1, -1):
  wb_list_keras.insert(stats_idx[i] + 1, bn_stats[i][1])
  wb_list_keras.insert(stats_idx[i] + 1, bn_stats[i][0])


# define keras vgg16 network
netin = tf.keras.Input((224,224,3))
net = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', name='block1_conv1')(netin)
net = tf.keras.layers.BatchNormalization(name='block1_bn1', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block1_relu1')(net)
net = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', name='block1_conv2')(net)
net = tf.keras.layers.BatchNormalization(name='block1_bn2', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block1_relu2')(net)
net = tf.keras.layers.MaxPool2D(pool_size=2, name='block1_pool')(net)
net = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', name='block2_conv1')(net)
net = tf.keras.layers.BatchNormalization(name='block2_bn1', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block2_relu1')(net)
net = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', name='block2_conv2')(net)
net = tf.keras.layers.BatchNormalization(name='block2_bn2', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block2_relu2')(net)
net = tf.keras.layers.MaxPool2D(pool_size=2, name='block2_pool')(net)
net = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', name='block3_conv1')(net)
net = tf.keras.layers.BatchNormalization(name='block3_bn1', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block3_relu1')(net)
net = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', name='block3_conv2')(net)
net = tf.keras.layers.BatchNormalization(name='block3_bn2', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block3_relu2')(net)
net = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', name='block3_conv3')(net)
net = tf.keras.layers.BatchNormalization(name='block3_bn3', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block3_relu3')(net)
net = tf.keras.layers.MaxPool2D(pool_size=2, name='block3_pool')(net)
net = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', name='block4_conv1')(net)
net = tf.keras.layers.BatchNormalization(name='block4_bn1', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block4_relu1')(net)
net = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', name='block4_conv2')(net)
net = tf.keras.layers.BatchNormalization(name='block4_bn2', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block4_relu2')(net)
net = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', name='block4_conv3')(net)
net = tf.keras.layers.BatchNormalization(name='block4_bn3', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block4_relu3')(net)
net = tf.keras.layers.MaxPool2D(pool_size=2, name='block4_pool')(net)
net = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', name='block5_conv1')(net)
net = tf.keras.layers.BatchNormalization(name='block5_bn1', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block5_relu1')(net)
net = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', name='block5_conv2')(net)
net = tf.keras.layers.BatchNormalization(name='block5_bn2', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block5_relu2')(net)
net = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', name='block5_conv3')(net)
net = tf.keras.layers.BatchNormalization(name='block5_bn3', epsilon=1e-5)(net)
net = tf.keras.layers.Activation('relu', name='block5_relu3')(net)
net = tf.keras.layers.MaxPool2D(pool_size=2, name='block5_pool')(net)
net = tf.keras.layers.Flatten()(net)
net = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(net)
net = tf.keras.layers.Dropout(0.5, name='drop1')(net)
net = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(net)
net = tf.keras.layers.Dropout(0.5, name='drop2')(net)
net = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(net)

vgg16 = tf.keras.Model(inputs=netin, outputs=net, name='vgg16_bn')
vgg16.set_weights(wb_list_keras)

netin = tf.keras.Input((224,224,3), dtype='uint8')
x = tf.cast(netin, 'float32')
x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode='torch')
x = vgg16(x)
vgg16 = tf.keras.Model(inputs=netin, outputs=x)

# save in Saved Model and TFLite (32 & 16 bits) formats
vgg16.save(route + 'vgg16bn')
converter = tf.lite.TFLiteConverter.from_keras_model(vgg16)
with tf.io.gfile.GFile(route + 'vgg16bn.tflite', 'wb') as f:
    f.write(converter.convert())
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
with tf.io.gfile.GFile(route + 'vgg16bn_f16.tflite', 'wb') as f:
    f.write(converter.convert())
print('Done')
