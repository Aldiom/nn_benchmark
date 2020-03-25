import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_saved_model_dir = 'allconv_cifar10'
output_saved_model_dir = 'allconv_cifar10_trt'
def input_fn():
	(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
	x_train = tf.data.Dataset.from_tensor_slices(x_train.astype('float32'))
	x_train = x_train.batch(16)
	for batch in x_train:
		yield (batch,)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
converter.convert()
converter.build(input_fn=input_fn)
converter.save(output_saved_model_dir)
