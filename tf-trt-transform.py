import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mod_path', help='input model path')
parser.add_argument('out_path', help='output model path')
args = parser.parse_args()

def input_fn():
	print('Importing ImageNet...')
	import imagenet
	train_ds = imagenet.img_ds.map(lambda x: tf.cast(x, 'uint8'))
	train_ds = train_ds.batch(64)
	data_len = 50000 // 64
	count = 0
	print('Training:')
	for batch in train_ds:
		print('\r%d/%d' % (count, data_len), end='')
		count += 1
		yield (batch,)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=args.mod_path)
converter.convert()
converter.build(input_fn=input_fn)
converter.save(args.out_path)
