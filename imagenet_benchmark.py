from timeit import timeit, repeat
from os.path import isdir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mod_file', help='model file')
parser.add_argument('b_sz', help='batch size', type=int)
parser.add_argument('--cpu', help='force CPU usage', action='store_true')
parser.add_argument('--acc', help='measure accuracy', action='store_true')
parser.add_argument('--short', help='short measurement', action='store_true')
parser.add_argument('-n', help='number of trials', type=int, default=1)
arguments = parser.parse_args()

def main(args):   
	mod_file = args.mod_file
	tflite = '.tflite' in mod_file
	sav_mod = isdir(mod_file)
	b_sz = args.b_sz
	if args.cpu:
		print('Forcing CPU usage')
		dev = 'device:CPU:0'
	else: #TODO: chequear si hay gpu en sistema
		dev = 'device:GPU:0'

	print('Loading model...')
	if tflite:
		model = tf.lite.Interpreter(model_path = mod_file)
	elif sav_mod:
		model = tf.saved_model.load(mod_file)
	else:
		model = tf.keras.models.load_model(mod_file, compile=False)
		
	if args.acc:
		print('Evaluating accuracy...')
		import imagenet
		if tflite:
			mod_type = 'tflite'
		elif sav_mod:
			mod_type = 'saved'
		else:
			mod_type = 'keras'
		acc = eval_accuracy(model, imagenet.test_ds, mod_type, (224,224,3))	
	
	test_sz = 256 if args.short else 1024
	test_ds = tf.random.uniform((224,224,3), minval=0, maxval=255, dtype=tf.int32)
	test_ds = tf.data.Dataset.from_tensors(test_ds)
	test_ds = test_ds.repeat(test_sz).map(lambda x: tf.cast(x, tf.uint8))
	test_ds = test_ds.cache().batch(b_sz)
	steps = test_sz // b_sz
	if args.short: print('Short test selected')
	print('Measuring speed...')
	if tflite:
		in_idx = model.get_input_details()[0]['index'] 
		out_idx = model.get_output_details()[0]['index'] 
		model.resize_tensor_input(in_idx, [b_sz, 224, 224, 3])
		model.allocate_tensors()
		test_code = ''.join(('for batch in test_ds:\n',
		' batch = cast(batch, "float32")\n'
		' model.set_tensor(in_idx, batch)\n',
		' model.invoke()\n',
		' prediction = model.get_tensor(out_idx)'))
		test_vars = {'test_ds':test_ds, 'model':model, 
				'in_idx':in_idx, 'out_idx':out_idx, 'cast':tf.cast}
	elif sav_mod: 
		infer = model.signatures['serving_default']
		output = infer.structured_outputs.keys()
		output = list(output)[0]
		test_code = ''.join(('for batch in test_ds:\n',
		' prediction = infer(batch)[output]'))
		test_vars = {'test_ds':test_ds, 'infer':infer, 'output':output}
	else:
		test_code = ''.join(('with device(dev):\n',
		' for batch in test_ds:\n',
		'  prediction = model.predict(batch)'))
		test_vars = {'device':tf.device, 'dev':dev,
				'test_ds':test_ds, 'model':model}

	time = repeat(test_code, number=1, globals=test_vars, repeat=args.n)
	time = min(time)
	print('Metrics for model "%s", with batch size %d:' % (mod_file, b_sz))
	print('Time: %.3f s' % time)
	print('Speed: %.1f inf/s' % (steps * b_sz / time))
	if args.acc:
		print('Accuracy: %.2f' % (100 * acc), '%')
	return 0

def eval_accuracy(model, test_ds, mod_type, in_shape=[32,32,3]):
	test_ds = test_ds.batch(64)
	in_shape = list(in_shape)

	if mod_type == 'keras':
		model.compile(loss='sparse_categorical_crossentropy',
           	metrics=['accuracy'])
		_, acc = model.evaluate(test_ds, verbose=1)
		return acc

	if mod_type == 'tflite':
		in_idx = model.get_input_details()[0]['index'] 
		out_idx = model.get_output_details()[0]['index'] 
		model.resize_tensor_input(in_idx, [64] + in_shape)
		model.allocate_tensors()

	if mod_type == 'saved':
		test_ds = test_ds.map(lambda x, y: (tf.cast(x,'uint8'), y))
		infer = model.signatures['serving_default']
		output = infer.structured_outputs.keys()
		output = list(output)[0]

	total_corrects = 0
	total_examples = 0
	for x_batch, y_batch in test_ds:
		b_sz = y_batch.shape[0]  
		if b_sz != 64:
			padding = ((0,64-b_sz), (0,0), (0,0), (0,0))
			x_batch = tf.pad(x_batch, padding)

		if mod_type == 'tflite':	
			model.set_tensor(in_idx, x_batch)
			model.invoke()
			outputs = model.get_tensor(out_idx)
		elif mod_type == 'saved':
			outputs = infer(x_batch)[output]
			outputs = outputs.numpy()

		outputs = outputs.argmax(axis=1).reshape((64,1))
		corrects = outputs[0:b_sz] == y_batch.numpy()
		total_corrects += corrects.sum()
		total_examples += b_sz

	return total_corrects / total_examples

import tensorflow as tf 
main(arguments)
