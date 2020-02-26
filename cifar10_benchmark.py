from tensorflow import keras, data, device
from timeit import timeit, repeat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mod_file', help='model file')
parser.add_argument('b_sz', help='batch size', type=int)
parser.add_argument('--cpu', help='force CPU usage', action='store_true')
parser.add_argument('--acc', help='measure accuracy', action='store_true')
parser.add_argument('-n', help='number of trials', type=int, default=1)
arguments = parser.parse_args()

def main(args):
	mod_file = args.mod_file
	b_sz = args.b_sz
	if args.cpu:
		print('Forcing CPU usage')
		dev = 'device:CPU:0'
	else:
		dev = 'device:GPU:0'

	model = keras.models.load_model(mod_file, compile=False)

	(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()

	x_ds = data.Dataset.from_tensor_slices(x_test.astype('float32'))
	y_ds = data.Dataset.from_tensor_slices(y_test.astype('float32'))
	eval_ds = data.Dataset.zip((x_ds, y_ds)).batch(64)

	model.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	
	if args.acc:
		print('Evaluating accuracy...')
		_, acc = model.evaluate(eval_ds, verbose=0)

	test_ds = x_ds.take(1024).batch(b_sz)
	steps = 1024 // b_sz
	print('Measuring speed...')
	test_code = ''.join(('with device(dev):\n',
	' for batch in test_ds:\n',
	'  prediction = model.predict(batch)'))
	time = repeat(test_code, number=1, globals={'device':device, 'dev':dev,
				'test_ds':test_ds, 'model':model}, repeat=args.n)
	time = min(time)
	print('Metrics for model "%s", with batch size %d:' % (mod_file, b_sz))
	print('Time: %.3f s' % time)
	print('Speed: %.1f inf/s' % (steps * b_sz / time))
	if args.acc:
		print('Accuracy: %.2f' % (100 * acc), '%')
	return 0

main(arguments)
