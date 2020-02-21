from tensorflow import keras, data, device
from timeit import timeit, repeat
from sys import argv

def main():
	# args: model batch_size
	if len(argv) != 3 and len(argv) != 4:
		print('Usage:', argv[0], 'model_file batch_size [--cpu]')
		return 1
	mod_file = argv[1]
	b_sz = int(argv[2])
	if '--cpu' in argv:
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

	print('Evaluating accuracy...')
	_, acc = model.evaluate(eval_ds, verbose=0)

	test_ds = x_ds.take(1024).batch(b_sz)
	steps = 1024 // b_sz
	print('Measuring speed...')
	test_code = ''.join(('with device(dev):\n',
	' for batch in test_ds:\n',
	'  prediction = model.predict(batch)'))
	time = timeit(test_code, number=1, globals={'device':device, 'dev':dev,
										'test_ds':test_ds, 'model':model})
	print('Metrics for model "%s", with batch size %d:' % (mod_file, b_sz))
	print('Time: %.3f s' % time)
	print('Speed: %.1f inf/s' % (steps * b_sz / time))
	print('Accuracy: %.2f' % (100 * acc), '%')
	return 0

main()

