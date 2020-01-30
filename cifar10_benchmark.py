from tensorflow import keras, data, device
from time import time
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

	(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

	x_ds = data.Dataset.from_tensor_slices(x_test.astype('float32'))
	y_ds = data.Dataset.from_tensor_slices(y_test.astype('float32'))

	model.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

	print('Evaluating accuracy...')
	_, acc = model.evaluate(x_test, y_test, verbose=0)

	test_ds = x_ds.take(1024).batch(b_sz)
	steps = 1024 // b_sz
	print('Measuring speed...')
	with device(dev): 
		start = time()
		for batch in test_ds:
			    prediction = model.predict(batch)
		stop = time()
	print('Metrics for model "%s", with batch size %d:' % (mod_file, b_sz))
	print('Time: %.3f s' % (stop - start))
	print('Speed: %.1f inf/s' % (steps * b_sz / (stop-start)))
	print('Accuracy: %.2f' % (100 * acc), '%')
	return 0

main()

