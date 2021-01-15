from timeit import timeit, repeat
from os.path import isdir
import subprocess 
import threading
import time as t
import socket
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mod_file', help='model file')
parser.add_argument('b_sz', help='batch size')
parser.add_argument('--cpu', help='force CPU usage', action='store_true')
parser.add_argument('--acc', help='measure accuracy', action='store_true')
parser.add_argument('--mem', help='measure RAM usage', action='store_true')
parser.add_argument('--pow_serv', help='power measuring server IP')
parser.add_argument('--input', help='input size', type=int, default=224)
parser.add_argument('-n', help='number of trials', type=int, default=1)
parser.add_argument('-s', help='measurement sample size', type=int, default=1024)
arguments = parser.parse_args()

def main(args):   
	mod_file = args.mod_file
	tflite = '.tflite' in mod_file
	sav_mod = isdir(mod_file)
	b_sizes = args.b_sz.split(',')
	if args.cpu:
		print('Forcing CPU usage')
		dev = 'device:CPU:0'
	else: #TODO: chequear si hay gpu en sistema
		dev = 'device:GPU:0'

	in_shape = (args.input, args.input, 3)
	b_sizes = list(map(int, b_sizes))

	if args.pow_serv:
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.connect((args.pow_serv, 7700)) # (host, port)
		#sock.sendall() recordar cerrar socket	

	if args.mem:
		mem_flag = threading.Event()
		syn_flag = threading.Event()
		mem_thread = threading.Thread(target=measure_ram, 
			args=(mem_flag, syn_flag, 0.2, len(b_sizes)))
		mem_thread.start()

	print('Loading model...')
	start = t.time()
	if tflite:
		model = tf.lite.Interpreter(model_path = mod_file)
	elif sav_mod:
		model = tf.saved_model.load(mod_file)
	else:
		model = tf.keras.models.load_model(mod_file, compile=False)
	stop = t.time()
	mins = (stop - start) // 60
	secs = (stop - start) % 60
	print('Loaded in %d min %.1f sec' % (mins, secs))
		
	if args.acc:
		print('Evaluating accuracy...')
		import imagenet
		if tflite:
			mod_type = 'tflite'
		elif sav_mod:
			mod_type = 'saved'
		else:
			mod_type = 'keras'
		eval_ds = imagenet.load_ds(in_shape[0:2])
		acc = eval_accuracy(model, eval_ds, mod_type, in_shape)	

	if args.s != 1024:
		print('Custom sample size:', args.s, 'samples')
	test_sz = args.s

	test_ds = None

	if tflite:
		in_idx = model.get_input_details()[0]['index'] 
		out_idx = model.get_output_details()[0]['index'] 
		test_code = ['for batch in test_ds:',
		' batch = cast(batch, "float32")',
		' model.set_tensor(in_idx, batch)',
		' model.invoke()',
		' prediction = model.get_tensor(out_idx)']
		in_type = model.get_input_details()[0]['dtype']
		if in_type == tf.uint8:
			test_code.pop(1)
		test_code = '\n'.join(test_code)
		test_vars = {'test_ds':test_ds, 'model':model, 
				'in_idx':in_idx, 'out_idx':out_idx, 'cast':tf.cast}
	elif sav_mod: 
		infer = model.signatures['serving_default']
		output = infer.structured_outputs.keys()
		output = list(output)[0]
		test_code = '\n'.join(('for batch in test_ds:',
		' prediction = infer(batch)[output]'))
		test_vars = {'test_ds':test_ds, 'infer':infer, 'output':output}
	else:
		test_code = '\n'.join(('with device(dev):',
		' for batch in test_ds:',
		'  prediction = model.predict(batch)'))
		test_vars = {'device':tf.device, 'dev':dev,
				'test_ds':test_ds, 'model':model}

	bench_ds = tf.random.uniform((test_sz,) + in_shape, minval=0, 
								maxval=255, dtype=tf.int32)
	bench_ds = tf.data.Dataset.from_tensor_slices(bench_ds)
	bench_ds = bench_ds.map(lambda x: tf.cast(x, tf.uint8))
	bench_ds = bench_ds.cache()
	N = 1 if args.pow_serv else args.n

	if args.pow_serv:
		#b_sz = 64
		#print('Calculating appropiate dataset size...')
		#while True:
		#	try:
		#		if tflite:
		#			model.resize_tensor_input(in_idx, (b_sz,) + in_shape)
		#			model.allocate_tensors()
		#		test_ds = tf.ones((b_sz,) + in_shape, tf.uint8)
		#		test_ds = tf.data.Dataset.from_tensors(test_ds)
		#		test_vars['test_ds'] = test_ds
		#		samp_time = min(repeat(test_code, number=1, globals=test_vars, repeat=2))
		#		test_sz = 2.5 * b_sz / samp_time # 2.5 s de inferencia min
		#		test_sz = max(b_sizes) * ((test_sz+max(b_sizes)-1) // max(b_sizes))
		#		test_sz = int(test_sz)
		#		break
		#	except:
		#		b_sz /= 2
		#		assert b_sz > 0
		print('Measuring idle power...')
		sock.sendall(b'trigMeas')
		idle_pow = int(sock.recv(16))

	for b_sz in b_sizes:
		test_ds = bench_ds.batch(b_sz)
		test_vars['test_ds'] = test_ds
		steps = test_sz // b_sz
		print('Measuring speed...')
		if tflite:
			model.resize_tensor_input(in_idx, (b_sz,) + in_shape)
			model.allocate_tensors()
		if args.mem:
			syn_flag.wait()
			syn_flag.clear()
			mem_flag.set()
		if args.pow_serv:
			repeat(test_code, number=1, globals=test_vars, repeat=max(1,args.n-1))
			sock.sendall(b'trigMeas')
		time = repeat(test_code, number=1, globals=test_vars, repeat=N)
		time = min(time)
		del test_ds
		print('Metrics for model "%s", with batch size %d:' % (mod_file, b_sz))
		print('Time: %.3f s' % time)
		print('Speed: %.1f inf/s' % (steps * b_sz / time))
		if args.acc:
			print('Accuracy: %.2f' % (100 * acc), '%')
		if args.pow_serv:
			power = int(sock.recv(16))
			print('Average power: %d mW (+ %d mW idle)' % (power-idle_pow, idle_pow))
		if args.mem:
			mem_flag.clear()
			#mem_thread.join()

	if args.pow_serv:
		sock.sendall(b'endMeas')
		sock.close()
	return 0

def eval_accuracy(model, test_ds, mod_type, in_shape=[224,224,3]):
	samples = 0
	for i in test_ds:
		samples += 1
	test_ds = test_ds.batch(64)
	in_shape = list(in_shape)

	if mod_type == 'keras':
		model.compile(loss='sparse_categorical_crossentropy',
           	metrics=['accuracy'])
		_, acc = model.evaluate(test_ds, verbose=1)
		return acc

	if mod_type == 'tflite':
		if model.get_input_details()[0]['dtype'] == tf.uint8:
			test_ds = test_ds.map(lambda x, y: (tf.cast(x,'uint8'), y))	
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
	bar = tf.keras.utils.Progbar(samples, interval=0.2)
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

		outputs = outputs.argmax(axis=1).reshape((64,))
		corrects = outputs[0:b_sz] == y_batch.numpy()
		total_corrects += corrects.sum()
		total_examples += b_sz
		bar.add(b_sz)

	return total_corrects / total_examples

def measure_ram(sig_in, sig_out, interval, num_tests=1):
	arch = str(subprocess.run(['uname', '-m'], stdout=subprocess.PIPE).stdout)
	if 'x86_64' in arch:
		command = ['nvidia-smi', '--query-gpu=memory.used', 
		'--format=csv,noheader,nounits']
	else:
		command = ['bash', 'mem.sh']
	initial_probe = subprocess.run(command, stdout=subprocess.PIPE).stdout
	for i in range(num_tests):
		measures = []
		sig_out.set()
		sig_in.wait()
		while sig_in.is_set():
			probe = subprocess.run(command, stdout=subprocess.PIPE).stdout
			measures.append(int(probe))
			t.sleep(interval)
		#print('Idle memory usage: %s MB' % initial_probe)
		print('Max memory usage: %d MB' % (max(measures) - int(initial_probe) - 256))
	return

import tensorflow as tf 
main(arguments)
