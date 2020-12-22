import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mod_path', help='input model path')
parser.add_argument('out_path', help='output model path')
parser.add_argument('--mode', help='precision mode', default='fp32')
parser.add_argument('--build', help='build engine with ImageNet', action='store_true')
parser.add_argument('--input', help='input size', type=int, default=224)
args = parser.parse_args()

assert args.mode in trt.TrtPrecisionMode.supported_precision_modes(), 'Invalid precision mode.'

def input_fn():
	print('Importing ImageNet...')
	import imagenet
	train_ds = imagenet.load_ds(2*(args.input,)).take(192).map(lambda x,y: tf.cast(x, 'uint8'))
	train_ds = train_ds.batch(64)
	data_len = 50000 // 64
	count = 0
	print('Training:')
	for batch in train_ds:
		print('\r%d/%d' % (count, data_len), end='')
		count += 1
		yield (batch,)

params = trt.TrtConversionParams(
    rewriter_config_template=None,
    max_workspace_size_bytes=1<<30,
    precision_mode=args.mode, minimum_segment_size=3,
    is_dynamic_op=True, maximum_cached_engines=1, use_calibration=True,
    max_batch_size=64, allow_build_at_runtime=True)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=args.mod_path, conversion_params=params)
calib_input = input_fn if args.mode == 'INT8' else None
converter.convert(calib_input)
if args.build:
	converter.build(input_fn=input_fn)
converter.save(args.out_path)
