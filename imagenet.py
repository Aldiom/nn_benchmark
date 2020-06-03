import tensorflow as tf
import os

data_path = 'ILSVRC/' 
train_datapath = data_path + 'Data/CLS-LOC/train/'
train_paths = os.listdir(train_datapath)
train_paths.sort()

label_to_index = dict(((train_paths[i], i) for i in range(1000)))
img_paths = []

#for label in train_paths:
#    for example in os.listdir(train_datapath + label):
#        img_paths.append(label + example)
        
		        
val_datapath = data_path + 'Data/CLS-LOC/val/'
val_paths = sorted(os.listdir(val_datapath))
val_labels = []
				    
for file in sorted(os.listdir(data_path + 'Annotations/CLS-LOC/val/')): 
	file = open(data_path + 'Annotations/CLS-LOC/val/' + file)
	for line in file:
		if '<name>' in line:
			line = line.strip()
			label = line[6:-7] # remove "<name> ... </name>"
			val_labels.append(label)
			break
	file.close()

box = [224, 224]

def preprocess_image(image, crop_box): # crop_box: target_h, target_w
    image = tf.image.decode_jpeg(image, channels=3) 
    im_shape = tf.cast(tf.shape(image)[-3:-1], 'float32')
    image = tf.image.resize(image, [int(0.875*im_shape[0]), int(0.875*im_shape[1])])
    image = tf.image.resize_with_crop_or_pad(image, crop_box[0], crop_box[1])
    image /= 255.0  # normalize to [0,1] range
    
    return image

def load_and_preprocess_image(path, crop_box, pre=''):
    image = tf.io.read_file(pre + path)
    return preprocess_image(image, crop_box)

img_ds = tf.data.Dataset.from_tensor_slices(val_paths)
label_ds = tf.data.Dataset.from_tensor_slices([label_to_index[i] for i in val_labels])

img_ds = img_ds.map(lambda x: load_and_preprocess_image(x, box, val_datapath))

test_ds = tf.data.Dataset.zip((img_ds, label_ds))
test_ds = test_ds.cache()
