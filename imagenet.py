import tensorflow as tf
import os

data_path = 'ILSVRC/' 
       
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
    
labels = list(set(val_labels))
labels.sort()
label_to_index = dict(((labels[i], i) for i in range(1000)))

#box = [224, 224]

def load_image(im_path, crop_box, pre=''): # crop_box: target_h, target_w
    image = tf.io.read_file(pre + im_path)
    image = tf.image.decode_jpeg(image, channels=3) 
    im_shape = tf.cast(tf.shape(image)[-3:-1], 'float32')
    if crop_box[0] / im_shape[0] < crop_box[1] / im_shape[1]:
	    scale = crop_box[1] / im_shape[1]
    else:
        scale = crop_box[0] / im_shape[0] 
    image = tf.image.resize(image, [int(scale*im_shape[0]), int(scale*im_shape[1])])
    image = tf.image.resize_with_crop_or_pad(image, crop_box[0], crop_box[1])
    #image /= 255.0  # normalize to [0,1] range
    
    return image

def load_ds(crop_box):
	img_ds = tf.data.Dataset.from_tensor_slices(val_paths)
	label_ds = tf.data.Dataset.from_tensor_slices([label_to_index[i] for i in val_labels])
	img_ds = img_ds.map(lambda x: load_image(x, crop_box, val_datapath))

	return tf.data.Dataset.zip((img_ds, label_ds))
