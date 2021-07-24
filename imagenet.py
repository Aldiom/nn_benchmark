import tensorflow as tf
import os

def prep_data(data_path='ILSVRC/', mode='val'):
    # mode: val or train
    assert mode in ('val', 'train')
    mode += '/'
    val_datapath = data_path + 'Data/CLS-LOC/' + mode
    if mode == 'train/':
        val_paths = []
        for i in sorted(os.listdir(val_datapath)):
            class_path = data_path + 'Data/CLS-LOC/' + mode + i
            for j in sorted(os.listdir(class_path)):
                val_paths.append(i + '/' + j)
    else:
        val_paths = sorted(os.listdir(val_datapath))
    val_labels = []
  
    for path in sorted(os.listdir(data_path + 'Annotations/CLS-LOC/' + mode)):
        if '.xml' in path:
            file = open(data_path + 'Annotations/CLS-LOC/' + mode + path)
            for line in file:
                if '<name>' in line:
                    line = line.strip()
                    label = line[6:-7] # remove "<name> ... </name>"
                    val_labels.append(label)
                    break
            file.close()
        else:
            for file in range(len(os.listdir(data_path + 'Annotations/CLS-LOC/' + mode + path))):
                val_labels.append(path)

    labels = list(set(val_labels))
    labels.sort()
    label_to_index = dict(((labels[i], i) for i in range(1000)))
    
    return val_paths, val_datapath, val_labels, label_to_index

#box = [224, 224]

def load_image(im_path, crop_box, pre=''): # crop_box: target_h, target_w
    image = tf.io.read_file(pre + im_path)
    image = tf.image.decode_jpeg(image, channels=3) 
    im_shape = tf.cast(tf.shape(image)[-3:-1], 'float32')
    if im_shape[0] < im_shape[1]:
        scale = 256 / im_shape[0]
    else:
        scale = 256 / im_shape[1] 
    image = tf.image.resize(image, [int(scale*im_shape[0]), int(scale*im_shape[1])])
    image = tf.image.resize_with_crop_or_pad(image, crop_box[0], crop_box[1])
    #image /= 255.0  # normalize to [0,1] range
    
    return image

def load_ds(crop_box, mode='val', data_path='ILSVRC/'):
    val_paths, val_datapath, val_labels, label_to_index = prep_data(data_path, mode)
    img_ds = tf.data.Dataset.from_tensor_slices(val_paths)
    label_ds = tf.data.Dataset.from_tensor_slices([label_to_index[i] for i in val_labels])
    if mode == 'val':
        img_ds = img_ds.map(lambda x: load_image(x, crop_box, val_datapath), 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return tf.data.Dataset.zip((img_ds, label_ds))
