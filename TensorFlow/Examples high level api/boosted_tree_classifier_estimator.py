import tensorflow as tf
import os
import zipfile
from urllib import request

data_dir = r'C:\Users\Aniket\Desktop\Aniket\learning-tensorflow\High level api\dogscat'

if not os.path.isdir(data_dir):
    # Download the data zip to our data directory and extract
    fallback_url = 'http://files.fast.ai/data/dogscats.zip'
    tf.keras.utils.get_file(
        os.path.join('/tmp', os.path.basename(fallback_url)), 
        fallback_url, 
        cache_dir='/tmp',
        extract=True)

def _img_string_to_tensor(image_string, image_size=(299, 299)):
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    # Resize to expected
    image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)
    return image_resized

def make_dataset(file_pattern, image_size=(299, 299), shuffle=False, batch_size=64, num_epochs=None, buffer_size=4096):
    
    def _path_to_img(path):
        # Get the parent folder of this file to get it's class name
        label = tf.string_split([path], delimiter='/').values[-2]
        
        # Read in the image from disk
        image_string = tf.read_file(path)
        image_resized = _img_string_to_tensor(image_string, image_size)
        
        return { 'image': image_resized }, label
    
    dataset = tf.data.Dataset.list_files(file_pattern)

    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.map(_path_to_img, num_parallel_calls=os.cpu_count())
    dataset = dataset.batch(batch_size).prefetch(buffer_size)

    return dataset
