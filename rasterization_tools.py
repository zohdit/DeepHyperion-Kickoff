import cairo
import numpy as np
from gi import require_version
import gi
from tensorflow import keras
#from utils import reshape
from properties import IMG_SIZE

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg

def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v

def rasterize_in_memory(xml_desc):
    img = cairo.ImageSurface(cairo.FORMAT_A8, 28, 28)
    ctx = cairo.Context(img)
    handle = Rsvg.Handle.new_from_data(xml_desc)
    handle.render_cairo(ctx)
    buf = img.get_data()
    img_array = np.ndarray(shape=(28, 28),
                           dtype=np.uint8,
                           buffer=buf)

    img_array = reshape(img_array)
    
    return img_array

def convert_xml_to_image(xml_desc):
    img = cairo.ImageSurface(cairo.FORMAT_A8, 28, 28)
    ctx = cairo.Context(img)
    handle = Rsvg.Handle.new_from_data(xml_desc)
    handle.render_cairo(ctx)
    buf = img.get_data()
    img_array = np.ndarray(shape=(28, 28),
                           dtype=np.uint8,
                           buffer=buf)
 
    return img_array
