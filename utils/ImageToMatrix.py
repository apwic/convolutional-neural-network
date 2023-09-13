import os
from numpy import asarray
from PIL import Image

script_dir = os.path.dirname(__file__)

def ImageToMatrix():
    img = Image.open('./test/9.jpeg')
    img_array = asarray(img)
    return img_array