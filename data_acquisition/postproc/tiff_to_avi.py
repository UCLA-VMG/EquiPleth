from PIL import Image
import numpy as np
import os
import imageio
import cv2
import sys

sys.path.insert(0, r"add/sensors/folder/to/path")
from config import *

def tiff_to_avi(input_filepath):
    path = os.path.dirname(input_filepath)
    filename_ext = os.path.basename(input_filepath)
    filename = os.path.splitext(filename_ext)[0]

    sensor_type = filename[:2]
    if(sensor_type == "rgb"):
        sensor_type = "rgb"
    elif(sensor_type == "rgbd"):
        sensor_type = "rgbd"
    else: 
        pass

    width = config.getint(sensor_type, "width")
    height = config.getint(sensor_type, "height")
    fps = config.getint("mmhealth", "fps")

    imarray = imageio.volread(input_filepath) 
    mask = np.ma.masked_invalid(imarray)
    imarray[mask.mask] = 0.0
    imarray = cv2.normalize(src=imarray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    imarray[mask.mask] = 255.0 
    NUM_FRAMES = imarray.shape[0]

    if(NUM_FRAMES == 1):
        frame = imarray[0]
        output_filepath = os.path.join(path, filename + ".jpeg")
        imageio.imwrite(output_filepath, frame.astype(np.uint8))
    else:
        output_filepath = os.path.join(path, filename + "_avi.avi")
        imageio.mimwrite(output_filepath, imarray.astype(np.uint8), fps = fps)      
    
    print("Tiff to Avi Conversion: Sensor {} done! Shape: {}".format(sensor_type, imarray.shape) )
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_filepath = r"path/to/tiff/file"
    tiff_to_avi(input_filepath)