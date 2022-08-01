# rgbd sensor

import pyzed.sl as sl
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from pathlib import Path
from scipy.io import savemat
import sys
import imageio

from config import *
from sensor import Sensor

class RGBD_Sensor(Sensor):

    def __init__(self, filename : str, foldername : str = "individual_sensor_test"):
        super().__init__(filename=filename, foldername=foldername)

        self.sensor_type = "rgbd_camera"
        self.fps     = config.getint("mmhealth", "fps")
        self.width   = config.getint("rgbd", "width") 
        self.height  = config.getint("rgbd", "height") 
        self.channels = config.getint("rgbd", "channels") #not settingg
        self.compression = config.getint("rgbd", "compression")
        self.calibrate_mode = config.getint("mmhealth", "calibration_mode") 
        self.calibrate_format = ".png"
        
        # Create a Camera object
        self.zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        if (self.width == 1920 and self.height == 1080 ):
            self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        else:
            self.init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
        self.init_params.camera_fps = self.fps  # Set fps at 30
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Capture 50 frames and stop
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.width = self.zed.get_camera_information().camera_resolution.width
        self.height = self.zed.get_camera_information().camera_resolution.height
        self.channels = 3
        self.format = ".tiff"

        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

    def __del__(self) -> None:
        self.release_sensor()
        print("Released {} resources.".format(self.sensor_type))
        # print(self.filepath)
        
    def acquire(self, acquisition_time : int) -> bool:
        if (self.calibrate_mode == 1):
            run = True
            while( run == True):
                # Grab an image, a RuntimeParameters object must be given to grab()
                if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # A new image is available if grab() returns SUCCESS
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH) #TODO
                    im_arr = self.image.get_data()[:, :, :3]
                    im_arr = im_arr[..., ::-1]
                    frame = cv2.resize(im_arr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    cv2.imshow('Input', frame)

                    key = cv2.waitKey(1)
                    if key == ord('s'):
                        start_num = 1
                        while(os.path.exists(self.filepath + "_" + str(start_num) + self.calibrate_format)):
                            start_num += 1
                        imageio.imwrite(self.filepath + "_" + str(start_num) + self.calibrate_format, im_arr)
                    elif key == ord('q'):
                        run = False
                        cv2.destroyAllWindows()
                        break
        
        else:
            NUM_FRAMES = self.init_params.camera_fps*acquisition_time  # number of images to capture

            im_frames = np.empty((NUM_FRAMES, self.height, self.width, self.channels), np.dtype('uint8'))
            depth_frames = np.empty((NUM_FRAMES, self.height, self.width), np.dtype('float32')) # TODO

            for i in range(NUM_FRAMES):
                # Grab an image, a RuntimeParameters object must be given to grab()
                if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # A new image is available if grab() returns SUCCESS
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                    self.record_timestamp()

                    im_arr = self.image.get_data()[:, :, :3]
                    im_arr = im_arr[..., ::-1]
                    depth_arr = self.depth.get_data()
                    im_frames[i] = im_arr
                    depth_frames[i] = depth_arr
            
            imageio.mimwrite(self.filepath + "_rgb" + self.format, im_frames, bigtiff=True)
            imageio.mimwrite(self.filepath + "_depth" + self.format, depth_frames, bigtiff=True)

            self.save_timestamps()
            self.time_stamps = []


    def release_sensor(self) -> bool:
        #Close the camera
        self.zed.close()

    def print_stats(self):
        print("_____________ RGBD Camera Specifications _____________")
        print("FPS = {} f/s".format(self.init_params.camera_fps))
        print("Resolution = {} x {}".format(self.width, self.height))
        print("Comression - ", self.compression)

# #To test code, run this file.
if __name__ == '__main__':

    rgbd_cam = RGBD_Sensor(filename="rgbd")
    rgbd_cam.acquire(acquisition_time=5)
    rgbd_cam.print_stats()