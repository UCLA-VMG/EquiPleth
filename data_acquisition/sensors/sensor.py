#Abstract Base Class for all sensors

import abc
import time
import os
from datetime import datetime
from config import *

class Sensor(abc.ABC):

    @abc.abstractmethod
    def __init__(self, filename : str, foldername : str = "individual_sensor_test"):
        self.folder = foldername
        self.filename = filename
        self.filepath = os.path.join(config.get("mmhealth", "data_path"), foldername, filename)
        self.time_stamps = []
        self.local_time_stamps = []
        self.sensor_type = "_abstract_sensor"
        self.sensor_on = False

    #@abc.abstractmethod
    def __del__(self) -> None:
        """[Deconstructor, mainly just calls self.release_sensor()]"""
        
        self.release_sensor()
        

    @abc.abstractmethod
    def acquire(self, acquisition_time : int) -> bool:
        """[Main Logic for acquiring data of sensor_type
            sensor should already be initialized and running after __init__()
            acquire only captures and saves data]

        Args:
            acquisition_time (int)

        Returns:
            bool: [true if data acquired]
        """

        """format:

            for i in amount of data:
                capture data 'i'
                timestamp data 'i'
                save data 'i'

            self.save_timestamps()
        """
        pass

    @abc.abstractmethod
    def release_sensor(self) -> bool:
        """[Routine for releasing sensor resources and saving any remaining data
            to be called when data is released]

        Returns:
            bool: [true if done]
        """
        pass

    def record_timestamp(self) -> None:
        """[Called within self.acquire() after each data capture to time stamp data]
        """
        self.time_stamps.append(time.perf_counter())
        self.local_time_stamps.append(str(datetime.now()))

    def save_timestamps(self) -> bool:
        try:
            with open(self.filepath + ".txt", "w") as output:
                output.write('\n'.join([str(stamp) for stamp in self.time_stamps]))
            with open(self.filepath + "_local.txt", "w") as output:
                output.write('\n'.join([stamp for stamp in self.local_time_stamps]))
            return True
        except:
            print("failed time stamp saving")
            return False

    @abc.abstractmethod
    def print_stats(self):
        """[generate printout to console of the sensor's current operating specifications and any conditions]
        """
        pass