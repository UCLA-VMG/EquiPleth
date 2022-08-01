import time
import os
import subprocess

from config import *
from sensor import Sensor

class MX800_Sensor(Sensor):

    def __init__(self, filename : str, foldername : str = "individual_sensor_test"):
        super().__init__(filename=filename, foldername=foldername)

        self.sensor_type = "mx800"
        self.filepath = r"path/to/mx800_binaries"
        wait_time = 1

        self.record = subprocess.Popen([self.filepath], stdin=subprocess.PIPE) # stdout=subprocess.PIPE, stdin=subprocess.PIPE, 
        time.sleep(wait_time)
        
    def __del__(self) -> None:
        self.release_sensor()
        print("Released {} resources.".format(self.sensor_type))
        
    def acquire(self, acquisition_time : int) -> bool:
        record_start = os.write(self.record.stdin.fileno(), chr(32).encode() )
        time.sleep(acquisition_time )
        record_esc = self.record.communicate(input= chr(27).encode() )[0]
        self.release_sensor()


    def release_sensor(self) -> bool:
        pass

    def print_stats(self):
        print("_____________ mx800 Specifications _____________")


# To test code, run this file.
if __name__ == '__main__':
    time_delta = 3.9 # empirically determined

    mx800 = MX800_Sensor(filename="test_run")
    mx800.acquire(acquisition_time=10 + time_delta)
