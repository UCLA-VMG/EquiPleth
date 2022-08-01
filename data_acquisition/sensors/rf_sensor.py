# RF Sensor

import numpy as np
import sys
import subprocess
import pickle
import glob
import time
import socket
import struct
import datetime
from datetime import datetime as dt

# from rf_UDP.listener import *
from config import *

from sensor import Sensor

class RF_Sensor(Sensor):

    def __init__(self, filename : str, foldername : str = "individual_sensor_test", sensor_on = False, static_ip='192.168.33.30', adc_ip='192.168.33.180',
                 data_port=4098, config_port=4096):
        super().__init__(filename=filename, foldername=foldername)

        self.sensor_type = "RF"
        self.format = ".pkl"
        self.MAX_PACKET_SIZE = 4096
        self.BYTES_IN_PACKET = 1456

        # Initialize capture through running windows executable that runs with mmwave_studio
        if(not sensor_on):
            subprocess.Popen(r'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\RunTime\run_mmwavestudio.cmd',
                                cwd=r'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\RunTime')
        # IMPORTANT, wait 120 seconds in all other processes, setup takes 55 seconds at least.

        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)
        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)

    def __del__(self) -> None:
        self.release_sensor()
        print("Released {} resources.".format(self.sensor_type))
        
    def acquire(self, acquisition_time : int) -> bool:
        all_data = self.read(acquisition_time)

        print("Start time: ", all_data[3])
        print("End time: ", all_data[4])

        print("self.filepath + self.format: {}".format(self.filepath + self.format) )
        with open(self.filepath + self.format, 'wb') as f:
            pickle.dump(all_data, f)

        print("Storing collected files in ", self.filepath + self.format)

    def release_sensor(self) -> bool:
        pass

    def print_stats(self):
        print("_____________ RF Specifications _____________")
        print("TODO print Config File")

    def write_to_file(self, all_data, packet_num_all, byte_count_all, num_chunks):
        to_store = (all_data, packet_num_all, byte_count_all)

        d = int(dt.timestamp(dt.utcnow())*1e6)
        with open(self.filepath + '_' + str(d) +'.pkl', 'wb') as f:
            pickle.dump(to_store, f)

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(self.MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        return packet_num, byte_count, packet_data

    def close(self):
        """Closes the sockets that are used for receiving and sending data

        Returns:
            None

        """
        self.data_socket.close()
        self.config_socket.close()

    def read(self, acquisition_time, timeout=1):
        """ Read in a single packet via UDP

        Args:
            timeout (float): Time to wait for packet before moving on

        Returns:
            Full frame as array if successful, else None

        """
        # Configure
        self.data_socket.settimeout(timeout)

        # Frame buffer
        all_data = []
        packet_num_all = []
        byte_count_all = []

        packet_in_chunk = 0
        num_all_packets = 0
        num_chunks = 0

        s_time = dt.utcnow()
        start_time = s_time.isoformat()+'Z'

        try:
            while True:
                packet_num, byte_count, packet_data = self._read_data_packet()
                self.record_timestamp()
                all_data.append(packet_data)
                packet_num_all.append(packet_num)
                byte_count_all.append(byte_count)
                packet_in_chunk += 1
                num_all_packets += 1

                #### Stopping after n seconds
                curr_time = dt.utcnow()
                if (curr_time - s_time) > datetime.timedelta(seconds=acquisition_time): #alternatively: N_SECONDS
                    end_time = dt.utcnow().isoformat()+'Z'
                    print("Total packets captured ", num_all_packets)
                    self.save_timestamps()
                    self.time_stamps = []
                    return (all_data, packet_num_all, byte_count_all, start_time, end_time)

        except socket.timeout:
            end_time = dt.utcnow().isoformat()+'Z'
            print("Total packets captured ", num_all_packets)
            self.save_timestamps()
            self.time_stamps = []
            return (all_data, packet_num_all, byte_count_all, start_time, end_time)

        except KeyboardInterrupt:
            end_time = dt.utcnow().isoformat()+'Z'
            print("Total packets captured ", num_all_packets)
            self.save_timestamps()
            self.time_stamps = []
            return (all_data, packet_num_all, byte_count_all, start_time, end_time)
    
    def record_timestamp(self) -> None:
        """[Called within self.acquire() after each data capture to time stamp data]
        """
        self.time_stamps.append(time.perf_counter())
        self.local_time_stamps.append(str(dt.now()))

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

# To test code, run this file.
if __name__ == '__main__':

    rf_s = RF_Sensor(filename="rf", sensor_on=True)
    # time.sleep(5)
    rf_s.acquire(acquisition_time=10)
    rf_s.print_stats()