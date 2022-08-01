from typing import Tuple
import torch
import numpy as np

def create_fast_slow_matrix(data: np.array) -> np.array:
    """ Create the range slow-time matrix (Radar data matrix).

    Args:
        data (np.array): The organized data from the RF sensors.

    Returns:
        np.array: The range slow-time matrix.
    """
    # Taking only 1 TX and RX (as per the dataset).
    data_ = data[:,0,0,:]
    # DC Compensation.
    data_f = np.fft.fft(data_, axis = 1)
    return data_f

def find_range(data_f: np.array, samp_f: float, freq_slope: float, samples: int, 
               max_range_allowed: float = 1, min_idx: int = 5) -> int:
    """ Find the max range from the Radar data matrix.

    Args:
        data_f (np.array): The range slow-time matrix.
        samp_f (float): _description_
        freq_slope (float): Frequency slope for the FMCW radar
        samples (int): _description_
        max_range_allowed (float, optional): _description_. Defaults to 1.
        min_idx (int, optional): Starting index to find the range. Skip the first few range bins.
                                   To ignore false detection when enclosing the hardware in a box. Defaults to 5.

    Returns:
        int: index of the maximum range bin
    """
    # Get the maximum index where to end.
    max_idx = max_range_allowed / (samp_f * 2.98e8 / freq_slope / 2 /samples)
    # Get the minimum index from where to start.
    data_f = data_f[:,min_idx:int(max_idx)]
    # Find the energy using the l1 norm
    data_f = np.abs(data_f)
    data_f = np.sum(data_f, axis=0)
    # Get the index and account for min_idx offset.
    index = np.argmax(data_f) + min_idx
    return index

# TODO: not used
def vibration_fft_windowing(data_f: int, range_index: int, 
                            window_size: int) -> Tuple[np.array, np.array]:
    """ Window the FFT of the Radara data matrix for vibration analysis.

    Args:
        data_f (int): _description_
        range_index (int): Best range bin. Obtained from find_range().
        window_size (int): _description_

    Returns:
        phase_f (np.array): Temporal FFT of the data_phase.
        data_phase (np.array): Unwrapped phase of the Radar matrix windowed around the best range bin.
    """
    data_phase = np.angle(data_f)
    data_phase = np.unwrap(data_phase, axis = 0)

    window = np.blackman(window_size)
    data_phase = data_phase[:, range_index-len(window)//2:range_index+len(window)//2 + 1] * window
    range_index = len(window)//2

    phase_f = np.fft.fft(data_phase, axis = 0)
    return phase_f, data_phase

def rotateIQ(iq_array):
    rand_degree = np.random.rand()*360
    theta = 2*np.pi*rand_degree/360
    rotation_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),np.cos(theta)]])
    for i in range(iq_array.shape[2]):
        iq_array[:,:,i,:] = torch.matmul(torch.tensor(rotation_mat).type(torch.float32),iq_array[:,:,i,:])
    return iq_array