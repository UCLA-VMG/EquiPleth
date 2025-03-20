# Equitable Plethysmography Blending Camera and 77 GHz Radar Sensing for Equitable, Robust Plethysmography

> __Blending Camera and 77 GHz Radar Sensing for Equitable, Robust Plethysmography__  
> [Alexander Vilesov](https://asvilesov.github.io/), [Pradyumna Chari](https://pradyumnachari.github.io/), [Adnan Armouti](https://adnan-armouti.github.io/), [Anirudh Bindiganavale Harish](https://anirudhbharish.github.io/), [Kimaya Kulkarni](https://kimayak.github.io/), [Ananya Deoghare](https://www.linkedin.com/in/ananya-deepak-deoghare-b60435106/), [Laleh Jalilian](https://www.linkedin.com/in/laleh-jalilian-md-162754115/), [Achuta Kadambi](https://www.ee.ucla.edu/achuta-kadambi/)<br/>
> _ACM Transactions on Graphics (__SIGGRAPH__), July 2022_  
> __[Project page](https://visual.ee.ucla.edu/equi_pleth_camera_rf.htm/)&nbsp;/ [Paper](https://dl.acm.org/doi/10.1145/3528223.3530161)&nbsp;/ [Dataset Request Form](https://docs.google.com/forms/d/e/1FAIpQLSfFqFXyp4-xH2NSx7Hk33FxSUb-MdwmwVCH3C_kUeQshJzn7Q/viewform)&nbsp;/ [Presentation](https://docs.google.com/presentation/d/1arSZlesm3VgBBglVNgi2xmAntTNgeLUfbt1LALtjDXw/edit#slide=id.g1428981a6f5_4_80)&nbsp;/ [Los-Res Paper](https://drive.google.com/file/d/1TSQpXoqkLtEgA3K1bSsQedVTjyQb65sa/view?usp=sharing)__

For details on the citation format, kindly refer to the [Citation](https://github.com/UCLA-VMG/EquiPleth#citation) section below.

**Repository Contributors**: [Adnan Armouti](https://github.com/adnan-armouti), [Anirudh Bindiganavale Harish](https://github.com/AnirudhBHarish), [Alexander (Sasha) Vilesov](https://github.com/asvilesov/), [Pradyumna Chari](https://github.com/pradyumnachari/).

<hr /> 

## Hardware Setup

This section is pertinent to those who wish to collect tier own data. All following instructions are with respect to the hardware used by the authors.

Prior to running the codes for data acquisition, please ensure the following:

**(1) MX800**

Kindly refer to the following [link](https://compatibility.rockwellautomation.com/Pages/MultiProductFindDownloads.aspx?crumb=112&mode=3&refSoft=1&versions=59657). The link will redirect to the software we have utilized to connect to the MX800 through the Ethernet port. A configuration named mx800.bpc has been provided in data_acquisition _/sensors/configs_ folder. However, you may need to regenerate one for your specific system.

Once connected, please clone this [GitHub Repository](https://github.com/xeonfusion/VSCaptureMP) that contains the C# code to collect data from the MX800. You will need to compile the C# code with a tool such as [Visual Studio](https://visualstudio.microsoft.com/). The generated binaries will need to be linked with the mx800_sensor.py file. The binaries can be linked through the \_\__init_\_\_ function of the _MX800\_Sensor_ class.

**(2) RF**

Any documentation for the AWR1443Boost radar can be found in the following [link](https://www.ti.com/tool/AWR1443BOOST).

To run data collection you will need to install [mmWave studio GUI tools for 1st-generation parts \(xWR1243, xWR1443, xWR1642, xWR1843, xWR6843, xWR6443\)](https://www.ti.com/tool/MMWAVE-STUDIO).

Also, please follow [these instructions](https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-h04ItoajtS/02.01.01.00/mmwave_studio_user_guide.pdf) provided by TI to install mmWave studio and any drivers necessary to operate the radar device correctly.

In-order to not configure the radar with certain parameters upon start-up each time, we have provided a lua script which can be fed into mmWave studio via the windows powershell to automatically boot the radar with the preset configurations. The following bash script commands (2 lines) can be used to boot mmWave studio runtime with the lua script.

```
>> cd "C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\RunTime"
```
```
>> cmd /C "C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\RunTime\mmWaveStudio.exe /lua "path_to_data_acquisition\sensors\configs\awr1443_config.lua"
```

**(3) RGB Camera**

If you would like to use the Zed Camera mentioned in the paper, kindly follow these [instructions](https://www.stereolabs.com/docs/get-started-with-zed/) provided by StereoLabs.

The _rgbd\_sensor.py_ file provided in the _data\_acquisition/sensors_ folder is for the Zed Camera used in the paper. For any other camera, please use _rgb\_sensor.py_.

<hr/>

## Data Acquisition

All runtime parameters can be adjusted by editing the _sensors\_config.ini_ file in _data\_acquisition/sensors/configs_.

The following command can be used to acquire data:
```
>> python sync_sensors.py
```
Please make sure navigate into the _data\_acquisition_ folder prior to running the file.

In _sync\_sensors.py_ please edit the rf\_dump\_path in _cleanup\_rf_. This is the location where mmWave studio continuously dumps the recorded data from the radar. This data is not needed, as _sync\_sensors.py_ records the required subset of the same data during the its runtime and creates the rf output file. The _cleanup\_rf_ function in _sync\_sensors.py_ deletes these unnecessary file to avoid redundancy.

<hr/>

## Dataset and Pre-prep

The EquiPleth dataset can be downloaded by filling this [Google Form](https://forms.gle/sajK7a3mGGufKNUEA).

If you choose to collect your own data, please adhere to the following pre-processing instructions to obtain a similar dataset to the EquiPleth dataset.

1) Use _data\_interpolation.py_ to interpolate the MX800 waveforms to the timestamps of the sensors.

2) Use a face cropping software (MTCNN in our case) to crop the face and save each frame as an image within the trial/volunteer's folder.

Hierarchy of the EquiPleth dataset - RGB Files
```
|
|--- rgb_files
|        |
|        |--- volunteer id 1 trial 1 (v_1_1)
|        |         |
|        |         |--- frame 0 (rgbd_rgb_0.png)
|        |         |--- frame 1 (rgbd_rgb_1.png)
|        |         |
|        |         |
|        |         |
|        |         |--- last frame (rgbd_rgb_899.png)
|        |         |--- ground truth PPG (rgbd_ppg.npy)
|        | 
|        | 
|        |--- volunteer id 1 trial 2 (v_1_2)
|        | 
|        | 
|        | 
|        |--- volunteer id 2 trial 1 (v_2_1)
|        |
|        |
|        |
|
|
|
|
|--- rf files
|        |
|        |---- volunteer id 1 trial 1 (1_1)
|        |           |
|        |           |--- Radar data (rf.pkl)
|        |           |--- ground truth PPG (vital_dict.npy)
|        |
|        |
|        |--- volunteer id 1 trial 2 (1_2)
|        |
|        |
|        |
|        |--- volunteer id 2 trial 1 (2_1)
|        |
|        |
|        |
|
|
|--- fitzpatrick labels file (fitzpatrick_labels.pkl)
|--- folds pickle file (demo_fold.pkl)
|--- {user generated fusion data after rgb & rf training (more details in the section below)}
```

Create a new folder (_dataset_ in our case) in _nndl_ and place the downloaded/processed dataset in the same.

<hr/>

## NNDL Execution

Please make sure to navigate into the _nndl_ folder prior to running the following scripts.

**(1) RGB / RF**

Run the following command to train the rf and the rgb models.
```
>> python {rf or rgb}/train.py --train-shuffle --verbose
```

Run the following command to test the rf and the rgb models.
```
>> python {rf or rgb}/test.py --verbose
```

**(2) Fusion Data Generation**

Run the following command to generate the pickle file with the data for the fusion model.
```
>> python data/fusion_gen.py --verbose
```

**(3) Fusion**

Run the following command to train the fusion model.
```
>> python fusion/train.py --shuffle --verbose
```

Run the following command to test the fusion model.
```
>> python fusion/test.py --verbose
```

**(4) Command Line Args**

For more info about the command line arguments, please run the following:
```
>> python {folder}/file.py --help
```

<hr/>

## References

1) Zheng, Tianyue, et al. "MoRe-Fi: Motion-robust and Fine-grained Respiration Monitoring via Deep-Learning UWB Radar." Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems. 2021.

2) Yu, Zitong, Xiaobai Li, and Guoying Zhao. "Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks." arXiv preprint arXiv:1905.02419 (2019).

<hr />

## Citation

```
@article{vilesov2022blending,
  title={Blending camera and 77 GHz radar sensing for equitable, robust plethysmography},
  author={Vilesov, Alexander and Chari, Pradyumna and Armouti, Adnan and Harish, Anirudh Bindiganavale and Kulkarni, Kimaya and Deoghare, Ananya and Jalilian, Laleh and Kadambi, Achuta},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  pages={1--14},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```
