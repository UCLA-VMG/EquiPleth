--- Lua Config for AWR 1443
dofile("C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\Startup.lua")

os.execute("sleep 10")


COM_PORT = 9
RADARSS_PATH = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr12xx_xwr14xx_radarss.bin"
MASTERSS_PATH = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr12xx_xwr14xx_masterss.bin"
SAVE_DATA_PATH = "C:\\ti\\mmwave_studio_02_00_00_02\\mmWaveStudio\\PostProc\\adc_data.bin"

-- FrameConfig
-----------------------------------------------------------

--------------------- --------
ar1.FullReset()
ar1.SOPControl(2)
ar1.Connect(COM_PORT,115200,1000)
------------------------------

ar1.Calling_IsConnected()
ar1.SelectChipVersion("AR1243")
ar1.SelectChipVersion("AR1243")

ar1.frequencyBandSelection("77G")
ar1.SelectChipVersion("XWR1443")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-------- DOWNLOAD FIRMARE --------
ar1.DownloadBSSFw(RADARSS_PATH)
ar1.GetBSSFwVersion()
ar1.GetBSSPatchFwVersion()
ar1.DownloadMSSFw(MASTERSS_PATH)
ar1.PowerOn(0, 1000, 0, 0)
ar1.RfEnable()
ar1.GetBSSFwVersion()
-------------------------------------

-------- STATIC CONFIG --------------
-- ar1.ChanNAdcConfig(1, 0, 0, 1, 0, 0, 0, 2, 1, 0)
ar1.ChanNAdcConfig(1, 0, 0, 1, 0, 0, 0, 2, 1, 0)
ar1.LPModConfig(0, 0)
ar1.RfInit()
-------------------------------------

-------- DATA CONFIG ----------------
ar1.DataPathConfig(513, 1216644097, 0)
ar1.LvdsClkConfig(1, 1)
-- ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
ar1.LVDSLaneConfig(0, 1, 0, 0, 0, 1, 0, 0)
-------------------------------------

-------- SENSOR CONFIG ----- --------
-- ar1.ProfileConfig(0, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, TX_START_TIME, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN)
ar1.ProfileConfig(0, 77, 30, 7, 62, 0, 0, 0, 0, 0, 0, 60.012, 1, 256, 5000, 0, 0, 30)
-- ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
-- ar1.FrameConfig(START_CHIRP_TX, END_CHIRP_TX, NUM_FRAMES, CHIRP_LOOPS, PERIODICITY, 0, 0, 1)
ar1.FrameConfig(0, 0, 0, 1, 8.333335, 0, 0, 1)
-------------------------------------

-------------------------------------
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(25)
-------------------------------------

ar1.CaptureCardConfig_StartRecord(SAVE_DATA_PATH, 1)
ar1.StartFrame()