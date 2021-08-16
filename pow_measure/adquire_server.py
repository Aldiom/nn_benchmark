from ctypes import *
from dwfconstants import *
import time
import sys 
import numpy as np
import socket

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

# declare ctype variables
hdwf = c_int()
sts = c_byte()
vScale = 1 #220/12 # voltage scaling factor
hzAcq = c_double(50000) # 50 kHz
captureTime = 2 
nSamples = int(captureTime * 50000)
rgdSamples1 = (c_double*nSamples)()
rgdSamples2= (c_double*nSamples)()
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()
fLost = 0 
fCorrupted = 0 

ch1 = c_int(0)
ch2 = c_int(1)

# print DWF version
version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

# open device
print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    print("failed to open device")
    quit()

# set up acquisition
dwf.FDwfAnalogInChannelEnableSet(hdwf, ch1, c_bool(True)) # enable ch 1
dwf.FDwfAnalogInChannelEnableSet(hdwf, ch2, c_bool(True)) # enable ch 2

dwf.FDwfAnalogInChannelRangeSet(hdwf, ch1, c_double(4))  # ch1 for current
dwf.FDwfAnalogInChannelRangeSet(hdwf, ch2, c_double(24)) # ch2 for voltage
dwf.FDwfAnalogInChannelOffsetSet(hdwf, ch2, c_double(12)) 

dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(captureTime)) # -1 infinite record length

# wait at least 2 seconds for the offset to stabilize
time.sleep(2)

HOST = '10.42.0.1'  # Standard loopback interface address (localhost)
PORT = 7700        # Port to listen on (non-privileged ports are > 1023)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen()
print('Server listening...')
while True:
    try:
        conn, addr = sock.accept()
        print('Connected by', addr)
        while True:
            data = conn.recv(32)
            if data == b'endMeas':
                break
            if data == b'trigMeas':
                print("Starting acquisition...")
                time.sleep(.2)
                #dwf.FDwfDeviceTriggerPC(hdwf)
                dwf.FDwfAnalogInConfigure(hdwf, c_bool(False), c_bool(True))
                cSamples = 0
                while cSamples < nSamples:
                    dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
                    if cSamples == 0 and sts in (DwfStateConfig, DwfStatePrefill, DwfStateArmed) :
                        # Acquisition not yet started.
                        continue
                    dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))

                    cSamples += cLost.value

                    if cLost.value :
                        fLost = 1
                    if cCorrupted.value :
                        fCorrupted = 1

                    if cAvailable.value==0 :
                        continue

                    if cSamples+cAvailable.value > nSamples :
                        cAvailable = c_int(nSamples-cSamples)

                    # get channel 1 data
                    dwf.FDwfAnalogInStatusData(hdwf, ch1, byref(rgdSamples1,sizeof(c_double)*cSamples), cAvailable) 
                    # get channel 2 data
                    dwf.FDwfAnalogInStatusData(hdwf, ch2, byref(rgdSamples2,sizeof(c_double)*cSamples), cAvailable)
                    cSamples += cAvailable.value

                print("Acquisition done")
                if fLost:
                    print("Samples were lost! Reduce frequency")
                if fCorrupted:
                    print("Samples could be corrupted! Reduce frequency")

                currMeas = 10000 * np.fromiter(rgdSamples1, dtype='float32') # 0.1 V/A
                voltMeas = vScale * np.fromiter(rgdSamples2, dtype='float32')
                #voltMeas = 19 * np.ones_like(currMeas)
                powMeas = int(round(np.mean(currMeas * voltMeas)))
                print('Measured power:', powMeas, 'mW')
                conn.sendall(str(powMeas).encode('utf-8'))
        conn.close()
        print('Connection closed')
    except KeyboardInterrupt:
        break
try:
	conn.close()
except NameError:
	pass
sock.close()
dwf.FDwfDeviceCloseAll()
