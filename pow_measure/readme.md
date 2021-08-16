### Power measurement script 

This script is meant to be used with a Digilent Analog Discovery 2. In my work I also used an AEMC Instruments SL261 Hall effect current probe, which I hooked to the power supply input of each of the devices I tested with. Voltage was also measured simultaneously on channel 2, using a pair of test probes on the power input. For measuring desktop PCs power usage remember to use a transformer in parallel to the PC, that way you'll get an in-phase proportional signal of the input voltage without frying anything. You'll have to adjust the scaling factor accordingly in the script code.

The script runs a server on a PC connected to the Discovery, that PC must have the [Waveforms](https://reference.digilentinc.com/software/waveforms/waveforms-3/start) software installed and it must be in the same local network of the device under test. You can change the server IP in the script code.
