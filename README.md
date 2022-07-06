# TiComp
Contactless extraction of physiological signals using thermal infrared imaging

Refer for setting thermal camera configuration - https://flir.custhelp.com/app/answers/detail/a_id/1021/~/temperature-linear-mode

Setting cameras to temperature linear mode
FLIR Ax5
To make a Ax5 camera stream temperature linear, three GenICam registers must be set. Set the registers as follows:

TemperatureLinearMode, should be set to true.
PixelFormat, should be set to Mono14.
CMOSBitDepth, should be set to bit14bit.
To transform the signal to temperature in Kelvin, the following formulas should be followed:

TemperatureLinearResolution is set to High: multiply signal by 0.04.
TemperatureLinearResolution is set to Low: multiply signal by 0.4.