# Robot Arm pico w Controller

Qt application to control a 6-axis robotic arm using a Raspberry Pi Pico.
[Robot Arm ](https://makerworld.com/es/models/1134925-robotic-arm-with-servo-arduino#profileId-1135927) by Emre Kalen 

## Features

- 6 angle sliders (0-180°)
- USB Serial communication
- Automatic packet transmission on movement
- Connect / Disconnect
- Refresh COM ports

## Packet format

```
90 90 90 90 90 90
```

9600 baud.

## Build

Windows

```
build.bat
```

Linux

```
./build.sh
```