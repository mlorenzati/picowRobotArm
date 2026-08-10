# Robot Arm pico w Controller

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/b55575d5-f2c4-44a3-befc-79849ea85ee8" />

Qt application to control a 6-axis robotic arm using a Raspberry Pi Pico with micropython.
[Robot Arm ](https://makerworld.com/es/models/1134925-robotic-arm-with-servo-arduino#profileId-1135927) by Emre Kalen 

## Features

- 6 angle sliders (0-180°)
- 7 servos control
- Individual servo disabling for testing and adjustment
- 6 planned app and firmware modes
-   USB Serial communication (implemented)
-   Bluetooth low energy mode (pending)
-   wifi mode (pending)
- Automatic packet transmission on movement
- Connect / Disconnect
- Refresh COM ports

## Packet format
Allows sending a D instead of a number, making that servo disabled
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
