# Aida DSP lv2 plugin bundle #

### What is this repository for? ###

* A bundle of audio plugins from [Aida DSP](http://aidadsp.cc)
* Bundle version: 1.0
* This bundle is intended to be used with moddevices's products and derivatives

### Plugin list ###

* rt-neural-lv2

#### rt-neural-lv2 ####

It's a simple headless lv2 plugin that leverages [RTNeural](https://github.com/jatinchowdhury18/RTNeural.git) to model
pedals or amps.

- This plugin supports json model files loading via specific atom messages

##### Generate json models #####

This implies neural network training. Please follow __*Automated_GuitarAmpModelling.ipynb*__ script available on

- [Automated-GuitarAmpModelling](https://github.com/MaxPayne86/Automated-GuitarAmpModelling/tree/aidadsp_devel)

##### Dataset #####

Since I was not satisfied with dataset proposed by orignal authors I've put together one:

- [Thomann Stompenberg Dataset](https://github.com/MaxPayne86/ThomannStompenbergDataset)

### Build ###

Below a guide on how to cross compile this bundle with [aidadsp sdk](https://drive.google.com/drive/folders/1-AAfAP-FAddCw0LJuvzsW8m_1lWHKXaV?usp=sharing).
You can extract cmake commands to fit your build system.

- RTNEURAL_XSIMD=ON or RTNEURAL_EIGEN=ON to select an available backend for RTNeural library

for other options see [RTNeural](https://github.com/jatinchowdhury18/RTNeural.git) project.

```
1. install sdk with ./poky-glibc-x86_64-aidadsp-sdk-image-aarch64-nanopi-neo2-toolchain-2.1.15.sh
3. source environment-setup-aarch64-poky-linux
4. git clone https://github.com/AidaDSP/aidadsp-lv2.git && cd aidadsp-lv2
5. mkdir build && cd build
6. cmake -DCMAKE_BUILD_TYPE=Release -DRTNEURAL_XSIMD=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON ../
7. cmake --build .
8. make install DESTDIR="/tmp/"

bundle will be placed in /tmp/ ready to be copied on your device
```
