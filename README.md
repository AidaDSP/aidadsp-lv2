# Aida DSP lv2 plugin bundle #

### What is this repository for? ###

* A bundle of audio plugins from [Aida DSP](http://aidadsp.cc)
* Bundle version: 1.0
* This bundle is intended to be used with moddevices's products and derivatives

### Plugin list ###

* rt-neural-lv2

#### rt-neural-lv2 ####

It's a simple headless lv2 plugin inspired by

- [NeuralPi](https://github.com/GuitarML/NeuralPi)
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural.git)

This lv2 plugin is a simple wrapper to inference classes used in NeuralPi project. All
the credits go to the original authors.

I've decided to implement this plugin to be able to compile original NeuralPi plugin without JUCE
and also to eliminate every additional effect that has been added during time.

__*WIP: currently this plugin is under work since CPU consumption is not acceptable, come back later!*__

##### Generate json models #####

This implies neural network training. Please follow __*Automated_GuitarAmpModelling.ipynb*__ script available on

- [my Automated-GuitarAmpModelling fork](https://github.com/MaxPayne86/Automated-GuitarAmpModelling/tree/aidadsp_devel)

##### Dataset #####

Since I was not satisfied with dataset proposed by orignal authors I've put together one:

- [Thomann Stompenberg Dataset](https://github.com/MaxPayne86/ThomannStompenbergDataset)
