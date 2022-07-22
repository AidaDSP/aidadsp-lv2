#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lv2.h>

#include <iostream>
#include <RTNeural/RTNeural.h>

/**********************************************************************************************************************************************************/

#define PLUGIN_URI "http://aidadsp.cc/plugins/aidadsp-bundle/rt-neural-generic"
#define LSTM_MODEL_JSON_FILE_NAME "lstm-model.json"
enum {IN, OUT_1, PARAM1, PARAM2, MASTER, BYPASS, PLUGIN_PORT_COUNT};

/**********************************************************************************************************************************************************/

class RtNeuralGeneric
{
public:
    RtNeuralGeneric() {}
    ~RtNeuralGeneric() {}
    static LV2_Handle instantiate(const LV2_Descriptor* descriptor, double samplerate, const char* bundle_path, const LV2_Feature* const* features);
    static void activate(LV2_Handle instance);
    static void deactivate(LV2_Handle instance);
    static void connect_port(LV2_Handle instance, uint32_t port, void *data);
    static void run(LV2_Handle instance, uint32_t n_samples);
    static void cleanup(LV2_Handle instance);
    static const void* extension_data(const char* uri);
    float *in;
    float *out_1;
    float *param1;
    float *param2;
    float *master;
    float master_old;
    int *bypass;
    int bypass_old;

    int model_loaded = 0;
    // The input vector size for the model
    // 1 is for a snap shot model otherwise is a conditioned model
    int input_size = 0;
    static void loadModel(LV2_Handle instance, const char *bundle_path, const char *fileName);

private:
    std::unique_ptr<RTNeural::Model<float>> model;

    // Pre-Allowcate arrays for feeding the models
    float inArray1 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[2] = { 0.0, 0.0 };
    float inArray2 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[3] = { 0.0, 0.0, 0.0 };

    static float calcGain(float gain, float gain_old, uint32_t n_samples, uint32_t index);
};