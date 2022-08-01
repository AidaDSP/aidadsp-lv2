#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lv2.h>

#include <iostream>
#include <RTNeural/RTNeural.h>

#include <lsp-plug.in/dsp/dsp.h>
#include <lsp-plug.in/dsp-units/units.h>
#include <lsp-plug.in/dsp-units/filters/Filter.h>

/**********************************************************************************************************************************************************/

#define PLUGIN_URI "http://aidadsp.cc/plugins/aidadsp-bundle/rt-neural-generic"
#define JSON_MODEL_FILE_NAME "model.json"
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

    static void loadModel(LV2_Handle instance, const char *bundle_path, const char *fileName);

private:
    int model_loaded = 0;
    // The number of layers in the nn model
    int n_layers = 0;
    // The input vector size for the model
    // 1 is for a snap shot model otherwise is a conditioned model
    int input_size = 0;
    int input_skip = 0; /* Means the model has been trained with input elements skipped to the output */
    std::string type; /* The type of the first layer of a nn composed by two hidden layers (e.g., LSTM, GRU) */
    int hidden_size = 0; /* The hidden size of the above layer */

    lsp::dspu::Filter dc_blocker_f;
    lsp::dspu::filter_params_t dc_blocker_fp;

    /* Dynamic: whatever json model but very slow performance */
    //std::unique_ptr<RTNeural::Model<float>> model;

    /* GRU 8 */
    /*RTNeural::ModelT<float, 1, 1,
        RTNeural::GRULayerT<float, 1, 8>,
        RTNeural::DenseT<float, 8, 1>> model;*/

    /* GRU 24 */
    /*RTNeural::ModelT<float, 1, 1,
        RTNeural::GRULayerT<float, 1, 24>,
        RTNeural::DenseT<float, 24, 1>> model;*/

    /* LSTM 12 */
    /*RTNeural::ModelT<float, 1, 1,
        RTNeural::LSTMLayerT<float, 1, 12>,
        RTNeural::DenseT<float, 12, 1>> model;*/

    /* LSTM 16 */
    RTNeural::ModelT<float, 2, 1,
        RTNeural::LSTMLayerT<float, 2, 16>,
        RTNeural::DenseT<float, 16, 1>> model;

    // Pre-Allowcate arrays for feeding the models
    float inArray1 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[2] = { 0.0, 0.0 };
    float inArray2 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[3] = { 0.0, 0.0, 0.0 };

    static float rampValue(float start, float end, uint32_t n_samples, uint32_t index);
    static void applyGainRamp(float *buffer, float start, float end, uint32_t n_samples);
};
