#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <lv2/lv2plug.in/ns/ext/atom/forge.h>
#include <lv2/lv2plug.in/ns/ext/atom/util.h>
#include <lv2/lv2plug.in/ns/ext/log/log.h>
#include <lv2/lv2plug.in/ns/ext/log/logger.h>
#include <lv2/lv2plug.in/ns/ext/midi/midi.h>
#include <lv2/lv2plug.in/ns/ext/patch/patch.h>
#include <lv2/lv2plug.in/ns/ext/state/state.h>
#include <lv2/lv2plug.in/ns/ext/urid/urid.h>
#include <lv2/lv2plug.in/ns/ext/worker/worker.h>
#include <lv2/lv2plug.in/ns/lv2core/lv2.h>

#include <iostream>
#include <RTNeural/RTNeural.h>

#include <Biquad.h>

#include "uris.h"

/**********************************************************************************************************************************************************/

typedef enum ports_t {IN, OUT_1, IN_VOL, PARAM1, PARAM2, MASTER, BYPASS, PLUGIN_CONTROL, PLUGIN_NOTIFY, PLUGIN_PORT_COUNT} ports;

#define PROCESS_ATOM_MESSAGES
typedef struct {
    LV2_Atom atom;
    char*  path;
} PluginResponseMessage;

/* Define a macro for converting a gain in dB to a coefficient */
#define DB_CO(g) ((g) > -90.0f ? powf(10.0f, (g) * 0.05f) : 0.0f)

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
    float *in_vol_db;
    float in_vol_old;
    float *param1;
    float *param2;
    float *master_db;
    float master_old;
    float *bypass;

    static LV2_State_Status restore(LV2_Handle instance,
                                       LV2_State_Retrieve_Function retrieve,
                                       LV2_State_Handle            handle,
                                       uint32_t                    flags,
                                       const LV2_Feature* const*   features);
    static LV2_State_Status save(LV2_Handle instance,
                                       LV2_State_Store_Function  store,
                                       LV2_State_Handle          handle,
                                       uint32_t                  flags,
                                       const LV2_Feature* const* features);
    static LV2_Worker_Status work(LV2_Handle instance,
                                       LV2_Worker_Respond_Function respond,
                                       LV2_Worker_Respond_Handle   handle,
                                       uint32_t                    size,
                                       const void*                 data);
    static LV2_Worker_Status work_response(LV2_Handle instance, uint32_t size, const void* data);
    static int loadModel(LV2_Handle instance, const char *path);

    // Features
    LV2_URID_Map*        map;
    LV2_Worker_Schedule* schedule;
    LV2_Log_Log*         log;

    // Forge for creating atoms
    LV2_Atom_Forge forge;

    // Logger convenience API
    LV2_Log_Logger logger;

    // Model json file path
    const char* path; // Path of file
    uint32_t path_len; // Length of path

    // Ports
    const LV2_Atom_Sequence* control_port;
    LV2_Atom_Sequence*       notify_port;
    float*                   output_port;
    float*                   input_port;

    // Forge frame for notify port (for writing worker replies)
    LV2_Atom_Forge_Frame notify_frame;

    // URIs
    PluginURIs uris;

    // Current position in run()
    uint32_t frame_offset;

private:
    double samplerate;

    int model_loaded; // Used to prevent audio thread from using model if not ready
    int model_new; // Used to send a small notification to host when model is changed

    // The number of layers in the nn model
    int n_layers;
    // The input vector size for the model
    // 1 is for a snap shot model otherwise is a conditioned model
    int input_size;
    int input_skip; /* Means the model has been trained with first input element skipped to the output */
    std::string type; /* The type of the first layer of a nn composed by two hidden layers (e.g., LSTM, GRU) */
    int hidden_size; /* The hidden size of the above layer */

    Biquad *dc_blocker;

    /* Static: only json files containing models below will be loaded */
    RTNeural::ModelT<float, 1, 1,
        RTNeural::LSTMLayerT<float, 1, 16>,
        RTNeural::DenseT<float, 16, 1>> lstm_16;
    RTNeural::ModelT<float, 1, 1,
        RTNeural::LSTMLayerT<float, 1, 12>,
        RTNeural::DenseT<float, 12, 1>> lstm_12;
    RTNeural::ModelT<float, 1, 1,
        RTNeural::GRULayerT<float, 1, 8>,
        RTNeural::DenseT<float, 8, 1>> gru_8;

    /* Dynamic: whatever json model will be loaded but poor performance */
    //std::unique_ptr<RTNeural::Model<float>> model;

    int model_index; /* Used to store model type */

    // Pre-allocate arrays for feeding the models
    float inArray1 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[2] = { 0.0, 0.0 };
    float inArray2 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[3] = { 0.0, 0.0, 0.0 };

    static float rampValue(float start, float end, uint32_t n_samples, uint32_t index);
    static void applyGainRamp(float *out, const float *in, float start, float end, uint32_t n_samples);
    static void applyBiquadFilter(float *out, const float *in, Biquad *filter, uint32_t n_samples);
    static void applyModel(float *out, const float *in, LV2_Handle instance, uint32_t n_samples);
    static void applyModel(float *out, const float *in, float param1, LV2_Handle instance, uint32_t n_samples);
    static void applyModel(float *out, const float *in, float param1, float param2, LV2_Handle instance, uint32_t n_samples);
};
