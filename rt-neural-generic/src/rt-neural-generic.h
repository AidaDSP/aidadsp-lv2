/*
 * aidadsp-lv2
 * Copyright (C) 2022-2023 Massimo Pennazio <maxipenna@libero.it>
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#ifndef AIDADSP_COMMERCIAL
#error AIDADSP_COMMERCIAL undefined, must be 0 or 1
#endif
#ifndef AIDADSP_MODEL_LOADER
#error AIDADSP_MODEL_LOADER undefined, must be 0 or 1
#endif

// enabled by default, can be turned off
#ifndef AIDADSP_CONDITIONED_MODELS
#define AIDADSP_CONDITIONED_MODELS 1
#endif

// DC blocker is optional for model loader
#if AIDADSP_MODEL_LOADER
#define AIDADSP_OPTIONAL_DCBLOCKER 1
#else
#define AIDADSP_OPTIONAL_DCBLOCKER 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <lv2/atom/forge.h>
#include <lv2/atom/util.h>
#include <lv2/log/log.h>
#include <lv2/log/logger.h>
#include <lv2/midi/midi.h>
#include <lv2/patch/patch.h>
#include <lv2/state/state.h>
#include <lv2/urid/urid.h>
#include <lv2/worker/worker.h>

#include <model_variant.hpp>

#include <Biquad.h>
#include <ValueSmoother.hpp>

#include "uris.h"

#if AIDADSP_COMMERCIAL
    #define TWINCLASSIC 0
    #define LEAD 1
    #define TWEAKER 2
    #define VIBRO 3
    #define JCVM 4
    #define SHOWCASE 5
#endif

#if AIDADSP_COMMERCIAL && (AIDADSP_MODEL_DEFINE != SHOWCASE)
    #include "libmodla.h"
#endif

#if AIDADSP_COMMERCIAL && (AIDADSP_MODEL_DEFINE == TWINCLASSIC \
    || AIDADSP_MODEL_DEFINE == VIBRO \
    || AIDADSP_MODEL_DEFINE == JCVM)
    #define AIDADSP_PARAMS 1
    #define AIDADSP_CHANNELS 1
#elif AIDADSP_COMMERCIAL && (AIDADSP_MODEL_DEFINE == TWEAKER)
    #define AIDADSP_PARAMS 1
    #define AIDADSP_CHANNELS 2
#elif AIDADSP_COMMERCIAL && (AIDADSP_MODEL_DEFINE == SHOWCASE)
    #ifdef AIDADSP_CONDITIONED_MODELS
        #undef AIDADSP_CONDITIONED_MODELS
    #endif
    #define AIDADSP_MODEL_LOADER 0
    #define AIDADSP_PARAMS 0
    #undef AIDADSP_CHANNELS
#else
    #define AIDADSP_PARAMS 2
#endif

/**********************************************************************************************************************************************************/

typedef enum {
    IN, OUT_1,
#if AIDADSP_MODEL_LOADER
    PLUGIN_CONTROL, PLUGIN_NOTIFY,
#else
    PLUGIN_MODEL_INDEX,
#endif
    IN_LPF, PREGAIN,
    NET_BYPASS,
#if AIDADSP_PARAMS >= 1
    PARAM1,
#if AIDADSP_PARAMS >= 2
    PARAM2,
#endif
#endif
    EQ_BYPASS, EQ_POS, BASS, BFREQ, MID, MFREQ, MIDQ, MTYPE, TREBLE, TFREQ, DEPTH, PRESENCE,
#if AIDADSP_OPTIONAL_DCBLOCKER
    DCBLOCKER,
#endif
    MASTER,
    INPUT_SIZE,
#if AIDADSP_CHANNELS >= 1
    CHANNEL1,
#if AIDADSP_CHANNELS >= 2
    CHANNEL2,
#endif
#endif
    PLUGIN_ENABLED,
    PLUGIN_PORT_COUNT} ports_t;

// Everything needed to run a model
struct DynamicModel {
    ModelVariantType variant;
#if AIDADSP_MODEL_LOADER
    char* path;
#endif
    bool input_skip; /* Means the model has been trained with first input element skipped to the output */
    float input_gain;
    float output_gain;
    float samplerate;
#if AIDADSP_CONDITIONED_MODELS
    LinearValueSmoother param1Coeff;
    LinearValueSmoother param2Coeff;
    bool paramFirstRun;
#endif
};

#define PROCESS_ATOM_MESSAGES
enum WorkerMessageType {
    kWorkerLoad,
    kWorkerApply,
    kWorkerFree
};

// common fields to all worker messages
struct WorkerMessage {
    WorkerMessageType type;
};

// WorkerMessage compatible, to be used for kWorkerLoad
struct WorkerLoadMessage {
    WorkerMessageType type;
#if AIDADSP_MODEL_LOADER
    char path[1024];
#else
    int modelIndex;
#endif
};

// WorkerMessage compatible, to be used for kWorkerApply or kWorkerFree
struct WorkerApplyMessage {
    WorkerMessageType type;
    DynamicModel* model;
};

/* Convert a value in dB's to a coefficent */
#define DB_CO(g) ((g) > -90.0f ? powf(10.0f, (g) * 0.05f) : 0.0f)
#define CO_DB(v) (20.0f * log10f(v))

/* Define a macro to scale % to coeff */
#define PC_CO(g) ((g) < 100.0f ? (g / 100.0f) : 1.0f)

/* Define a macro to re-maps a number from one range to another  */
#define MAP(x, in_min, in_max, out_min, out_max) (((x - in_min) * (out_max - out_min) / (in_max - in_min)) + out_min)

/* Defines for tone controls */
#define PEAK 0.0f
#define BANDPASS 1.0f
#define DEPTH_FREQ 75.0f
#define DEPTH_Q 0.707f
#define PRESENCE_FREQ 900.0f
#define PRESENCE_Q 0.707f

/* Defines for antialiasing filter */
#define INLPF_MAX_CO 0.99f * 0.5f /* coeff * ((samplerate / 2) / samplerate) */
#define INLPF_MIN_CO 0.25f * 0.5f /* coeff * ((samplerate / 2) / samplerate) */

/* Define the acceptable threshold for model test */
#define TEST_MODEL_THR 1.0e-5

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
    float *pregain_db;
    ExponentialValueSmoother preGain;
#if AIDADSP_CONDITIONED_MODELS
    float *param1;
    float *param2;
#endif
#if AIDADSP_OPTIONAL_DCBLOCKER
    float *dc_blocker_param;
#endif
    float *master_db;
    ExponentialValueSmoother masterGain;
    float *net_bypass;
    bool loading;
    float *in_lpf_pc;
    float in_lpf_pc_old;
    /* Eq section */
    float *eq_position;
    float *bass_boost_db;
    float bass_boost_db_old;
    float *bass_freq;
    float bass_freq_old;
    float *mid_boost_db;
    float mid_boost_db_old;
    float *mid_freq;
    float mid_freq_old;
    float *mid_q;
    float mid_q_old;
    float *mid_type;
    float mid_type_old;
    float *treble_boost_db;
    float treble_boost_db_old;
    float *treble_freq;
    float treble_freq_old;
    float *depth_boost_db;
    float depth_boost_db_old;
    float *presence_boost_db;
    float presence_boost_db_old;
    float *eq_bypass;
    float *input_size;
    float *enabled;

    // to be used for reporting input_size to GUI (0 for error/unloaded, otherwise matching input_size)
    int last_input_size;
#if ! AIDADSP_MODEL_LOADER
    float *model_index;
    float model_index_old;
#ifdef AIDADSP_CHANNELS
    std::vector<float*> channel_switch;
#endif
#endif

#if AIDADSP_MODEL_LOADER
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
#endif

    static LV2_Worker_Status work(LV2_Handle instance,
                                       LV2_Worker_Respond_Function respond,
                                       LV2_Worker_Respond_Handle   handle,
                                       uint32_t                    size,
                                       const void*                 data);
    static LV2_Worker_Status work_response(LV2_Handle instance, uint32_t size, const void* data);
#if AIDADSP_MODEL_LOADER
    static DynamicModel* loadModelFromPath(LV2_Log_Logger* logger, const char* path, int* input_size_ptr, const float old_param1, const float old_param2);
#else
    static DynamicModel* loadModelFromIndex(LV2_Log_Logger* logger, int modelIndex, int* input_size_ptr, const float old_param1, const float old_param2);
    static float controlsToModelIndex(int modelIndex, const std::vector<float>& ctrls);
#endif
    static void freeModel(DynamicModel* model);

    // Features
    LV2_URID_Map*        map;
    LV2_Worker_Schedule* schedule;
    LV2_Log_Log*         log;

#if AIDADSP_MODEL_LOADER
    // Forge for creating atoms
    LV2_Atom_Forge forge;

    // Forge frame for notify port (for writing worker replies)
    LV2_Atom_Forge_Frame notify_frame;
#endif

    // Logger convenience API
    LV2_Log_Logger logger;

    // Ports
#if AIDADSP_MODEL_LOADER
    const LV2_Atom_Sequence* control_port;
    LV2_Atom_Sequence*       notify_port;
#endif
    float*                   output_port;
    float*                   input_port;

    // URIs
    PluginURIs uris;

private:
    double samplerate;
#if AIDADSP_COMMERCIAL && (AIDADSP_MODEL_DEFINE != SHOWCASE)
    uint32_t run_count;
#endif

    Biquad *dc_blocker;
    Biquad *in_lpf;
    Biquad *bass;
    Biquad *mid;
    Biquad *treble;
    Biquad *depth;
    Biquad *presence;

    DynamicModel* model;

    static void applyBiquadFilter(float *out, const float *in, Biquad *filter, uint32_t n_samples);
    static void applyModel(DynamicModel *model, float *out, uint32_t n_samples);
    static void applyToneControls(float *out, const float *in, LV2_Handle instance, uint32_t n_samples);
    static bool testModel(LV2_Log_Logger* logger, DynamicModel *model, const std::vector<float>& xData, const std::vector<float>& yData);
};

