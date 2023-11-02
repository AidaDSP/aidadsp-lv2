/*
 * aidadsp-lv2
 * Copyright (C) 2022-2023 Massimo Pennazio <maxipenna@libero.it>
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "rt-neural-generic.h"

/**********************************************************************************************************************************************************/

static const LV2_Descriptor Descriptor = {
    PLUGIN_URI,
    RtNeuralGeneric::instantiate,
    RtNeuralGeneric::connect_port,
    RtNeuralGeneric::activate,
    RtNeuralGeneric::run,
    RtNeuralGeneric::deactivate,
    RtNeuralGeneric::cleanup,
    RtNeuralGeneric::extension_data
};

/**********************************************************************************************************************************************************/

LV2_SYMBOL_EXPORT
const LV2_Descriptor* lv2_descriptor(uint32_t index)
{
    if (index == 0) return &Descriptor;
    else return NULL;
}

/**********************************************************************************************************************************************************/

// Apply a gain ramp to a buffer
static void applyGainRamp(ExponentialValueSmoother& smoother, float *out, const float *in, uint32_t n_samples) {
    for(uint32_t i=0; i<n_samples; i++) {
        out[i] = in[i] * smoother.next();
    }
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::applyBiquadFilter(float *out, const float *in, Biquad *filter, uint32_t n_samples) {
    for(uint32_t i=0; i<n_samples; i++) {
        out[i] = filter->process(in[i]);
    }
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::applyToneControls(float *out, const float *in, LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    float bass_boost_db = *self->bass_boost_db;
    float bass_freq = *self->bass_freq;
    uint8_t bass_has_changed = 0;
    float mid_boost_db = *self->mid_boost_db;
    float mid_freq = *self->mid_freq;
    float mid_q = *self->mid_q;
    float mid_type = *self->mid_type;
    uint8_t mid_has_changed = 0;
    float treble_boost_db = *self->treble_boost_db;
    float treble_freq = *self->treble_freq;
    uint8_t treble_has_changed = 0;
    float depth_boost_db = *self->depth_boost_db;
    float presence_boost_db = *self->presence_boost_db;

    /* Bass */
    if (bass_boost_db != self->bass_boost_db_old) {
        self->bass_boost_db_old = bass_boost_db;
        bass_has_changed++;
    }
    if (bass_freq != self->bass_freq_old) {
        self->bass_freq_old = bass_freq;
        bass_has_changed++;
    }
    if (bass_has_changed) {
        self->bass->setBiquad(bq_type_lowshelf, bass_freq / self->samplerate, 0.707f, bass_boost_db);
    }

    /* Mid */
    if (mid_boost_db != self->mid_boost_db_old) {
        self->mid_boost_db_old = mid_boost_db;
        mid_has_changed++;
    }
    if (mid_freq != self->mid_freq_old) {
        self->mid_freq_old = mid_freq;
        mid_has_changed++;
    }
    if (mid_q != self->mid_q_old) {
        self->mid_q_old = mid_q;
        mid_has_changed++;
    }
    if (mid_type !=self->mid_type_old) {
        self->mid_type_old = mid_type;
        mid_has_changed++;
    }
    if (mid_has_changed) {
        if(mid_type == BANDPASS) {
            self->mid->setBiquad(bq_type_bandpass, mid_freq / self->samplerate, mid_q, mid_boost_db);
        }
        else {
            self->mid->setBiquad(bq_type_peak, mid_freq / self->samplerate, mid_q, mid_boost_db);
        }
    }

    /* Treble */
    if (treble_boost_db != self->treble_boost_db_old) {
        self->treble_boost_db_old = treble_boost_db;
        treble_has_changed++;
    }
    if (treble_freq != self->treble_freq_old) {
        self->treble_freq_old = treble_freq;
        treble_has_changed++;
    }
    if (treble_has_changed) {
        self->treble->setBiquad(bq_type_highshelf, treble_freq / self->samplerate, 0.707f, treble_boost_db);
    }

    /* Depth & Presence */
    if(depth_boost_db != self->depth_boost_db_old) {
        self->depth_boost_db_old = depth_boost_db;
        self->depth->setBiquad(bq_type_peak, DEPTH_FREQ / self->samplerate, DEPTH_Q, depth_boost_db);
    }
    if(presence_boost_db != self->presence_boost_db_old) {
        self->presence_boost_db_old = presence_boost_db;
        self->presence->setBiquad(bq_type_highshelf, PRESENCE_FREQ / self->samplerate, PRESENCE_Q, presence_boost_db);
    }

    /* Run biquad cascade filters */
    if(mid_type == BANDPASS) {
        applyBiquadFilter(out, in, self->mid, n_samples);
    }
    else {
        applyBiquadFilter(out, in, self->depth, n_samples);
        applyBiquadFilter(out, out, self->bass, n_samples);
        applyBiquadFilter(out, out, self->mid, n_samples);
        applyBiquadFilter(out, out, self->treble, n_samples);
        applyBiquadFilter(out, out, self->presence, n_samples);
    }
}

/**********************************************************************************************************************************************************/

/**
 * This function carries model calculations for snapshot models, models with one parameter and
 * models with two parameters.
 */
void RtNeuralGeneric::applyModel(DynamicModel* model, float* out, uint32_t n_samples)
{
    const bool input_skip = model->input_skip;
    const float input_gain = model->input_gain;
    const float output_gain = model->output_gain;
#if AIDADSP_CONDITIONED_MODELS
    LinearValueSmoother& param1Coeff = model->param1Coeff;
    LinearValueSmoother& param2Coeff = model->param2Coeff;
#endif

    std::visit (
        [input_skip, &out, n_samples, input_gain, output_gain
#if AIDADSP_CONDITIONED_MODELS
        , &param1Coeff, &param2Coeff
#endif
        ] (auto&& custom_model)
        {
            using ModelType = std::decay_t<decltype (custom_model)>;
            if constexpr (ModelType::input_size == 1)
            {
                if (input_skip)
                {
                    for (uint32_t i=0; i<n_samples; ++i) {
                        out[i] *= input_gain;
                        out[i] += custom_model.forward (out + i);
                        out[i] *= output_gain;
                    }
                }
                else
                {
                    for (uint32_t i=0; i<n_samples; ++i) {
                        out[i] *= input_gain;
                        out[i] = custom_model.forward (out + i);
                        out[i] *= output_gain;
                    }
                }
            }
#if AIDADSP_CONDITIONED_MODELS
            else if constexpr (ModelType::input_size == 2)
            {
                float inArray1 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[2] = { 0.0, 0.0 };
                if (input_skip)
                {
                    for (uint32_t i=0; i<n_samples; ++i) {
                        out[i] *= input_gain;
                        inArray1[0] = out[i];
                        inArray1[1] = param1Coeff.next();
                        out[i] += custom_model.forward (inArray1);
                        out[i] *= output_gain;
                    }
                }
                else
                {
                    for (uint32_t i=0; i<n_samples; ++i) {
                        out[i] *= input_gain;
                        inArray1[0] = out[i];
                        inArray1[1] = param1Coeff.next();
                        out[i] = custom_model.forward (inArray1);
                        out[i] *= output_gain;
                    }
                }
            }
            else if constexpr (ModelType::input_size == 3)
            {
                float inArray2 alignas(RTNEURAL_DEFAULT_ALIGNMENT)[3] = { 0.0, 0.0, 0.0 };
                if (input_skip)
                {
                    for (uint32_t i=0; i<n_samples; ++i) {
                        out[i] *= input_gain;
                        inArray2[0] = out[i];
                        inArray2[1] = param1Coeff.next();
                        inArray2[2] = param2Coeff.next();
                        out[i] += custom_model.forward (inArray2);
                        out[i] *= output_gain;
                    }
                }
                else
                {
                    for (uint32_t i=0; i<n_samples; ++i) {
                        out[i] *= input_gain;
                        inArray2[0] = out[i];
                        inArray2[1] = param1Coeff.next();
                        inArray2[2] = param2Coeff.next();
                        out[i] = custom_model.forward (inArray2);
                        out[i] *= output_gain;
                    }
                }
            }
#endif
        },
        model->variant
    );
}

/**********************************************************************************************************************************************************/

LV2_Handle RtNeuralGeneric::instantiate(const LV2_Descriptor* descriptor, double samplerate, const char* bundle_path, const LV2_Feature* const* features)
{
    RtNeuralGeneric *self = new RtNeuralGeneric();

    self->samplerate = samplerate;

#if AIDADSP_COMMERCIAL
    self->run_count = 0;
    mod_license_check(features, PLUGIN_URI);
#endif

    // Get host features
    for (int i = 0; features[i]; ++i) {
        if (!strcmp(features[i]->URI, LV2_URID__map)) {
            self->map = (LV2_URID_Map*)features[i]->data;
        } else if (!strcmp(features[i]->URI, LV2_WORKER__schedule)) {
            self->schedule = (LV2_Worker_Schedule*)features[i]->data;
        } else if (!strcmp(features[i]->URI, LV2_LOG__log)) {
            self->log = (LV2_Log_Log*)features[i]->data;
        }
    }
    if (!self->map) {
        std::cout << "Error! Missing feature urid:map " << __func__ << " " << __LINE__ << std::endl;
        delete self;
        return 0;
    } else if (!self->schedule) {
        std::cout << "Error! Missing feature work:schedule " << __func__ << " " << __LINE__ << std::endl;
        delete self;
        return 0;
    }

    // Map URIs and initialize forge
    map_plugin_uris(self->map, &self->uris);
    lv2_log_logger_init(&self->logger, self->map, self->log);
#if AIDADSP_MODEL_LOADER
    lv2_atom_forge_init(&self->forge, self->map);
#endif

    // Setup initial values
    self->preGain.setSampleRate(self->samplerate);
    self->preGain.setTimeConstant(0.1f);
    self->preGain.setTargetValue(1.f);
    self->preGain.clearToTargetValue();
    self->masterGain.setSampleRate(self->samplerate);
    self->masterGain.setTimeConstant(0.1f);
    self->masterGain.setTargetValue(1.f);
    self->masterGain.clearToTargetValue();

    // Setup fixed frequency dc blocker filter (high pass)
    self->dc_blocker = new Biquad(bq_type_highpass, 35.0f / samplerate, 0.707f, 0.0f);

    // Setup variable high frequencies roll-off filter (low pass)
    self->in_lpf_pc_old = 66.216f;
    self->in_lpf = new Biquad(bq_type_lowpass, MAP(self->in_lpf_pc_old, 0.0f, 100.0f, INLPF_MAX_CO, INLPF_MIN_CO), 0.707f, 0.0f);

    // Setup equalizer section
    self->bass_boost_db_old = 0.0f;
    self->bass_freq_old = 250.0f;
    self->bass = new Biquad(bq_type_lowshelf, self->bass_freq_old / samplerate, 0.707f, self->bass_boost_db_old);
    self->mid_boost_db_old = 0.0f;
    self->mid_freq_old = 600.0f;
    self->mid_q_old = 0.707f;
    self->mid_type_old = 0.0f;
    self->mid = new Biquad(bq_type_peak, self->mid_freq_old / samplerate, self->mid_q_old, self->mid_boost_db_old);
    self->treble_boost_db_old = 0.0f;
    self->treble_freq_old = 1500.0f;
    self->treble = new Biquad(bq_type_highshelf, self->treble_freq_old / samplerate, 0.707f, self->treble_boost_db_old);
    self->depth_boost_db_old = 0.0f;
    self->depth = new Biquad(bq_type_peak, DEPTH_FREQ / samplerate, DEPTH_Q, self->depth_boost_db_old);
    self->presence_boost_db_old = 0.0f;
    self->presence = new Biquad(bq_type_highshelf, PRESENCE_FREQ / samplerate, PRESENCE_Q, self->presence_boost_db_old);

    self->last_input_size = 0;

    self->loading = false;

#if AIDADSP_MODEL_LOADER
    // initial model triggered by host default state load later on
    self->model = nullptr;
#else
    // start with 1st model loaded
    self->model_index_old = 0.0f;
    self->model = loadModelFromIndex(&self->logger, 1, &self->last_input_size);
#endif

    return (LV2_Handle)self;
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::activate(LV2_Handle instance)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;

    self->preGain.clearToTargetValue();
    self->masterGain.clearToTargetValue();

    if (self->model == nullptr)
        return;

    // @TODO: include the activate function code here
    // @TODO: if (self->samplerate != self->model->samplerate) ???
#if AIDADSP_CONDITIONED_MODELS
    self->model->paramFirstRun = true;
#endif
    std::visit (
        [] (auto&& custom_model)
        {
            using ModelType = std::decay_t<decltype (custom_model)>;
            if constexpr (! std::is_same_v<ModelType, NullModel>)
            {
                custom_model.reset();
            }
        },
        self->model->variant);
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::deactivate(LV2_Handle instance)
{
    // @TODO: include the deactivate function code here
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::connect_port(LV2_Handle instance, uint32_t port, void *data)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;

    switch((ports_t)port)
    {
        case IN:
            self->in = (float*) data;
            break;
        case OUT_1:
            self->out_1 = (float*) data;
            break;
        case PREGAIN:
            self->pregain_db = (float*) data;
            break;
#if AIDADSP_CONDITIONED_MODELS
        case PARAM1:
            self->param1 = (float*) data;
            break;
        case PARAM2:
            self->param2 = (float*) data;
            break;
#endif
        case MASTER:
            self->master_db = (float*) data;
            break;
        case NET_BYPASS:
            self->net_bypass = (float*) data;
            break;
#if AIDADSP_MODEL_LOADER
        case PLUGIN_CONTROL:
            self->control_port = (const LV2_Atom_Sequence*)data;
            break;
        case PLUGIN_NOTIFY:
            self->notify_port = (LV2_Atom_Sequence*)data;
            break;
#else
        case PLUGIN_MODEL_INDEX:
            self->model_index = (float*)data;
            break;
#endif
        case IN_LPF:
            self->in_lpf_pc = (float*) data;
            break;
        case EQ_POS:
            self->eq_position = (float*) data;
            break;
        case BASS:
            self->bass_boost_db = (float*) data;
            break;
        case BFREQ:
            self->bass_freq = (float*) data;
            break;
        case MID:
            self->mid_boost_db = (float*) data;
            break;
        case MFREQ:
            self->mid_freq = (float*) data;
            break;
        case MIDQ:
            self->mid_q = (float*) data;
            break;
        case MTYPE:
            self->mid_type = (float*) data;
            break;
        case TREBLE:
            self->treble_boost_db = (float*) data;
            break;
        case TFREQ:
            self->treble_freq = (float*) data;
            break;
        case DEPTH:
            self->depth_boost_db = (float*) data;
            break;
        case PRESENCE:
            self->presence_boost_db = (float*) data;
            break;
        case EQ_BYPASS:
            self->eq_bypass = (float*) data;
            break;
#if AIDADSP_OPTIONAL_DCBLOCKER
        case DCBLOCKER:
            self->dc_blocker_param = (float*) data;
            break;
#endif
        case INPUT_SIZE:
            self->input_size = (float*) data;
            break;
        case PLUGIN_ENABLED:
            self->enabled = (float*) data;
            break;
    }
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::run(LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    PluginURIs* uris   = &self->uris;

    const float pregain = DB_CO(*self->pregain_db);
    const float master = DB_CO(*self->master_db);
#if AIDADSP_CONDITIONED_MODELS
    const bool net_bypass = *self->net_bypass > 0.5f;
    const float in_lpf_pc = *self->in_lpf_pc;
    const float eq_position = *self->eq_position;
    const float eq_bypass = *self->eq_bypass;
    const bool enabled = *self->enabled > 0.5f;
#if AIDADSP_CONDITIONED_MODELS && (AIDADSP_PARAMS == 1)
    const float param1 = *self->param1;
    const float param2 = 0.f;
#elif AIDADSP_CONDITIONED_MODELS && (AIDADSP_PARAMS == 2)
    const float param1 = *self->param1;
    const float param2 = *self->param2;
#endif

    self->preGain.setTargetValue(pregain);
    if (in_lpf_pc != self->in_lpf_pc_old) { /* Update filter coeffs */
        self->in_lpf->setBiquad(bq_type_lowpass, MAP(in_lpf_pc, 0.0f, 100.0f, INLPF_MAX_CO, INLPF_MIN_CO), 0.707f, 0.0f);
        self->in_lpf_pc_old = in_lpf_pc;
    }
    *self->input_size = self->last_input_size;

#if AIDADSP_COMMERCIAL
    self->run_count = mod_license_run_begin(self->run_count, n_samples);
#endif

#if AIDADSP_MODEL_LOADER
#ifdef PROCESS_ATOM_MESSAGES
    /*++++++++ READ ATOM MESSAGES ++++++++*/
    // Set up forge to write directly to notify output port.
    const uint32_t notify_capacity = self->notify_port->atom.size;
    lv2_atom_forge_set_buffer(&self->forge,
            (uint8_t*)self->notify_port,
            notify_capacity);

    // Start a sequence in the notify output port.
    lv2_atom_forge_sequence_head(&self->forge, &self->notify_frame, 0);

    // Read incoming events
    LV2_ATOM_SEQUENCE_FOREACH(self->control_port, ev) {
        if (lv2_atom_forge_is_object_type(&self->forge, ev->body.type)) {
            const LV2_Atom_Object* obj = (const LV2_Atom_Object*)&ev->body;
            if (obj->body.otype == uris->patch_Set) {
                // Get the property and value of the set message
                const LV2_Atom* property = NULL;
                const LV2_Atom* value    = NULL;
                lv2_atom_object_get(obj,
                        uris->patch_property, &property,
                        uris->patch_value,    &value,
                        0);
                if (!property) {
                    lv2_log_trace(&self->logger,
                        "patch:Set message with no property\n");
                    continue;
                } else if (property->type != uris->atom_URID) {
                    lv2_log_trace(&self->logger,
                        "patch:Set property is not a URID\n");
                    continue;
                } else if (((const LV2_Atom_URID*)property)->body != uris->json) {
                    lv2_log_trace(&self->logger,
                        "patch:Set property body is not json\n");
                    continue;
                }
                if (!value) {
                    lv2_log_trace(&self->logger,
                        "patch:Set message with no value\n");
                    continue;
                } else if (value->type != uris->atom_Path) {
                    lv2_log_trace(&self->logger,
                        "patch:Set value is not a Path\n");
                    continue;
                }

                // Json model file change, send it to the worker.
                lv2_log_trace(&self->logger, "Queueing set message\n");
                WorkerLoadMessage msg = { kWorkerLoad, {} };
                std::memcpy(msg.path, value + 1, std::min(value->size, static_cast<uint32_t>(sizeof(msg.path) - 1u)));
                self->schedule->schedule_work(self->schedule->handle, sizeof(msg), &msg);
                self->loading = true;
            } else {
                lv2_log_trace(&self->logger,
                    "Unknown object type %d\n", obj->body.otype);
            }
        } else {
            lv2_log_trace(&self->logger,
                "Unknown event type %d\n", ev->body.type);
        }
    }
    /*++++++++ END READ ATOM MESSAGES ++++++++*/
#endif
#else
    float model_index = *self->model_index;

    if (model_index != self->model_index_old) {
        self->model_index_old = model_index;

        // Json model file change, send it to the worker.
        lv2_log_trace(&self->logger, "Queueing set message\n");
        WorkerLoadMessage msg = { kWorkerLoad, static_cast<int>(model_index + 1.5f) }; // round to int + 1
        self->schedule->schedule_work(self->schedule->handle, sizeof(msg), &msg);
        self->loading = true;
    }
#endif

    // 0 samples means pre-run, nothing left for us to do
    if (n_samples == 0) {
        return;
    }

    // not enabled (bypass)
    if (!enabled) {
        if (self->out_1 != self->in)
            std::memcpy(self->out_1, self->in, sizeof(float)*n_samples);
#if AIDADSP_COMMERCIAL
        mod_license_run_silence(self->run_count, self->out_1, n_samples, 0);
#endif
        return;
    }

    /*++++++++ AUDIO DSP ++++++++*/
    if (in_lpf_pc != 0.0f) {
        applyBiquadFilter(self->out_1, self->in, self->in_lpf, n_samples); // High frequencies roll-off (lowpass)
    } else {
        std::memcpy(self->out_1, self->in, sizeof(float)*n_samples);
    }
    applyGainRamp(self->preGain, self->out_1, self->out_1, n_samples); // Pre-gain
    if(eq_position == 1.0f && eq_bypass == 0.0f) {
        applyToneControls(self->out_1, self->out_1, instance, n_samples); // Equalizer section
    }
    if (self->model != nullptr) {
        if (!net_bypass) {
#if AIDADSP_CONDITIONED_MODELS
            self->model->param1Coeff.setTargetValue(param1);
            self->model->param2Coeff.setTargetValue(param2);
            if (self->model->paramFirstRun) {
                self->model->paramFirstRun = false;
                self->model->param1Coeff.clearToTargetValue();
                self->model->param2Coeff.clearToTargetValue();
                //lv2_log_trace(&self->logger, "n: %.02f\n", self->model->param1Coeff.next());
                printf("param1: %.02f\n", param1);
                printf("next: %.02f\n", self->model->param1Coeff.next());
                self->model->param1Coeff.dump();
            }
#endif
            applyModel(self->model, self->out_1, n_samples);
        }
    }
#if AIDADSP_OPTIONAL_DCBLOCKER
    if (*self->dc_blocker_param == 1.0f)
#endif
    {
        applyBiquadFilter(self->out_1, self->out_1, self->dc_blocker, n_samples); // Dc blocker filter (highpass)
    }
    if(eq_position == 0.0f && eq_bypass == 0.0f) {
        applyToneControls(self->out_1, self->out_1, instance, n_samples); // Equalizer section
    }
    self->masterGain.setTargetValue(self->loading ? 0.f : master);
    applyGainRamp(self->masterGain, self->out_1, self->out_1, n_samples); // Master volume
#if AIDADSP_COMMERCIAL
    mod_license_run_silence(self->run_count, self->out_1, n_samples, 0);
#endif
    /*++++++++ END AUDIO DSP ++++++++*/
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::cleanup(LV2_Handle instance)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;

    freeModel (self->model);
    delete self->dc_blocker;
    delete self->in_lpf;
    delete self->bass;
    delete self->mid;
    delete self->treble;
    delete self->depth;
    delete self->presence;
    delete self;
}

/**********************************************************************************************************************************************************/

const void* RtNeuralGeneric::extension_data(const char* uri)
{
#if AIDADSP_MODEL_LOADER
    static const LV2_State_Interface state = { save, restore };
    if (!strcmp(uri, LV2_STATE__interface)) {
        return &state;
    }
#endif
    static const LV2_Worker_Interface worker = { work, work_response, NULL };
    if (!strcmp(uri, LV2_WORKER__interface)) {
        return &worker;
    }
#if AIDADSP_COMMERCIAL
    return mod_license_interface(uri);
#else
    return NULL;
#endif
}

/**********************************************************************************************************************************************************/

#if AIDADSP_MODEL_LOADER
/**
 * This function is invoked during startup, after RtNeuralGeneric::instantiate
 * or whenever a state is restored
 */
LV2_State_Status RtNeuralGeneric::restore(LV2_Handle instance,
    LV2_State_Retrieve_Function retrieve,
    LV2_State_Handle            handle,
    uint32_t                    flags,
    const LV2_Feature* const*   features)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;

    size_t   size;
    uint32_t type;
    uint32_t valflags;
    int      res;

    const void* value = retrieve(
            handle,
            self->uris.json,
            &size, &type, &valflags);

    if (value) {
        lv2_log_note(&self->logger, "Restoring file %s\n", (const char*)value);

        // send to worker for loading
        WorkerLoadMessage msg = { kWorkerLoad, {} };

        LV2_State_Map_Path* map_path = NULL;
        for (int i = 0; features[i]; ++i) {
            if (!strcmp(features[i]->URI, LV2_STATE__mapPath)) {
                map_path = (LV2_State_Map_Path*)features[i]->data;
            }
        }

        if (map_path) {
            char* apath = map_path->absolute_path(map_path->handle, (const char*)value);
            std::memcpy(msg.path, apath, std::min(size, strlen(msg.path) - 1u));
            free(apath);
        } else {
            std::memcpy(msg.path, value, std::min(size, sizeof(msg.path) - 1u));
        }

        self->schedule->schedule_work(self->schedule->handle, sizeof(msg), &msg);
    }

    return LV2_STATE_SUCCESS;
}

/**********************************************************************************************************************************************************/

LV2_State_Status RtNeuralGeneric::save(LV2_Handle instance,
    LV2_State_Store_Function  store,
    LV2_State_Handle          handle,
    uint32_t                  flags,
    const LV2_Feature* const* features)
{
    RtNeuralGeneric* self = (RtNeuralGeneric*) instance;

    // nothing loaded yet
    if (!self->model) {
        return LV2_STATE_SUCCESS;
    }

    LV2_State_Map_Path* map_path = NULL;
    for (int i = 0; features[i]; ++i) {
        if (!strcmp(features[i]->URI, LV2_STATE__mapPath)) {
            map_path = (LV2_State_Map_Path*)features[i]->data;
        }
    }

    if (map_path) {
        char* apath = map_path->abstract_path(map_path->handle, self->model->path);
        store(handle,
                self->uris.json,
                apath,
                strlen(apath) + 1,
                self->uris.atom_Path,
                LV2_STATE_IS_POD | LV2_STATE_IS_PORTABLE);
        free(apath);
        return LV2_STATE_SUCCESS;
    } else {
        return LV2_STATE_ERR_NO_FEATURE;
    }
}
#endif

/**********************************************************************************************************************************************************/

/**
 * Do work in a non-realtime thread.
 * This is called for every piece of work scheduled in the audio thread using
 * self->schedule->schedule_work(). A reply can be sent back to the audio
 * thread using the provided respond function.
 */
LV2_Worker_Status RtNeuralGeneric::work(LV2_Handle instance,
    LV2_Worker_Respond_Function respond,
    LV2_Worker_Respond_Handle   handle,
    uint32_t                    size,
    const void*                 data)
{
    RtNeuralGeneric* self = (RtNeuralGeneric*) instance;
    const WorkerMessage* msg = (const WorkerMessage*)data;

    switch (msg->type)
    {
    case kWorkerLoad:
#if AIDADSP_MODEL_LOADER
        if (DynamicModel* newmodel = RtNeuralGeneric::loadModelFromPath(&self->logger, ((const WorkerLoadMessage*)data)->path, &self->last_input_size))
#else
        if (DynamicModel* newmodel = RtNeuralGeneric::loadModelFromIndex(&self->logger, ((const WorkerLoadMessage*)data)->modelIndex, &self->last_input_size))
#endif
        {
            WorkerApplyMessage reply = { kWorkerApply, newmodel };
            respond (handle, sizeof(reply), &reply);
        }
        return LV2_WORKER_SUCCESS;

    case kWorkerFree:
        freeModel (((const WorkerApplyMessage*)data)->model);
        return LV2_WORKER_SUCCESS;

    case kWorkerApply:
        // should not happen!
        break;
    }

    return LV2_WORKER_ERR_UNKNOWN;
}

/**********************************************************************************************************************************************************/

/**
 * Handle a response from work() in the audio thread.
 *
 * When running normally, this will be called by the host after run().  When
 * freewheeling, this will be called immediately at the point the work was
 * scheduled.
*/
LV2_Worker_Status RtNeuralGeneric::work_response(LV2_Handle instance, uint32_t size, const void* data)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;

    const WorkerMessage* const msg = static_cast<const WorkerMessage*>(data);

    if (msg->type != kWorkerApply)
        return LV2_WORKER_ERR_UNKNOWN;

    // prepare reply for deleting old model
    WorkerApplyMessage reply = { kWorkerFree, self->model };

    // swap current model with new one
    self->model = static_cast<const WorkerApplyMessage*>(data)->model;

    // send reply
    self->schedule->schedule_work(self->schedule->handle, sizeof(reply), &reply);

    // log about new model in use
    lv2_log_trace(&self->logger, "New model in use\n");

#if AIDADSP_MODEL_LOADER
    // report change to host/ui
    lv2_atom_forge_frame_time(&self->forge, 0);
    write_set_file(&self->forge,
                   &self->uris,
                   self->model->path,
                   strlen(self->model->path));
#endif

    self->loading = false;
    lv2_log_trace(&self->logger, "loading = false\n");

    return LV2_WORKER_SUCCESS;
}

/**********************************************************************************************************************************************************/

/**
 * This function tests the inference engine
*/
bool RtNeuralGeneric::testModel(LV2_Log_Logger* logger, DynamicModel *model, const std::vector<float>& xData, const std::vector<float>& yData)
{
    std::unique_ptr<float[]> out(new float [xData.size()]);
    float in_gain = model->input_gain;
    float out_gain = model->output_gain;
    /* Gain correction inject unwanted errors */
    model->input_gain = 1.f;
    model->output_gain = 1.f;
#if AIDADSP_CONDITIONED_MODELS
    /* Conditioned models tested with all params at 0 */
    float param1 = model->param1Coeff.getTargetValue();
    float param2 = model->param2Coeff.getTargetValue();
    model->param1Coeff.setTargetValue(0.f);
    model->param1Coeff.clearToTargetValue();
    model->param2Coeff.setTargetValue(0.f);
    model->param2Coeff.clearToTargetValue();
#endif
    for(size_t i = 0; i < xData.size(); i++) {
        out[i] = xData[i];
    }
    applyModel(model, out.get(), xData.size());
    /* Restore params previously saved */
    model->input_gain = in_gain;
    model->output_gain = out_gain;
#if AIDADSP_CONDITIONED_MODELS
    model->param1Coeff.setTargetValue(param1);
    model->param1Coeff.clearToTargetValue();
    model->param2Coeff.setTargetValue(param2);
    model->param2Coeff.clearToTargetValue();
#endif
    constexpr double threshold = TEST_MODEL_THR;
    size_t nErrs = 0;
    float max_error = 0.0f;
    std::vector<float> inputErrors;
    for(size_t i = 0; i < xData.size(); i++) {
        auto err = std::abs(out[i] - yData[i]);
        max_error = std::max(err, max_error);
        if(err > threshold) {
            nErrs++;
            inputErrors.push_back(std::abs(xData[i]));
        }
    }
    if(nErrs > 0)
    {
        lv2_log_trace(logger, "Failure %s: %d errs!\n", __func__, (int)nErrs);
        lv2_log_trace(logger, "Max err: %.12f, thr: %.12f\n", max_error, threshold);
        lv2_log_trace(logger, "< %.6f [dB]\n", CO_DB(*std::max_element(inputErrors.begin(), inputErrors.end())));
    }
    else
    {
        lv2_log_trace(logger, "Success %s: %d errs!\n", __func__, (int)nErrs);
        lv2_log_trace(logger, "Max err: %.12f, thr: %.12f\n", max_error, threshold);
        return true;
    }
    return false;
}

/**********************************************************************************************************************************************************/

#if AIDADSP_MODEL_LOADER
/**
 * This function loads a pre-trained neural model from a json file
*/
DynamicModel* RtNeuralGeneric::loadModelFromPath(LV2_Log_Logger* logger, const char* path, int* input_size_ptr)
{
    int input_skip;
    int input_size;
    float input_gain;
    float output_gain;
    float model_samplerate;
    nlohmann::json model_json;

    try {
        std::ifstream jsonStream(path, std::ifstream::binary);
        jsonStream >> model_json;

        /* Understand which model type to load */
        input_size = model_json["in_shape"].back().get<int>();
        if (input_size > MAX_INPUT_SIZE) {
            throw std::invalid_argument("Value for input_size not supported");
        }

        if (model_json["in_skip"].is_number()) {
            input_skip = model_json["in_skip"].get<int>();
            if (input_skip > 1)
                throw std::invalid_argument("Values for in_skip > 1 are not supported");
        }
        else {
            input_skip = 0;
        }

        if (model_json["in_gain"].is_number()) {
            input_gain = DB_CO(model_json["in_gain"].get<float>());
        }
        else {
            input_gain = 1.0f;
        }

        if (model_json["out_gain"].is_number()) {
            output_gain = DB_CO(model_json["out_gain"].get<float>());
        }
        else {
            output_gain = 1.0f;
        }

        if (model_json["metadata"]["samplerate"].is_number()) {
            model_samplerate = model_json["metadata"]["samplerate"].get<float>();
        }
        else if (model_json["samplerate"].is_number()) {
            model_samplerate = model_json["samplerate"].get<float>();
        }
        else {
            model_samplerate = 48000.0f;
        }

        lv2_log_note(logger, "Successfully loaded json file: %s\n", path);
    }
    catch (const std::exception& e) {
        lv2_log_error(logger, "Unable to load json file: %s\nError: %s\n", path, e.what());
        return nullptr;
    }

    std::unique_ptr<DynamicModel> model = std::make_unique<DynamicModel>();

    try {
        if (! custom_model_creator (model_json, model->variant))
            throw std::runtime_error ("Unable to identify a known model architecture!");

        std::visit (
            [&model_json] (auto&& custom_model)
            {
                using ModelType = std::decay_t<decltype (custom_model)>;
                if constexpr (! std::is_same_v<ModelType, NullModel>)
                {
                    custom_model.parseJson (model_json, true);
                    custom_model.reset();
                }
            },
            model->variant);
        lv2_log_note(logger, "%s %d: mdl rst!\n", __func__, __LINE__);
    }
    catch (const std::exception& e) {
        lv2_log_error(logger, "Error loading model: %s\n", e.what());
        return nullptr;
    }

    /* Save extra info */
    model->path = strdup(path);
    model->input_skip = input_skip != 0;
    model->input_gain = input_gain;
    model->output_gain = output_gain;
    model->samplerate = model_samplerate;
#if AIDADSP_CONDITIONED_MODELS
    model->param1Coeff.setSampleRate(model_samplerate);
    model->param1Coeff.setTimeConstant(0.1f);
    model->param2Coeff.setSampleRate(model_samplerate);
    model->param2Coeff.setTimeConstant(0.1f);
    model->paramFirstRun = true;
#endif

    /* Sanity check on inference engine with loaded model, also serves as pre-buffer
    * to avoid "clicks" during initialization */
    if (model_json["input_batch"].is_array() && model_json["input_batch"].is_array()) {
        std::vector<float> input_batch = model_json["/input_batch"_json_pointer];
        std::vector<float> output_batch = model_json["/output_batch"_json_pointer];
        testModel(logger, model.get(), input_batch, output_batch);
    }
    else
    {
        float out[2048] = {};
        applyModel(model.get(), out, 2048);
    }

    // cache input size for later
    *input_size_ptr = input_size;

    return model.release();
}
#endif

/**********************************************************************************************************************************************************/

/**
 * This function deletes a model instance and its related details
*/
void RtNeuralGeneric::freeModel(DynamicModel* model)
{
    if (model == nullptr)
        return;
#if AIDADSP_MODEL_LOADER
    free (model->path);
#endif
    delete model;
}
