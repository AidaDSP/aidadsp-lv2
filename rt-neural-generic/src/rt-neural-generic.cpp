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

// Ramp calculation
float RtNeuralGeneric::rampValue(float start, float end, uint32_t n_samples, uint32_t index) {
    return (start + ((end - start)/n_samples) * (index+1));
}

/**********************************************************************************************************************************************************/

// Apply a gain ramp to a buffer
void RtNeuralGeneric::applyGainRamp(float *out, const float *in, float start, float end, uint32_t n_samples) {
    static uint32_t i;
    for(i=0; i<n_samples; i++) {
        out[i] = in[i] * rampValue(start, end, n_samples, i);
    }
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::applyBiquadFilter(float *out, const float *in, Biquad *filter, uint32_t n_samples) {
    static uint32_t i;
    for(i=0; i<n_samples; i++) {
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
        self->mid->setBiquad(bq_type_lowshelf, bass_freq / self->samplerate, 0.707f, bass_boost_db);
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
 * This function carries model calculations for snapshot models
 */
void RtNeuralGeneric::applyModel(float *out, const float *in, LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    uint32_t i;
    int skip = self->input_skip;
    switch((rnn_t)self->model_index)
    {
        case LSTM_40:
            for(i=0; i<n_samples; i++) {
                out[i] = self->lstm_40.forward(in + i) + (in[i] * skip);
            }
            break;
        case LSTM_20:
            for(i=0; i<n_samples; i++) {
                out[i] = self->lstm_20.forward(in + i) + (in[i] * skip);
            }
            break;
        case LSTM_16:
            for(i=0; i<n_samples; i++) {
                out[i] = self->lstm_16.forward(in + i) + (in[i] * skip);
            }
            break;
        case LSTM_12:
            for(i=0; i<n_samples; i++) {
                out[i] = self->lstm_12.forward(in + i) + (in[i] * skip);
            }
            break;
        case GRU_12:
            for(i=0; i<n_samples; i++) {
                out[i] = self->gru_12.forward(in + i) + (in[i] * skip);
            }
            break;
        case GRU_8:
            for(i=0; i<n_samples; i++) {
                out[i] = self->gru_8.forward(in + i) + (in[i] * skip);
            }
            break;
    }
}

/**********************************************************************************************************************************************************/

/**
 * This function carries model calculations for conditioned models with single param
 */
void RtNeuralGeneric::applyModel(float *out, const float *in, float param1, LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    uint32_t i;
    int skip = self->input_skip;
    switch((rnn_t)self->model_index)
    {
        case LSTM_40_cond1:
            for(i=0; i<n_samples; i++) {
                self->inArray1[0] = in[i];
                self->inArray1[1] = self->rampValue(self->inArray1[1], param1, n_samples, i);
                out[i] = self->lstm_40_cond1.forward(self->inArray1) + (in[i] * skip);
            }
            break;
    }
}

/**********************************************************************************************************************************************************/

/**
 * This function carries model calculations for conditioned models with two params
 */
void RtNeuralGeneric::applyModel(float *out, const float *in, float param1, float param2, LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    uint32_t i;
    int skip = self->input_skip;
    switch((rnn_t)self->model_index)
    {
        case LSTM_40_cond2:
            for(i=0; i<n_samples; i++) {
                self->inArray2[0] = in[i];
                self->inArray1[1] = self->rampValue(self->inArray1[1], param1, n_samples, i);
                self->inArray1[2] = self->rampValue(self->inArray1[2], param2, n_samples, i);
                out[i] = self->lstm_40.forward(self->inArray2) + (in[i] * skip);
            }
            break;
    }
}

/**********************************************************************************************************************************************************/

LV2_Handle RtNeuralGeneric::instantiate(const LV2_Descriptor* descriptor, double samplerate, const char* bundle_path, const LV2_Feature* const* features)
{
    RtNeuralGeneric *self = new RtNeuralGeneric();

    self->model_index = 0;
    self->samplerate = samplerate;

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
        free(self);
        return 0;
    } else if (!self->schedule) {
        std::cout << "Error! Missing feature work:schedule " << __func__ << " " << __LINE__ << std::endl;
        free(self);
        return 0;
    }

    // Map URIs and initialize forge
    map_plugin_uris(self->map, &self->uris);
    lv2_atom_forge_init(&self->forge, self->map);
    lv2_log_logger_init(&self->logger, self->map, self->log);

    // Setup initial values
    self->pregain_old = 1.0f;
    self->master_old = 1.0f;

    // Setup fixed frequency dc blocker filter (high pass)
    self->dc_blocker = new Biquad(bq_type_highpass, 35.0f / samplerate, 0.707f, 0.0f);

    // Setup variable high frequencies roll-off filter (low pass)
    self->in_lpf_f_old = samplerate / 4.0f;
    self->in_lpf = new Biquad(bq_type_lowpass, self->in_lpf_f_old / samplerate, 0.707f, 0.0f);

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

    /* Prevent audio thread to use the model */
    self->model_loaded = 0;

    /* No pending notifications */
    self->model_new = 0;

    self->path_len = 0;

    return (LV2_Handle)self;
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::activate(LV2_Handle instance)
{
    // TODO: include the activate function code here
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::deactivate(LV2_Handle instance)
{
    // TODO: include the deactivate function code here
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
        case PARAM1:
            self->param1 = (float*) data;
            break;
        case PARAM2:
            self->param2 = (float*) data;
            break;
        case MASTER:
            self->master_db = (float*) data;
            break;
        case NET_BYPASS:
            self->net_bypass = (float*) data;
            break;
        case PLUGIN_CONTROL:
            self->control_port = (const LV2_Atom_Sequence*)data;
            break;
        case PLUGIN_NOTIFY:
            self->notify_port = (LV2_Atom_Sequence*)data;
            break;
        case IN_LPF:
            self->in_lpf_f = (float*) data;
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
    }
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::run(LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    PluginURIs* uris   = &self->uris;

    float pregain = DB_CO(*self->pregain_db);
    float master = DB_CO(*self->master_db);
    float net_bypass = *self->net_bypass;
    float in_lpf_f = *self->in_lpf_f * 1000.0f;
    float eq_position = *self->eq_position;
    float eq_bypass = *self->eq_bypass;
    float param1 = *self->param1;
    float param2 = *self->param2;

    if (in_lpf_f != self->in_lpf_f_old) { /* Update filter coeffs */
        self->in_lpf->setBiquad(bq_type_lowpass, in_lpf_f / self->samplerate, 0.707f, 0.0f);
        self->in_lpf_f_old = in_lpf_f;
    }

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
        self->frame_offset = ev->time.frames;
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
                }

                const uint32_t key = ((const LV2_Atom_URID*)property)->body;
                if (key == uris->json) {
                    // Json model file change, send it to the worker.
                    lv2_log_trace(&self->logger, "Queueing set message\n");
                    self->model_loaded = 0; // Stop model access in dsp code below
                    self->schedule->schedule_work(self->schedule->handle,
                                                    lv2_atom_total_size(&ev->body),
                                                    &ev->body);
                }
            } else {
                lv2_log_trace(&self->logger,
                    "Unknown object type %d\n", obj->body.otype);
            }
        } else {
            lv2_log_trace(&self->logger,
                "Unknown event type %d\n", ev->body.type);
        }
    }

    if(self->model_new)
    {
        /* We send a notification we're using a new model */
        lv2_log_trace(&self->logger, "New model in use\n");
        lv2_atom_forge_frame_time(&self->forge, self->frame_offset);
        write_set_file(&self->forge, &self->uris,
                                        self->path,
                                        self->path_len);
        self->model_new = 0;
    }
    /*++++++++ END READ ATOM MESSAGES ++++++++*/
#endif

    /*++++++++ AUDIO DSP ++++++++*/
    applyBiquadFilter(self->out_1, self->in, self->in_lpf, n_samples); // High frequencies roll-off (lowpass)
    applyGainRamp(self->out_1, self->out_1, self->pregain_old, pregain, n_samples); // Pre-gain
    if(eq_position == 1.0f && eq_bypass == 0.0f) {
        applyToneControls(self->out_1, self->out_1, instance, n_samples); // Equalizer section
    }
    if (net_bypass == 0.0f && self->model_loaded) {
        switch(self->input_size) { // Process model based on input_size (snapshot model or conditioned model)
            case 1:
                applyModel(self->out_1, self->out_1, instance, n_samples);
                break;
            case 2:
                applyModel(self->out_1, self->out_1, param1, instance, n_samples);
                break;
            case 3:
                applyModel(self->out_1, self->out_1, param1, param2, instance, n_samples);
                break;
            default:
                break;
        }
    }
    applyBiquadFilter(self->out_1, self->out_1, self->dc_blocker, n_samples); // Dc blocker filter (highpass)
    if(eq_position == 0.0f && eq_bypass == 0.0f) {
        applyToneControls(self->out_1, self->out_1, instance, n_samples); // Equalizer section
    }
    applyGainRamp(self->out_1, self->out_1, self->master_old, master, n_samples); // Master volume
    self->pregain_old = pregain;
    self->master_old = master;
    /*++++++++ END AUDIO DSP ++++++++*/
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::cleanup(LV2_Handle instance)
{
    delete ((RtNeuralGeneric*) instance);
}

/**********************************************************************************************************************************************************/

const void* RtNeuralGeneric::extension_data(const char* uri)
{
    static const LV2_State_Interface state = { save, restore };
    static const LV2_Worker_Interface worker = { work, work_response, NULL };
    if (!strcmp(uri, LV2_STATE__interface)) {
        return &state;
    } else if (!strcmp(uri, LV2_WORKER__interface)) {
        return &worker;
    }
    return NULL;
}

/**********************************************************************************************************************************************************/

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
        const char* path = (const char*)value;
        lv2_log_note(&self->logger, "Restoring file %s\n", path);
        res = self->loadModel(instance, path);
        if (res) {
            return LV2_STATE_ERR_UNKNOWN;
        } else {
            self->model_loaded = 1; // Unlock model usage in dsp
        }
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
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;

    if (!self->model_loaded) {
        return LV2_STATE_SUCCESS;
    }

    LV2_State_Map_Path* map_path = NULL;
    for (int i = 0; features[i]; ++i) {
        if (!strcmp(features[i]->URI, LV2_STATE__mapPath)) {
            map_path = (LV2_State_Map_Path*)features[i]->data;
        }
    }

    if (map_path) {
        char* apath = map_path->abstract_path(map_path->handle, self->path);
        store(handle,
                self->uris.json,
                apath,
                strlen(self->path) + 1,
                self->uris.atom_Path,
                LV2_STATE_IS_POD | LV2_STATE_IS_PORTABLE);
        free(apath);
        return LV2_STATE_SUCCESS;
    } else {
        return LV2_STATE_ERR_NO_FEATURE;
    }
}

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
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    int res;

    const LV2_Atom* atom = (const LV2_Atom*)data;

    // Handle set message (load json).
    const LV2_Atom_Object* obj = (const LV2_Atom_Object*)data;

    // Get file path from message
    const LV2_Atom* file_path = read_set_file(&self->uris, obj);
    if (!file_path) {
        return LV2_WORKER_ERR_UNKNOWN;
    }

    res = self->loadModel(instance, (const char*)(LV2_ATOM_BODY_CONST(file_path)));
    if (res) {
        return LV2_WORKER_ERR_UNKNOWN;
    } else {
        // Model is ready, send response to run() to enable dsp.
        respond(handle, file_path->size, LV2_ATOM_BODY_CONST(file_path));
    }

    return LV2_WORKER_SUCCESS;
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

    self->model_loaded = 1; // Enable dsp in next run() iteration
    self->model_new = 1; // Pending notification

    return LV2_WORKER_SUCCESS;
}

/**********************************************************************************************************************************************************/

/**
 * This function loads a pre-trained neural model from a json file
*/
int RtNeuralGeneric::loadModel(LV2_Handle instance, const char *path)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;

    self->path = path;
    self->path_len = strlen(path);

    std::string filePath;

    filePath.append(path);

    lv2_log_note(&self->logger, "Loading json file: %s\n", path);

    try {
        std::ifstream jsonStream1(filePath, std::ifstream::binary);
        nlohmann::json modelData;
        jsonStream1 >> modelData;

        /* Understand which model type to load */
        self->n_layers = modelData["layers"].size();
        self->input_size = modelData["in_shape"].back().get<int>();
        if(self->input_size > 1) {
            throw std::invalid_argument("Values for input_size > 1 are not supported");
        }

        if (modelData["in_skip"].is_number()) {
            self->input_skip = modelData["in_skip"].get<int>();
            if (self->input_skip > 1)
                throw std::invalid_argument("Values for in_skip > 1 are not supported");
        }
        else {
            self->input_skip = 0;
        }

        self->type = modelData["layers"][self->n_layers-1-1]["type"];
        self->hidden_size = modelData["layers"][self->n_layers-1-1]["shape"].back().get<int>();

        self->model_index = -1;

        if(self->type == std::string("lstm")) {
            if(self->hidden_size == 40 && self->input_size == 1) {
                self->model_index = LSTM_40;
            }
            else if(self->hidden_size == 40 && self->input_size == 2) {
                self->model_index = LSTM_40_cond1;
            }
            else if(self->hidden_size == 40 && self->input_size == 3) {
                self->model_index = LSTM_40_cond2;
            }
            else if(self->hidden_size == 20 && self->input_size == 1) {
                self->model_index = LSTM_20;
            }
            else if(self->hidden_size == 16 && self->input_size == 1) {
                self->model_index = LSTM_16;
            }
            else if(self->hidden_size == 12 && self->input_size == 1) {
                self->model_index = LSTM_12;
            }
        }
        else if(self->type == std::string("gru")) {
            if(self->hidden_size == 12 && self->input_size == 1) {
                self->model_index = GRU_12;
            }
            else if(self->hidden_size == 8 && self->input_size == 1) {
                self->model_index = GRU_8;
            }
        }

        if(self->model_index < 0)
            throw std::invalid_argument( "Unsupported model type" );

        std::ifstream jsonStream2(filePath, std::ifstream::binary);
        switch((rnn_t)self->model_index)
        {
            case LSTM_40:
                self->lstm_40.parseJson(jsonStream2, true);
                break;
            case LSTM_40_cond1:
                self->lstm_40_cond1.parseJson(jsonStream2, true);
                break;
            case LSTM_40_cond2:
                self->lstm_40_cond2.parseJson(jsonStream2, true);
                break;
            case LSTM_20:
                self->lstm_20.parseJson(jsonStream2, true);
                break;
            case LSTM_16:
                self->lstm_16.parseJson(jsonStream2, true);
                break;
            case LSTM_12:
                self->lstm_12.parseJson(jsonStream2, true);
                break;
            case GRU_12:
                self->gru_12.parseJson(jsonStream2, true);
                break;
            case GRU_8:
                self->gru_8.parseJson(jsonStream2, true);
                break;
        }

        lv2_log_note(&self->logger, "Successfully loaded json file: %s\n", path);
    }
    catch (const std::exception& e) {
        lv2_log_error(&self->logger, "Unable to load json file: %s\nError: %s\n", path, e.what());
        return 1;
    }

    // Before running inference, it is recommended to "reset" the state
    // of your model (if the model has state).
    switch(self->model_index)
    {
        case LSTM_40:
            self->lstm_40.reset();
            break;
        case LSTM_40_cond1:
            self->lstm_40_cond1.reset();
            break;
        case LSTM_40_cond2:
            self->lstm_40_cond2.reset();
            break;
        case LSTM_20:
            self->lstm_20.reset();
            break;
        case LSTM_16:
            self->lstm_16.reset();
            break;
        case LSTM_12:
            self->lstm_12.reset();
            break;
        case GRU_12:
            self->gru_12.reset();
            break;
        case GRU_8:
            self->gru_8.reset();
            break;
    }

    // Pre-buffer to avoid "clicks" during initialization
    float in[2048] = { };
    float out[2048] = { };
    switch(self->input_size) {
        case 1:
            applyModel(out, in, self, 2048);
            break;
        case 2:
            applyModel(out, in, 0, self, 2048);
            break;
        case 3:
            applyModel(out, in, 0, 0, self, 2048);
            break;
        default:
            break;
    }

    return 0;
}
