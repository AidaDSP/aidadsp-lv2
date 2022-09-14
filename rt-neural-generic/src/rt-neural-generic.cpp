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
    return (start + ((end - start)/n_samples) * index);
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

/**
 * This function carries model calculations for snapshot models
 */
void RtNeuralGeneric::applyModel(float *out, const float *in, LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    uint32_t i;
    int skip = self->input_skip;
    switch(self->model_index)
    {
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
    self->inArray1[1] = param1;
    switch(self->model_index)
    {
        case LSTM_16:
            for(i=0; i<n_samples; i++) {
                self->inArray1[0] = in[i];
                out[i] = self->lstm_16.forward(self->inArray1) + (in[i] * skip);
            }
            break;
        case LSTM_12:
            for(i=0; i<n_samples; i++) {
                self->inArray1[0] = in[i];
                out[i] = self->lstm_12.forward(self->inArray1) + (in[i] * skip);
            }
            break;
        case GRU_12:
            for(i=0; i<n_samples; i++) {
                self->inArray1[0] = in[i];
                out[i] = self->gru_12.forward(self->inArray1) + (in[i] * skip);
            }
            break;
        case GRU_8:
            for(i=0; i<n_samples; i++) {
                self->inArray1[0] = in[i];
                out[i] = self->gru_8.forward(self->inArray1) + (in[i] * skip);
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
    self->inArray2[1] = param1;
    self->inArray2[2] = param2;
    switch(self->model_index)
    {
        case LSTM_16:
            for(i=0; i<n_samples; i++) {
                self->inArray2[0] = in[i];
                out[i] = self->lstm_16.forward(self->inArray2) + (in[i] * skip);
            }
            break;
        case LSTM_12:
            for(i=0; i<n_samples; i++) {
                self->inArray2[0] = in[i];
                out[i] = self->lstm_12.forward(self->inArray2) + (in[i] * skip);
            }
            break;
        case GRU_12:
            for(i=0; i<n_samples; i++) {
                self->inArray2[0] = in[i];
                out[i] = self->gru_12.forward(self->inArray2) + (in[i] * skip);
            }
            break;
        case GRU_8:
            for(i=0; i<n_samples; i++) {
                self->inArray2[0] = in[i];
                out[i] = self->gru_8.forward(self->inArray2) + (in[i] * skip);
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

    // Map URIs and initialise forge
    map_plugin_uris(self->map, &self->uris);
    lv2_atom_forge_init(&self->forge, self->map);
    lv2_log_logger_init(&self->logger, self->map, self->log);

    // Setup fixed frequency dc blocker filter (high pass)
    self->dc_blocker = new Biquad(bq_type_highpass, 35.0f / samplerate, 0.707, 0.0);

    // Setup variable high frequencies roll-off filter (low pass)
    self->in_lpf_f_old = 12000.0f;
    self->in_lpf = new Biquad(bq_type_lowpass, self->in_lpf_f_old / samplerate, 0.707, 0.0);

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

    switch (port)
    {
        case IN:
            self->in = (float*) data;
            break;
        case OUT_1:
            self->out_1 = (float*) data;
            break;
        case IN_VOL:
            self->in_vol_db = (float*) data;
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
        case BYPASS:
            self->bypass = (float*) data;
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
        case IN_LPF_BYP:
            self->in_lpf_bypass = (float*) data;
            break;
    }
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::run(LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;
    PluginURIs* uris   = &self->uris;

    float in_vol, in_vol_old;
    float param1 = *self->param1;
    float param2 = *self->param2;
    float master, master_old;
    float bypass = *self->bypass;
    float in_lpf_bypass = *self->in_lpf_bypass;
    float in_lpf_f, in_lpf_f_old;

    in_vol = DB_CO(*self->in_vol_db);
    in_vol_old = self->in_vol_old;
    self->in_vol_old = in_vol;

    master = DB_CO(*self->master_db);
    master_old = self->master_old;
    self->master_old = master;

    in_lpf_f = *self->in_lpf_f * 1000.0f;
    in_lpf_f_old = self->in_lpf_f_old;
    self->in_lpf_f_old = in_lpf_f;
    if ((in_lpf_f != in_lpf_f_old) && in_lpf_bypass != 1.0f) /* Update filter coeffs */
    {
        self->in_lpf->setBiquad(bq_type_lowpass, in_lpf_f / self->samplerate, 0.707, 0.0);
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
    if (bypass == 0.0f && self->model_loaded == 1) {
        if (in_lpf_bypass == 1.0f) {
            applyGainRamp(self->out_1, self->in, in_vol_old, in_vol, n_samples); // Input volume
        }
        else { // Apply lpf filter before input volume
            applyBiquadFilter(self->out_1, self->in, self->in_lpf, n_samples); // High frequencies roll-off (lowpass)
            applyGainRamp(self->out_1, self->out_1, in_vol_old, in_vol, n_samples); // Input volume
        }
        // Process model based on input_size (snapshot model or conditioned model)
        switch(self->input_size) {
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
        applyBiquadFilter(self->out_1, self->out_1, self->dc_blocker, n_samples); // Dc blocker filter (highpass)
        applyGainRamp(self->out_1, self->out_1, master_old, master, n_samples); // Master volume
    }
    else {
        std::copy(self->in, self->in + n_samples, self->out_1); // Passthrough
    }
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
            lv2_log_error(&self->logger, "File %s couldn't be loaded\n", path);
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
    if (!res) {
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

        if(self->type == std::string("lstm")) {
            if(self->hidden_size == 16) {
                self->model_index = LSTM_16;
            }
            else if(self->hidden_size == 12) {
                self->model_index = LSTM_12;
            }
        }
        else if(self->type == std::string("gru")) {
            if(self->hidden_size == 12) {
                self->model_index = GRU_12;
            }
            else if(self->hidden_size == 8) {
                self->model_index = GRU_8;
            }
        }

        std::ifstream jsonStream2(filePath, std::ifstream::binary);
        switch(self->model_index)
        {
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
