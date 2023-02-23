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
void RtNeuralGeneric::applyModel(DynamicModel* model, float* out, uint32_t n_samples)
{
    const bool input_skip = model->input_skip;

    std::visit (
        [&input_skip, &out, &n_samples] (auto&& custom_model)
        {
            using ModelType = std::decay_t<decltype (custom_model)>;
            if constexpr (ModelType::input_size == 1)
            {
                if (input_skip)
                {
                    for (uint32_t i=0; i<n_samples; ++i)
                        out[i] = custom_model.forward (out + i);
                }
                else
                {
                    for (uint32_t i=0; i<n_samples; ++i)
                        out[i] += custom_model.forward (out + i);
                }
            }
            else
            {
                // TODO
            }
        },
        model->variant
    );
}

/**********************************************************************************************************************************************************/

LV2_Handle RtNeuralGeneric::instantiate(const LV2_Descriptor* descriptor, double samplerate, const char* bundle_path, const LV2_Feature* const* features)
{
    RtNeuralGeneric *self = new RtNeuralGeneric();

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
        delete self;
        return 0;
    } else if (!self->schedule) {
        std::cout << "Error! Missing feature work:schedule " << __func__ << " " << __LINE__ << std::endl;
        delete self;
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

    self->model = nullptr;

    return (LV2_Handle)self;
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::activate(LV2_Handle instance)
{
    RtNeuralGeneric *self = (RtNeuralGeneric*) instance;

    if (self->model == nullptr)
        return;

    // TODO: include the activate function code here
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
    /*float param1 = *self->param1;*/
    /*float param2 = *self->param2;*/
    float param1 = 0.0f;
    float param2 = 0.0f;

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

    /*++++++++ AUDIO DSP ++++++++*/
    applyBiquadFilter(self->out_1, self->in, self->in_lpf, n_samples); // High frequencies roll-off (lowpass)
    applyGainRamp(self->out_1, self->out_1, self->pregain_old, pregain, n_samples); // Pre-gain
    if(eq_position == 1.0f && eq_bypass == 0.0f) {
        applyToneControls(self->out_1, self->out_1, instance, n_samples); // Equalizer section
    }
    if (net_bypass == 0.0f && self->model != nullptr) {
        applyModel(self->model, self->out_1, n_samples);
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
        lv2_log_note(&self->logger, "Restoring file %s\n", (const char*)value);
        // send to worker for loading
        WorkerLoadMessage msg = { kWorkerLoad, {} };
        std::memcpy(msg.path, value, std::min(size, sizeof(msg.path) - 1u));
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
        if (DynamicModel* newmodel = RtNeuralGeneric::loadModel(&self->logger, ((const WorkerLoadMessage*)data)->path))
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

    // report change to host/ui
    lv2_atom_forge_frame_time(&self->forge, 0);
    write_set_file(&self->forge,
                   &self->uris,
                   self->model->path,
                   strlen(self->model->path));

    return LV2_WORKER_SUCCESS;
}

/**********************************************************************************************************************************************************/

/**
 * This function loads a pre-trained neural model from a json file
*/
DynamicModel* RtNeuralGeneric::loadModel(LV2_Log_Logger* logger, const char* path)
{
    int input_skip;
    nlohmann::json model_json;

    try {
        std::ifstream jsonStream(path, std::ifstream::binary);
        jsonStream >> model_json;

        /* Understand which model type to load */
        if(model_json["in_shape"].back().get<int>() > 1) {
            throw std::invalid_argument("Values for input_size > 1 are not supported");
        }

        if (model_json["in_skip"].is_number()) {
            input_skip = model_json["in_skip"].get<int>();
            if (input_skip > 1)
                throw std::invalid_argument("Values for in_skip > 1 are not supported");
        }
        else {
            input_skip = 0;
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
    }
    catch (const std::exception& e) {
        lv2_log_error(logger, "Error loading model: %s\n", e.what());
        return nullptr;
    }

    // save extra info
    model->path = strdup(path);
    model->input_skip = input_skip != 0;

    // Pre-buffer to avoid "clicks" during initialization
    float out[2048] = {};
    applyModel(model.get(), out, 2048);

    return model.release();
}

/**********************************************************************************************************************************************************/

/**
 * This function deletes a model instance and its related details
*/
void RtNeuralGeneric::freeModel(DynamicModel* model)
{
    if (model == nullptr)
        return;
    free (model->path);
    delete model;
}
