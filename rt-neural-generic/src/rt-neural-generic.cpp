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

// Apply gain setting with a ramp to avoid zypper noise
float RtNeuralGeneric::calcGain(float gain, float gain_old, uint32_t n_samples, uint32_t index) {
    return (gain_old + ((gain - gain_old)/n_samples) * index);
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::loadModel(LV2_Handle instance, const char *bundle_path, const char *fileName)
{
    RtNeuralGeneric *plugin;
    plugin = (RtNeuralGeneric *) instance;

    std::string filePath;

    filePath.append(bundle_path);
    filePath.append(fileName);

    std::cout << "Loading json file: " << filePath << std::endl;

    try {
        std::ifstream jsonStream(filePath, std::ifstream::binary);
        plugin->model = RTNeural::json_parser::parseJson<float>(jsonStream, true);

        plugin->input_size = plugin->model->layers[0]->in_size;

        // If we are good: let's say so
        plugin->model_loaded = 1;

        std::cout << "Successfully loaded json file: " << filePath << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << std::endl << "Unable to load json file: " << filePath << std::endl;
        std::cout << e.what() << std::endl;

        // If we are not good: let's say no
        plugin->model_loaded = 0;
    }
}

/**********************************************************************************************************************************************************/

LV2_Handle RtNeuralGeneric::instantiate(const LV2_Descriptor* descriptor, double samplerate, const char* bundle_path, const LV2_Feature* const* features)
{
    RtNeuralGeneric *plugin = new RtNeuralGeneric();

    // Load lstm model json file
    plugin->loadModel((LV2_Handle)plugin, bundle_path, LSTM_MODEL_JSON_FILE_NAME);

    // Before running inference, it is recommended to "reset" the state
    // of your model (if the model has state).
    plugin->model->reset();

    plugin->bypass_old = 0;

    return (LV2_Handle)plugin;
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
    RtNeuralGeneric *plugin;
    plugin = (RtNeuralGeneric *) instance;

    switch (port)
    {
        case IN:
            plugin->in = (float*) data;
            break;
        case OUT_1:
            plugin->out_1 = (float*) data;
            break;
        case PARAM1:
            plugin->param1 = (float*) data;
            break;
        case PARAM2:
            plugin->param2 = (float*) data;
            break;
        case MASTER:
            plugin->master = (float*) data;
            break;
        case BYPASS:
            plugin->bypass = (int*) data;
            break;
    }
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::run(LV2_Handle instance, uint32_t n_samples)
{
    RtNeuralGeneric *plugin;
    plugin = (RtNeuralGeneric *) instance;

    float param1 = *plugin->param1;
    float param2 = *plugin->param2;
    int bypass = *plugin->bypass; // NOTE: since float 1.0 is sent instead of (int 32bit) 1, then we have 1065353216 as 1
    float master, master_old, tmp;
    uint32_t i;

    master = *plugin->master;
    master_old = plugin->master_old;
    plugin->master_old = master;

    if (bypass != plugin->bypass_old) {
        std::cout << "Bypass status changed to: " << bypass << std::endl;
        plugin->bypass_old = bypass;
    }

    if (bypass == 0) {
        if (plugin->model_loaded == 1) {
            // Process model based on input_size (snapshot model or conditioned model)
            switch(plugin->input_size) {
                case 1:
                    for(i=0; i<n_samples; i++) {
                        plugin->out_1[i] = plugin->model->forward(plugin->in + i) + plugin->in[i];
                        plugin->out_1[i] *= plugin->calcGain(master, master_old, n_samples, i);
                    }
                    break;
                case 2:
                    for(i=0; i<n_samples; i++) {
                        plugin->inArray1[0] = plugin->in[i];
                        plugin->inArray1[1] = param1;
                        plugin->out_1[i] = plugin->model->forward(plugin->inArray1) + plugin->in[i];
                        plugin->out_1[i] *= plugin->calcGain(master, master_old, n_samples, i);
                    }
                    break;
                case 3:
                    for(i=0; i<n_samples; i++) {
                        plugin->inArray2[0] = plugin->in[i];
                        plugin->inArray2[1] = param1;
                        plugin->inArray2[2] = param2;
                        plugin->out_1[i] = plugin->model->forward(plugin->inArray2) + plugin->in[i];
                        plugin->out_1[i] *= plugin->calcGain(master, master_old, n_samples, i);
                    }
                    break;
                default:
                    break;
            }
            // @TODO: volume normalization may be useful when switching between models!
            // @TODO: some offset may be present at neural network output, original code
            // add a dc block filter in this position
        }
    }
    else
    {
        std::copy(plugin->in, plugin->in + n_samples, plugin->out_1); // Passthrough
    }
}

/**********************************************************************************************************************************************************/

void RtNeuralGeneric::cleanup(LV2_Handle instance)
{
    delete ((RtNeuralGeneric *) instance);
}

/**********************************************************************************************************************************************************/

const void* RtNeuralGeneric::extension_data(const char* uri)
{
    return NULL;
}
