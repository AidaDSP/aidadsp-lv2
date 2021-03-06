#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lv2.h>
#include "RTNeuralLSTM.h"

/**********************************************************************************************************************************************************/

#define PLUGIN_URI "http://aidadsp.cc/plugins/aidadsp-bundle/rt-neural-generic"
#define LSTM_MODEL_JSON_FILE_NAME "lstm-model.json"
enum {IN, OUT_1, GAIN, MASTER, BYPASS, PLUGIN_PORT_COUNT};

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
    float *gain;
    float *master;
    int *bypass;

    RT_LSTM LSTM;
    int model_loaded = 0;
    // The number of parameters for the model
    // 0 is for a snap shot model
    int params = 0;
    static void loadConfig(LV2_Handle instance, const char *bundle_path, const char *fileName);
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

void RtNeuralGeneric::loadConfig(LV2_Handle instance, const char *bundle_path, const char *fileName)
{
    RtNeuralGeneric *plugin;
    plugin = (RtNeuralGeneric *) instance;

    std::string filePath;

    filePath.append(bundle_path);
    filePath.append(fileName);

    std::cout << std::endl << "Loading json file: " << filePath << std::endl;

    try {
        // Load the JSON file into the correct model
        plugin->LSTM.load_json(filePath.c_str());

        // Check what the input size is and then update the GUI appropirately
        if (plugin->LSTM.input_size == 1) {
            plugin->params = 0;
        }
        else if (plugin->LSTM.input_size == 2) {
            plugin->params = 1;
        }
        else if (plugin->LSTM.input_size == 3) {
            plugin->params = 2;
        }

        // If we are good: let's say so
        plugin->model_loaded = 1;
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
    plugin->loadConfig((LV2_Handle)plugin, bundle_path, LSTM_MODEL_JSON_FILE_NAME);

    // Before running inference, it is recommended to "reset" the state
    // of your model (if the model has state).
    plugin->LSTM.reset();

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
        case GAIN:
            plugin->gain = (float*) data;
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
    float gain = *plugin->gain;
    float master = *plugin->master;
    int bypass = *plugin->bypass;

    if (plugin->model_loaded == 1) {
        // Process LSTM based on input_size (snapshot model or conditioned model)
        if (plugin->LSTM.input_size == 1) {
            plugin->LSTM.process(plugin->in, plugin->out_1, n_samples, bypass);
        }
        else if (plugin->LSTM.input_size == 2) {
            plugin->LSTM.process(plugin->in, gain, plugin->out_1, n_samples, bypass);
        }
        else if (plugin->LSTM.input_size == 3) {
            plugin->LSTM.process(plugin->in, gain, master, plugin->out_1, n_samples, bypass);
        }
    }

    // @TODO: volume normalization may be useful when switching between models!
    // @TODO: some offset may be present at neural network output, original code
    // add a dc block filter in this position
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
