#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <RTNeural/RTNeural.h>

#define JSON_MODEL_FILE_NAME "model.json"

using namespace std;

int main(void) {
    RTNeural::ModelT<float, 1, 1,
        RTNeural::LSTMLayerT<float, 1, 16>,
        RTNeural::DenseT<float, 16, 1>> lstm_16;
    RTNeural::ModelT<float, 1, 1,
        RTNeural::LSTMLayerT<float, 1, 12>,
        RTNeural::DenseT<float, 12, 1>> lstm_12;
    RTNeural::ModelT<float, 1, 1,
        RTNeural::GRULayerT<float, 1, 8>,
        RTNeural::DenseT<float, 8, 1>> gru_8;
    int model_index = 0;
    int n_layers = 0;
    int input_size = 0;
    int hidden_size = 0;
    std::string type;
    int in_skip = 0;

    std::string filePath;

    filePath.append(JSON_MODEL_FILE_NAME);

    std::cout << "Loading json file: " << filePath << std::endl;

    try {
        std::ifstream jsonStream(filePath);
        std::ifstream jsonStream2(filePath);
        nlohmann::json modelData;
        jsonStream2 >> modelData;

        /* Understand which model type to load */
        n_layers = modelData["layers"].size(); /* Get how many layers we have in this nn model */

        input_size = modelData["in_shape"].back().get<int>();

        if (modelData["in_skip"].is_number())
            in_skip = modelData["in_skip"].get<int>();
        else
            in_skip = 0;

        type = modelData["layers"][n_layers-1-1]["type"];
        hidden_size = modelData["layers"][n_layers-1-1]["shape"].back().get<int>();

        std::cout << "input_size: " << input_size << std::endl;
        std::cout << "in_skip: " << in_skip << std::endl;
        std::cout << "n_layers: " << n_layers << std::endl;
        std::cout << "type: " << type << std::endl;
        std::cout << "hidden_size: " << hidden_size << std::endl;

        if(type == std::string("lstm"))
        {
            if(hidden_size == 16)
            {
                model_index = 0;
            }
            else if(hidden_size == 12)
            {
                model_index = 1;
            }
        }
        else if(type == std::string("gru"))
        {
            if(hidden_size == 8)
            {
                model_index = 2;
            }
        }

        switch(model_index)
        {
            case 0:
                lstm_16.parseJson(jsonStream, true);
                break;
            case 1:
                lstm_12.parseJson(jsonStream, true);
                break;
            case 2:
                gru_8.parseJson(jsonStream, true);
                break;
        }
    }
    catch (const std::exception& e) {
        std::cout << std::endl << "Unable to load json file: " << filePath << std::endl;
        std::cout << e.what() << std::endl;
    }
}