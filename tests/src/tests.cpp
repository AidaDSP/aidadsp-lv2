#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <variant>
#include <RTNeural/RTNeural.h>

#define JSON_MODEL_FILE_NAME "model.json"

using namespace std;

int main(void) {
    RTNeural::ModelT<float, 3, 1,
        RTNeural::DenseT<float, 3, 4>,
        RTNeural::GRULayerT<float, 4, 8>,
        RTNeural::DenseT<float, 8, 1>> model;

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
        model.parseJson(jsonStream, true);

        n_layers = modelData["layers"].size(); /* Get how many layers we have in this nn model */

        input_size = modelData["in_shape"].back().get<int>();

        if (modelData["in_skip"].is_number())
            in_skip = modelData["in_skip"].get<int>();
        else
            in_skip = 0;

        type = modelData["layers"][n_layers-1]["type"];
        hidden_size = modelData["layers"][n_layers-1]["shape"].back().get<int>();

        std::cout << "input_size: " << input_size << std::endl;
        std::cout << "in_skip: " << in_skip << std::endl;
        std::cout << "n_layers: " << n_layers << std::endl;
        std::cout << "type: " << type << std::endl;
        std::cout << "hidden_size: " << hidden_size << std::endl;

        std::cout << "Successfully loaded json file: " << filePath << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << std::endl << "Unable to load json file: " << filePath << std::endl;
        std::cout << e.what() << std::endl;
    }
}