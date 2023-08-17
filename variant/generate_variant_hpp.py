#!/usr/bin/env python3

max_input_size = 3
layer_types = ('GRU', 'LSTM')
input_sizes = tuple(range(1, max_input_size + 1))
hidden_sizes = (8, 12, 16, 20, 24, 32, 40, 64, 80)

model_variant_using_declarations = []
model_variant_types = []
model_type_checkers = []

def add_model(input_size, layer_type, hidden_size, model_type):
    model_type_alias = f'ModelType_{layer_type}_{hidden_size}_{input_size}'
    model_variant_using_declarations.append(f'using {model_type_alias} = {model_type};\n')
    model_variant_types.append(model_type_alias)
    model_type_checkers.append(f'''inline bool is_model_type_{model_type_alias} (const nlohmann::json& model_json) {{
    const auto json_layers = model_json.at ("layers");
    const auto rnn_layer_type = json_layers.at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "{layer_type.lower()}";
    const auto hidden_size = json_layers.at (0).at ("shape").back().get<int>();
    const auto is_hidden_size_correct = hidden_size == {hidden_size};
    const auto input_size = model_json.at ("in_shape").back().get<int>();
    const auto is_input_size_correct = input_size == {input_size};
    return is_layer_type_correct && is_hidden_size_correct && is_input_size_correct;
}}\n\n''')

for layer_type in layer_types:
    for hidden_size in hidden_sizes:
        for input_size in input_sizes:
            print(f'Setting up Model: {layer_type} w/ RNN dims {input_size} / {hidden_size}, w/ I/O dims {input_size} / 1')

            if layer_type == 'GRU':
                rnn_layer_type = f'RTNeural::GRULayerT<float, {input_size}, {hidden_size}>'
            elif layer_type == 'LSTM':
                rnn_layer_type = f'RTNeural::LSTMLayerT<float, {input_size}, {hidden_size}>'

            dense_layer_type = f'RTNeural::DenseT<float, {hidden_size}, 1>'

            model_type = f'RTNeural::ModelT<float, {input_size}, 1, {rnn_layer_type}, {dense_layer_type}>'
            add_model(input_size, layer_type, hidden_size, model_type)

with open("rt-neural-generic/src/model_variant.hpp", "w") as header_file:
    header_file.write('#include <variant>\n')
    header_file.write('#include <RTNeural/RTNeural.h>\n')
    header_file.write('\n')

    header_file.write(f'#define MAX_INPUT_SIZE {max_input_size}\n')

    header_file.write('struct NullModel { static constexpr int input_size = 0; static constexpr int output_size = 0; };\n')
    header_file.writelines(model_variant_using_declarations)
    header_file.write(f'using ModelVariantType = std::variant<NullModel,{",".join(model_variant_types)}>;\n')
    header_file.write('\n')

    header_file.writelines(model_type_checkers)
    
    header_file.write('inline bool custom_model_creator (const nlohmann::json& model_json, ModelVariantType& model) {\n')
    if_statement = 'if'
    for type_checker, alias in zip(model_type_checkers, model_variant_types):
        header_file.write(f'    {if_statement} (is_model_type_{alias} (model_json)) {{\n')
        header_file.write(f'        model.emplace<{alias}>();\n')
        header_file.write(f'        return true;\n')
        header_file.write('    }\n')
        if_statement = 'else if'
    header_file.write(f'    model.emplace<NullModel>();\n')
    header_file.write(f'    return false;\n')
    header_file.write('}\n')
