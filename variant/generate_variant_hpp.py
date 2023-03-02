#!/usr/bin/env python3

layer_types = ('GRU', 'LSTM')
io_dims = tuple(range(1, 2 + 1))
rnn_dims = (8, 12, 16, 20, 32, 40)

model_variant_using_declarations = []
model_variant_types = []
model_type_checkers = []

def add_model(layer_type, rnn_dim, io_dim, model_type, sigmoid):
    model_type_alias = f'ModelType_{layer_type}_{rnn_dim}_{io_dim}{"_sigmoid" if sigmoid else ""}'
    model_variant_using_declarations.append(f'using {model_type_alias} = {model_type};\n')
    model_variant_types.append(model_type_alias)
    model_type_checkers.append(f'''inline bool is_model_type_{model_type_alias} (const nlohmann::json& model_json) {{
    const auto json_layers = model_json.at ("layers");
    const auto rnn_layer_type = json_layers.at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "{layer_type.lower()}";
    const auto rnn_dim = json_layers.at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == {rnn_dim};
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == {io_dim};
    const auto has_sigmoid_activation = json_layers.size() == 3 && json_layers.at (1).at ("activation") == "sigmoid";
    const auto is_sigmoid_activation_correct = has_sigmoid_activation == {"true" if sigmoid else "false"};
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct && is_sigmoid_activation_correct;
}}\n\n''')

for layer_type in layer_types:
    for rnn_dim in rnn_dims:
        for io_dim in io_dims:
            print(f'Setting up Model: {layer_type} w/ RNN dims {rnn_dim}, w/ I/O dims {io_dim}')

            if layer_type == 'GRU':
                rnn_layer_type = f'RTNeural::GRULayerT<float, {io_dim}, {rnn_dim}>'
            elif layer_type == 'LSTM':
                rnn_layer_type = f'RTNeural::LSTMLayerT<float, {io_dim}, {rnn_dim}>'

            dense_layer_mid_type = f'RTNeural::DenseT<float, {rnn_dim}, {rnn_dim}>'
            dense_layer_end_type = f'RTNeural::DenseT<float, {rnn_dim}, {io_dim}>'

            # with sigmoid activation
            activation_layer_type = f'RTNeural::SigmoidActivationT<float, {rnn_dim}>'
            model_type = f'RTNeural::ModelT<float, {io_dim}, {io_dim}, {rnn_layer_type}, {dense_layer_mid_type}, {activation_layer_type}, {dense_layer_end_type}>'
            add_model(layer_type, rnn_dim, io_dim, model_type, True)

            # without sigmoid activation
            model_type = f'RTNeural::ModelT<float, {io_dim}, {io_dim}, {rnn_layer_type}, {dense_layer_end_type}>'
            add_model(layer_type, rnn_dim, io_dim, model_type, False)

with open("rt-neural-generic/src/model_variant.hpp", "w") as header_file:
    header_file.write('#include <variant>\n')
    header_file.write('#include <RTNeural/RTNeural.h>\n')
    header_file.write('\n')

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
