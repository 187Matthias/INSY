#ifndef XOR_H
#define XOR_H

#define XOR_NUM_LAYERS 2
const int xor_input_size = 2;
const int xor_layer_sizes[2] = {2, 1};

const double xor_l0_bias[2] = {-6.8782864987, 2.2510696836};
const double xor_l0_weights[2][2] = {
    {4.4189489410, 4.4199073121},
    {-5.9017529033, -5.9059065821}
};

const double xor_l1_bias[1] = {4.3310633832};
const double xor_l1_weights[1][2] = {
    {-8.8741601202, -8.7157079379}
};

#endif /* XOR_H */
