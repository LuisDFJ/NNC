#include "common/common.h"

#ifndef DNN_H
#define DNN_H


#define CREATE_DNN_CONTEXT( layername )\
        float32_t layername##_weights[ layername.weights_size.m ][ layername.weights_size.n ];  \
        float32_t layername##_output[ layername.output_size.m ][ layername.output_size.n ];     \
        float32_t layername##_zeta[ layername.output_size.m ][ layername.output_size.n ];       \
        float32_t layername##_grad[ layername.output_size.m ][ layername.output_size.n ];       \
        float32_t layername##_gradOut[ layername.input_size.m ][ layername.input_size.n ];      \
        layername.weights = (float32_t*) layername##_weights;                                   \
        layername.output = (float32_t*) layername##_output;                                     \
        layername.zeta = (float32_t*) layername##_zeta;                                         \
        layername.grad = (float32_t*) layername##_grad;                                         \
        layername.gradOut = (float32_t*) layername##_gradOut;                                   \
        Initialize_Matrix( layername.weights, layername.weights_size.m, layername.weights_size.n, random_initializer ); \
        Initialize_Matrix( layername.output, layername.output_size.m, layername.output_size.n, zero_initializer );      \
        Initialize_Matrix( layername.zeta, layername.output_size.m, layername.output_size.n, zero_initializer );        \
        Initialize_Matrix( layername.grad, layername.output_size.m, layername.output_size.n, zero_initializer );        \
        Initialize_Matrix( layername.gradOut, layername.input_size.m, layername.input_size.n, zero_initializer );       \

/* CONSTRUCTOR */
void    Construct_DNN_Layer     ( DNNLayer_t*, uint16_t, uint16_t, uint16_t, uint8_t, float32_t );
/* METHODS */
void    Feedforward_DNN_Layer           ( DNNLayer_t* );
void    Backpropagation_DNN_Layer       ( DNNLayer_t* );
void    Print_DNN_Layer                 ( DNNLayer_t* );

#endif