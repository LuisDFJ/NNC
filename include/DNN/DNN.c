#include "DNN/DNN.h"

void    Construct_DNN_Layer ( DNNLayer_t* self, uint16_t input, uint16_t output, uint16_t batch, uint8_t act, float32_t nambda )
{
    srand( time(NULL) );
    self->input_size.m      =   input;
    self->input_size.n      =   batch;
    self->weights_size.m    =   output;
    self->weights_size.n    =   input;
    self->output_size.m     =   output;
    self->output_size.n     =   batch;
    self->nambda = nambda;
    self->print             =   Print_DNN_Layer;
    self->feedforward       =   Feedforward_DNN_Layer;
    self->backpropagation   =   Backpropagation_DNN_Layer;
    switch (act)
    {
    case 'r':
    case 'R':
        self->activation_function   =   relu_activation; 
        self->gradient_function     =   relu_gradient;
        break;
    case 's':
    case 'S':
    default:
        self->activation_function   =   sigmoid_activation;
        self->gradient_function     =   sigmoid_gradient;
        break;
    }
    printf( "DONE DNN LAYER\n" );
}

void Feedforward_DNN_Layer  ( DNNLayer_t* self )
{
    Multiplication_Matrix( self->output, self->zeta, self->weights, self->input, self->activation_function, self->output_size.m, self->output_size.n, self->input_size.m );
}

void Backpropagation_DNN_Layer  ( DNNLayer_t* self )
{
    Hadamard_Product_Matrix( self->grad, self->gradIn, self->zeta, self->gradient_function, self->output_size.m, self->output_size.n );
    Multiplication_Matrix_TA( self->gradOut, self->weights, self->grad, self->input_size.m, self->input_size.n, self->output_size.m );
    Multiplication_Matrix_TB( self->weights, self->grad, self->input, self->nambda, self->weights_size.m, self->weights_size.n, self->output_size.n );
}

void Print_DNN_Layer        ( DNNLayer_t* self )
{
    printf( "-----------------------------------\n" );
    printf( "*DNN LAYER:*\n" );
    printf( "-FEEDFORWARD-\n" );
    printf( "INPUT:\n" );
    Print_Matrix( self->input, self->input_size.m, self->input_size.n );
    printf( "WEIGHTS:\n" );
    Print_Matrix( self->weights, self->weights_size.m, self->weights_size.n );
    printf( "ZETA:\n" );
    Print_Matrix( self->zeta, self->output_size.m, self->output_size.n );
    printf( "OUTPUT:\n" );
    Print_Matrix( self->output, self->output_size.m, self->output_size.n );
    printf( "-BACKPROPAGATION-\n" );
    printf( "GRADIENT OUT:\n" );
    Print_Matrix( self->gradOut, self->input_size.m, self->input_size.n );
    printf( "GRADIENT:\n" );
    Print_Matrix( self->grad, self->output_size.m, self->output_size.n );
    printf( "GRADIENT IN:\n" );
    Print_Matrix( self->gradIn, self->output_size.m, self->output_size.n );
    printf( "-----------------------------------\n" );
}

