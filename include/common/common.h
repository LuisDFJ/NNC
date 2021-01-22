#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef COMMON_H
#define COMMON_H

typedef float   float32_t;
typedef double  float64_t;

typedef struct Size2D
{
    uint16_t m;
    uint16_t n;
} Size2D_t;

typedef struct DNNLayer
{
    Size2D_t input_size;
    Size2D_t weights_size;
    Size2D_t output_size;
    float32_t nambda;
    float32_t (*activation_function)    (float32_t);
    float32_t (*gradient_function)      (float32_t);
    void (*print)       (   struct DNNLayer*);
    void (*feedforward)     (struct DNNLayer*);
    void (*backpropagation) (struct DNNLayer*);
    float32_t* input;
    float32_t* weights;
    float32_t* zeta;
    float32_t* output;
    float32_t* gradIn;
    float32_t* grad;
    float32_t* gradOut;

} DNNLayer_t;

typedef struct TOPLayer
{
    Size2D_t size;
    uint16_t index;
    float32_t  global_loss;
    float32_t (*activation_function)    (float32_t);
    void (*print)       (struct TOPLayer*);
    void (*feedforward) (struct TOPLayer*);
    void (*loss)        (struct TOPLayer*);
    float32_t* input;
    float32_t* output;
    float32_t* gradOut;

} TOPLayer_t;

typedef struct LayersType
{
    void*   Layer;
    uint8_t Type;
} Layers_t;

typedef struct NetworkType
{
    Layers_t* Layers;
    uint16_t  n;
} Network_t;

/* SCALAR OPERATIONS */
float32_t sigmoid_activation    ( float32_t );
float32_t sigmoid_gradient      ( float32_t );
float32_t relu_activation       ( float32_t );
float32_t relu_gradient         ( float32_t );
float32_t linear_activation     ( float32_t );
float32_t square_activation     ( float32_t );
float32_t random_initializer    ( void );
float32_t zero_initializer      ( void );
float32_t one_initializer       ( void );
float32_t softmax_loss          ( float32_t*, uint16_t, uint16_t );
/* LOSS FUNCTIONS */
float32_t Mean_Absolute_Error   ( float32_t*, uint16_t, uint16_t );
/* MATRIX OPERATIONS */
void    Multiplication_Matrix   ( float32_t*, float32_t*, float32_t*, float32_t*, float32_t (*)(float32_t), uint16_t, uint16_t, uint16_t );
void    Multiplication_Matrix_TA( float32_t*,  float32_t*, float32_t*, uint16_t, uint16_t, uint16_t );
void    Multiplication_Matrix_TB( float32_t*,  float32_t*, float32_t*, float32_t, uint16_t, uint16_t, uint16_t );
void    Hadamard_Product_Matrix ( float32_t*, float32_t*, float32_t*, float32_t (*)(float32_t), uint16_t, uint16_t );
void    Substraction_Matrix     ( float32_t*, float32_t*, float32_t*, float32_t (*)(float32_t), uint16_t, uint16_t );
/* UTILS */
void    Initialize_Matrix   ( float32_t*, uint16_t, uint16_t, float32_t (*)() );
void    Print_Matrix        ( float32_t*, uint16_t, uint16_t );


#endif