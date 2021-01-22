#include "TOP/TOP.h"

void Construct_TOP_Layer    ( TOPLayer_t* self, uint16_t input, uint16_t batch, uint16_t index, uint8_t act )
{
    self->size.m = input;
    self->size.n = batch;
    self->index = index;
    self->global_loss = 0;
    self->print = Print_TOP_Layer;
    self->feedforward = Feedforward_TOP_Layer;
    self->loss = Loss_TOP_Layer;
    switch (act)
    {
    case 's':
    case 'S':
        self->activation_function = square_activation;
        break;
    case 'l':
    case 'L':
    default:
        self->activation_function = linear_activation;
        break;
    }

    printf( "DONE TOP LAYER\n" );
}

void Feedforward_TOP_Layer  ( TOPLayer_t* self )
{
    Substraction_Matrix( self->gradOut, self->output, self->input, self->activation_function, self->size.m, self->size.n );
    self->loss( self );
}

void Loss_TOP_Layer         ( TOPLayer_t* self )
{
    self->global_loss = Mean_Absolute_Error( self->gradOut, self->size.m, self->size.n );
}

void Print_TOP_Layer    ( TOPLayer_t* self )
{
    printf( "-----------------------------------\n" );
    printf( "*TOP LAYER:*  (LOSS: %0.5f)\n", self->global_loss );
    printf( "INPUT:\n" );
    Print_Matrix( self->input, self->size.m, self->size.n );
    printf( "OUTPUT:\n" );
    Print_Matrix( self->output, self->size.m, self->size.n );
    printf( "GRADIENT OUT:\n" );
    Print_Matrix( self->gradOut, self->size.m, self->size.n );
    printf( "-----------------------------------\n\n\n" );

}

