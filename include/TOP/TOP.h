#include "common/common.h"

#ifndef TOP_H
#define TOP_H

#define CREATE_TOP_CONTEXT( layername )\
        float32_t layername##_gradOut[ layername.size.m ][ layername.size.n ];  \
        layername.gradOut = (float32_t*) layername##_gradOut;                   \
        Initialize_Matrix( layername.gradOut, layername.size.m, layername.size.n, zero_initializer );\

void    Construct_TOP_Layer ( TOPLayer_t*, uint16_t, uint16_t, uint16_t, uint8_t );
/* METHODS */
void    Feedforward_TOP_Layer   ( TOPLayer_t* );
void    Loss_TOP_Layer          ( TOPLayer_t* );
void    Print_TOP_Layer         ( TOPLayer_t* );


#endif