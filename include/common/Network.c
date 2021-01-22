#include "common/Network.h"

void Construct_Network      ( Network_t* self, float32_t* input, float32_t* output )
{
    uint16_t    i;
    DNNLayer_t* pDNN;
    TOPLayer_t* pTOP;
    float32_t*  pOut;
    pOut = input;
    for( i = 0 ; i < self->n ; i++ )
    {
        switch ( ( self->Layers + i )->Type )
        {
        case 'd':
        case 'D':
            pDNN = (DNNLayer_t*)( self->Layers + i )->Layer;
            pDNN->input = pOut;
            pOut = pDNN->output;
            break;
        case 't':
        case 'T':
            pTOP = (TOPLayer_t*)( self->Layers + i )->Layer;
            pTOP->input = pOut;
            break;
        default:
            break;
        }
    }

    pOut = output;
    
    for( i = 0 ; i < self->n ; i++ )
    {
        switch ( ( self->Layers + self->n - 1 - i )->Type )
        {
        case 'd':
        case 'D':
            pDNN = (DNNLayer_t*)( self->Layers + self->n - 1 - i )->Layer;
            pDNN->gradIn = pOut;
            pOut = pDNN->gradOut;
            break;
        case 't':
        case 'T':
            pTOP = (TOPLayer_t*)( self->Layers + self->n - 1 - i )->Layer;
            pTOP->output = pOut;
            break;
        default:
            break;
        }
    }
}
/*
 *  PRINT A LAYER ARRAY
 */
void Print_Network          ( Network_t* self )
{
    uint16_t     i;
    DNNLayer_t* pDNN;
    TOPLayer_t* pTOP;
    for( i = 0 ; i < self->n ; i++ )
    {
        switch ( ( self->Layers + i )->Type )
        {
        case 'd':
        case 'D':
            pDNN    =   (DNNLayer_t*) ( self->Layers + i )->Layer;
            pDNN->print(pDNN);
            break;
        case 't':
        case 'T':
            pTOP    =   (TOPLayer_t*) ( self->Layers + i )->Layer;
            pTOP->print(pTOP);
            break;
        default:
            break;
        }
    }
}