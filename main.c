#include "main.h"

int main( void )
{
    float32_t matrix1[3][6] = {
        { 1, 0, 1, 0, 0, 1},
        { 1, 1, 0, 1, 0, 0},
        { 1, 0, 1, 1, 1, 0}    
    };
    float32_t matrix2[1][6] = {
        { 1, 0, 1, 1, 0, 0 }    
    };
    /*
    float32_t matrix3[3][6] = {
        { 1, 0, 1, 0, 1, 1},
        { 0, 1, 0, 0, 0, 1},
        { 1, 1, 0, 0, 1, 0}    
    };
    float32_t matrix4[1][6] = {
        { 1, 1, 0, 0, 1, 1 }    
    };
    */
    DNNLayer_t layer1, layer2;
    TOPLayer_t layer3;

    Construct_DNN_Layer(&layer1,3,3,6,'s', 0.5 );
    Construct_DNN_Layer(&layer2,3,1,6,'s', 0.5 );
    Construct_TOP_Layer(&layer3,1,6,0,'l' );

    CREATE_DNN_CONTEXT(layer1);
    CREATE_DNN_CONTEXT(layer2);
    CREATE_TOP_CONTEXT(layer3);

    Layers_t Layers[3] = {
        { (void*) &layer1, 'd' },
        { (void*) &layer2, 'd' },
        { (void*) &layer3, 't' }
    };

    Network_t Network = { (Layers_t*) &Layers, 3 };

    Construct_Network(&Network,(float32_t*)matrix1,(float32_t*)matrix2);

    
    Print_Network(&Network);

    return 0;
}