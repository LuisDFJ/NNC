#include "common/common.h"


/*
 *  SIGMOID ACTIVATION FUNCTION
 */
float32_t sigmoid_activation    ( float32_t x )
{
    float32_t val   =   ( 1.0 ) / ( 1.0 + exp(-x) );
    return val;
}
/*
 *  SIGMOID GRADIENT FUNCTION
 */
float32_t sigmoid_gradient      ( float32_t x )
{
    float32_t val   =   sigmoid_activation( x ) * ( 1 - sigmoid_activation( x ) );
    return val;
}
/*
 *  RELU ACTIVATION FUNCTION
 */
float32_t relu_activation       ( float32_t x )
{
    float32_t val   =   x > 0 ? x : 0;
    return val;
}
/*
 *  RELU GRADIENT FUNCTION
 */
float32_t relu_gradient         ( float32_t x )
{
    float32_t val =     x > 0 ? 1 : 0;
    return val;
}
/*
 *  LINEAR ACTIVATION FUNCTION
 */
float32_t linear_activation     ( float32_t x )
{
    return x;
}
/*
 *  SQUARE ACTIVATION FUNCTION
 */
float32_t square_activation     ( float32_t x )
{
    float32_t val = ( x * x ) / 2;
    return val;
}
/*
 *  RANDOM INITIALIZER
 */
float32_t random_initializer    ( void )
{
    float32_t val = ( rand() % 20000 ) / 10000.0 - 1 ;
    return val;
}
/*
 *  ZERO INITIALIZER
 */
float32_t zero_initializer      ( void )
{
    return 0.0;
}
/*
 *  ONE INITIALIZER
 */
float32_t one_initializer       ( void )
{
    return 1.0;
}
/*
 *  SOFTMAX LOSS
 */
float32_t softmax_loss          ( float32_t* p, uint16_t size, uint16_t index )
{
    float32_t val = 0.0;
    uint16_t i;
    for( i = 0 ; i < size ; i++ ) 
    {
        val += exp( *( p + i ) );
    }
    val = exp( *(p + index) ) / val;
    return val;
}
/* LOSS FUNCTIONS */
/*
 *  MEAN ABSOLUTE ERROR
 */
float32_t Mean_Absolute_Error   ( float32_t* pG, uint16_t i, uint16_t j )
{
    uint16_t ci, cj;
    float32_t val;
    float32_t avg = 0;
    for( cj = 0 ; cj < j ; cj++ )
    {
        val = 0;
        for( ci = 0 ; ci < i ; ci++ )
        {
            val += *( pG + ci * j + cj ) > 0 ? *( pG + ci * j + cj ) : *( pG + ci * j + cj ) * -1 ;
        }
        val /= i;
        avg += val;
    }
    avg /= j;
    return avg;
}

/* MATRIX OPERATIONS */

/*
 *  MATRIX MULTIPLICATION
 */
void    Multiplication_Matrix   ( float32_t* pO, float32_t* pZ,  float32_t* pA, float32_t* pB, float32_t (*func) (float32_t), uint16_t i, uint16_t j, uint16_t n)
{
    uint16_t ci, cj, cn;
    float32_t val;
    for( ci = 0 ; ci < i ; ci++ )
    {
        for( cj = 0 ; cj < j ; cj++ )
        {
            val = 0;
            for( cn = 0 ; cn < n ; cn++ )
            {
                val += ( *( pA + ci * n + cn ) ) * ( *( pB + cn * j + cj ) );
            }
            *( pZ + ci * j + cj ) = val;
            *( pO + ci * j + cj ) = func( val );
        }
    }
}
/*
 *  TRASPOSE A MATRIX MULTIPLICATION 
 */
void    Multiplication_Matrix_TA( float32_t* pO,  float32_t* pT, float32_t* pB, uint16_t i, uint16_t j, uint16_t n)
{
    uint16_t ci, cj, cn;
    float32_t val;
    for( ci = 0 ; ci < i ; ci++ )
    {
        for( cj = 0 ; cj < j ; cj++ )
        {
            val = 0;
            for( cn = 0 ; cn < n ; cn++ )
            {
                val += ( *( pT + cn * i + ci ) ) * ( *( pB + cn * j + cj ) );
            }
            *( pO + ci * j + cj ) = val;
        }
    }
}
/*
 *  TRASPOSE B MATRIX MULTIPLICATION 
 */
void    Multiplication_Matrix_TB( float32_t* pO,  float32_t* pA, float32_t* pT, float32_t nambda, uint16_t i, uint16_t j, uint16_t n)
{
    uint16_t ci, cj, cn;
    float32_t val;
    for( ci = 0 ; ci < i ; ci++ )
    {
        for( cj = 0 ; cj < j ; cj++ )
        {
            val = 0;
            for( cn = 0 ; cn < n ; cn++ )
            {
                val += ( *( pA + ci * n + cn ) ) * ( *( pT + cj * n + cn ) );
            }
            val *= nambda;
            *( pO + ci * j + cj ) += val;
        }
    }
}
/*
 *  MATRIX HADAMARD PRODUCT
 */
void    Hadamard_Product_Matrix ( float32_t* pO, float32_t* pA, float32_t* pZ, float32_t (*func) (float32_t), uint16_t i, uint16_t j )
{
    uint16_t ci, cj;
    for( ci = 0 ; ci < i ; ci++ )
    {
        for( cj = 0 ; cj < j ; cj++ )
        {
            *( pO + ci * j + cj ) = *( pA + ci * j + cj ) * func( *( pZ + ci * j + cj ) );
        }
    }
}
/*
 *  MATRIX SUBSTRACTION
 */
void    Substraction_Matrix     ( float32_t* pGO, float32_t* pO, float32_t* pI, float32_t (*func) (float32_t),uint16_t i, uint16_t j )
{
    uint16_t ci, cj;
    for( ci = 0 ; ci < i ; ci++ )
    {
        for( cj = 0 ; cj < j ; cj++ )
        {
            *( pGO + ci * j + cj ) = func( *( pO + ci * j + cj ) - *( pI + ci * j + cj ) );
        }
    }
}
/* UTILS */

/*
 *  MATRIX INITIALIZER
 */
void Initialize_Matrix  ( float32_t* p, uint16_t m, uint16_t n, float32_t (*func)(void) )
{
    uint16_t i, j;
    for ( i = 0 ; i < m ; i++ )
    {
        for ( j = 0 ; j < n ; j++ )
        {
            *( p + i * n + j ) = func();
        }
    }
}
/*
 *  MATRIX PRINTER
 */
void Print_Matrix       ( float32_t* p, uint16_t m, uint16_t n )
{
    uint16_t i, j;
    for ( i = 0 ; i < m ; i++ )
    {
        for ( j = 0 ; j < n ; j++ )
        {
            printf( " %0.5f " , *( p + i * n + j ) );
        }
        printf( "\n" );
    }
}
