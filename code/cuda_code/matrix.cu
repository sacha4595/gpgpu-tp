#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include "error.h"
#include <cuda_runtime.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define TILE_WIDTH 16

// matrix_t * alloc_matrix(unsigned rows, unsigned columns)
// {
//     matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
//     res->m = (double *) calloc(columns * rows, sizeof(double));
//     res->columns = columns;
//     res->rows = rows;
//     return res;
// }


matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->columns = columns;
    res->rows = rows;

    // Allocate managed memory
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&res, sizeof(matrix_t)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&res->m, columns * rows * sizeof(double)));

    return res;
}


// void destroy_matrix(matrix_t *m)
// {
//     //printf("free %p %p\n", m, m->m);
//     free(m->m);
//     free(m);
// }

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    CHECK_CUDA_ERROR(cudaFree(m->m));
    CHECK_CUDA_ERROR(cudaFree(m));
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}

// void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
// {
//     assert ( (m1->columns == m2->rows)  &&
//              (m1->rows == res->rows)    &&
//              (m2->columns == res->columns));

//     for (int row = 0; row < m1->rows; row ++)
//     {
//         for (int col = 0; col < m2->columns; col ++)
//         {
//             int idx = col + row * m2->columns;
//             double var = 0.0;

//             for (int ii = 0; ii < m1->columns; ii++)
//             {
//                 var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
//             }

//             res->m[idx] = var;
//         }
//     }
// }

__global__
void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res) {
    __shared__ float M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N[TILE_WIDTH][TILE_WIDTH];
    int block_x = blockIdx.x, block_y = blockIdx.y, thread_x = threadIdx.x, thread_y = threadIdx.y,
    row = block_y * TILE_WIDTH + thread_y, col = block_x * TILE_WIDTH + thread_x;
    float P = 0;

    for (int m = 0; m < (m1->columns - 1) / TILE_WIDTH + 1; ++m) {
        if (row < m1->rows && m * TILE_WIDTH + thread_x < m1->columns){
            M[thread_y][thread_x] = m1->m[row * m1->columns + m * TILE_WIDTH + thread_x];
        }
        else{
            M[thread_y][thread_x] = 0;
        }
        if (col < m2->columns && m * TILE_WIDTH + thread_y < m2->rows){
            N[thread_y][thread_x] = m2->m[(m * TILE_WIDTH + thread_y) * m2->columns + col];
        }
        else{
            N[thread_y][thread_x] = 0;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i){
            P += M[thread_y][i] * N[i][thread_x];
        }
        __syncthreads();
    }
    if (row < m1->rows && col < m2->columns){
        res->m[row * m2->columns + col] = P;
    }
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}