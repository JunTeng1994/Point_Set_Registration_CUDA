#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <limits>
#include <assert.h>

#define M_PI 3.14159265358979323846

struct myMatrix
{
	int cols;
	int rows;
	double* Mat;
};

__global__ void init(double *m_g, double *source, double beta, int sourows);
__global__ void probability1(double* target, double* source, double* p,
	double ksig, int tarrows, int sourows);
__global__ void probability2(double* p, double* sp, double* pt1, double*l, double outlier_tmp, int tarrows, int sourows);
__global__ void probability3(double* target, double* p, double* sp, double* p1, double* p1_max, double* px, double* correspondence,
	int tarrows, int sourows);
__global__ void probability4(double *temp_l, double *l, double res_sigma2, int tarrows);
__global__ void com_result1(double *x, double *y, double *p1, double *px, double *m_g, double *source, double lambda, double sigma2, int sourows);
__global__ void com_result2(double *source, double *result, double *m_g, double *m_w, double *x, int sourows);
__global__ void com_result3(double* target, double *result, double* pt1, double *p1, double* cache, int tarrows, int sourows);
__global__ void com_result4(double *result, double *cache, double *px, double *sigma2, double np, int num, int sourows);
cudaError_t ResultCompute(myMatrix* target, myMatrix* source, double* result,
	double sigma2, double outliers, double tolerance, int max_iter, double m_beta, double m_lambda);