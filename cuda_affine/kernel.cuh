#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <limits>
#include <algorithm>
#include <assert.h>

#define M_PI 3.14159265358979323846

struct myMatrix
{
	int cols;
	int rows;
	double* Mat;
};

__global__ void Probability1(double* target, double* source, double* p, double ksig, int tarrows, int sourows);
__global__ void Probability2(double* p, double* sp, double* pt1, double*l, double outlier_tmp, int tarrows, int sourows);
__global__ void Probability3(double* target, double* p, double* sp, double* p1, double* px, int tarrows, int sourows);

__global__ void Computel(double* temp_l, double* cache, int tarrows);
__global__ void ComputeNp(double* p1, double* cache, int sourows);

__global__ void ComputeMu(double* target, double* source, double* p1, double* pt1,double *mu_x, double *mu_y, int tarrows, int sourows, double np);
__global__ void ComputeB(double *b1, double *b2, double *px, double *p1, double *mu_x, double *mu_y, double *source, double np, int sourows);
__global__ void ComputeTrans(double *b1, double *b2, double *inv_b2, double *mu_x, double *mu_y, double *transform, double *translation);
__global__ void ComputeSigma1(double* target, double* pt1, double* cache, int tarrows);
__global__ void ComputeSigma2(double *cache, double *mu_x, double *mu_y, double *b1, double *transform, double *sigma2, double np, int num);
__global__ void ComputeRes(double* source, double* transform, double* translation, double* result, int sourows);

cudaError_t Compute(myMatrix* target, myMatrix* source, double* result, double* transform, double* translation, double sigma2, double outliers, double tolerance, int max_iter);