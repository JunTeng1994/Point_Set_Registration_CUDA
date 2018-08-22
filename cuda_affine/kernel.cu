
#include "kernel.cuh"

/*__global__ void PointsMean(double *points, double *means, int rows)
{
	int tid = threadIdx.x;
	if (tid < 3){
		for (int i = 0; i < rows; i++)
		{
			means[tid] += points[i * 3 + tid];
		}
		means[tid] /= rows;
	}
}

__global__ void PointsDis(double* points,double *means, int rows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < rows && j < 3)
	{
		points[i * 3 + j] -= means[j];
	}
}

__global__ void PointsScale(double *points, double *cache, int rows)
{
	__shared__ double temp[1024];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tempIndex = threadIdx.x;
	temp[tempIndex] = 0;
	if (tid < rows)
	{
		temp[tempIndex] = pow(points[tid],2)/rows;
	}
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i)
		{
			temp[tempIndex] += temp[tempIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (tempIndex == 0)
	{
		cache[blockIdx.x] = temp[0];
	}
}

__global__ void PointsNorm(double *points,double scale, int rows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < rows && j < 3)
	{
		points[i * 3 + j] /= scale;
	}

}*/

__global__ void Probability1(double* target, double* source, double* p, double ksig, int tarrows, int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < tarrows && j < sourows)
	{
		double razn = pow(target[3 * i] - source[3 * j], 2) + pow(target[3 * i + 1] - source[3 * j + 1], 2) +
			pow(target[3 * i + 2] - source[3 * j + 2], 2);
		p[i*sourows + j] = exp(razn / ksig);
	}
}

__global__ void Probability2(double* p, double* sp, double* pt1, double*l, double outlier_tmp, int tarrows, int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < tarrows)
	{
		sp[i] = 0;
		for (int j = 0; j < sourows; j++)
		{
			sp[i] += p[i*sourows + j];
		}
		sp[i] += outlier_tmp;
		pt1[i] = 1 - outlier_tmp / sp[i];
		l[i] = -log(sp[i]);
	}
}

__global__ void Probability3(double* target, double* p, double* sp, double* p1, double* px, int tarrows, int sourows)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (j < sourows)
	{
		double temp;
		px[j * 3] = 0;
		px[j * 3 + 1] = 0;
		px[j * 3 + 2] = 0;
		p1[j] = 0;
		for (int i = 0; i < tarrows; i++)
		{
			temp = p[i*sourows + j] / sp[i];
			p1[j] += temp;
			px[j * 3] += target[i * 3] * temp;
			px[j * 3 + 1] += target[i * 3 + 1] * temp;
			px[j * 3 + 2] += target[i * 3 + 2] * temp;
		}
	}
}

__global__ void Computel(double* temp_l, double* cache, int tarrows)
{
	__shared__ double temp[1024];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tempIndex = threadIdx.x;
	temp[tempIndex] = 0;
	if (tid < tarrows)
	{
		temp[tempIndex] = temp_l[tid];
	}
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i)
		{
			temp[tempIndex] += temp[tempIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (tempIndex == 0)
	{
		cache[blockIdx.x] = temp[0];
	}
}

__global__ void ComputeNp(double* p1, double* cache, int sourows)
{
	__shared__ double temp[1024];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tempIndex = threadIdx.x;
	temp[tempIndex] = 0;
	if (tid < sourows)
	{
		temp[tempIndex] = p1[tid];
	}
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i)
		{
			temp[tempIndex] += temp[tempIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (tempIndex == 0)
	{
		cache[blockIdx.x] = temp[0];
	}
}

__global__ void ComputeMu(double* target, double* source, double* p1, double* pt1, double *mu_x, double *mu_y, int tarrows, int sourows, double np)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < 6)
	{
		if (tid < 3)
		{
			int i = tid;
			mu_x[i] = 0;
			for (int dy = 0; dy < tarrows; dy++)
			{
				mu_x[i] += target[dy * 3 + i] * pt1[dy] / np;
			}
		}
		else
		{
			int i = tid - 3;
			mu_y[i] = 0;
			for (int dy = 0; dy < sourows; dy++)
			{
				mu_y[i] += source[dy * 3 + i] * p1[dy] / np;
			}
		}
	}
}

__global__ void ComputeB(double *b1, double *b2, double *px, double *p1,double *mu_x, double *mu_y, double *source, double np, int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < 3 && j < 6)
	{
		if (j < 3)
		{
			b1[i * 3 + j] = 0;
			for (int dy = 0; dy < sourows; dy++)
			{
				b1[i * 3 + j] += px[dy * 3 + i] * source[dy * 3 + j];
			}
			b1[i * 3 + j] -= np * mu_x[i] * mu_y[j];
		}
		else
		{
			int x = i;
			int y = j - 3;
			b2[x * 3 + y] = 0;
			for (int dy = 0; dy < sourows; dy++)
			{
				b2[x * 3 + y] += source[dy * 3 + x] * p1[dy] * source[dy * 3 + y];
			}
			b2[x * 3 + y] -= np * mu_y[x] * mu_y[y];
		}
	}
}

__global__ void ComputeTrans(double *b1,double *b2, double *inv_b2, double *mu_x, double *mu_y, double *transform, double *translation)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < 3 && j < 3)
	{
		double temp[4];
		double det = b2[0] * b2[4] * b2[8] + b2[3] * b2[7] * b2[2] + b2[6] * b2[1] * b2[5]
			- b2[0] * b2[7] * b2[5] - b2[3] * b2[1] * b2[8] - b2[6] * b2[4] * b2[2];
		int tid = 0;
		for (int x = 0; x < 3; x++){
			if (x != j){
				for (int y = 0; y < 3; y++){
					if (y != i){
						temp[tid] = b2[x * 3 + y];
						tid++;
					}
				}
			}
		}
		inv_b2[i * 3 + j] = pow(-1.0, i + j) * (temp[0] * temp[3] - temp[1] * temp[2]) / det;

		__syncthreads();

		transform[i * 3 + j] = b1[i * 3] * inv_b2[j] + b1[i * 3 + 1] * inv_b2[3 + j] + b1[i * 3 + 2] * inv_b2[6 + j];

		__syncthreads();

		if (j == 0)
		{
			translation[i] = mu_x[i] -
				(transform[i * 3] * mu_y[0] + transform[i * 3 + 1] * mu_y[1] + transform[i * 3 + 2] * mu_y[2]);
		}
	}
}

__global__ void ComputeSigma1(double* target, double* pt1, double* cache, int tarrows)
{
	__shared__ double temp[1024];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tempIndex = threadIdx.x;
	temp[tempIndex] = 0;
	if (tid < tarrows)
	{
		temp[tempIndex] = (pow(target[tid * 3], 2) + pow(target[tid * 3 + 1], 2) + pow(target[tid * 3 + 2], 2))* pt1[tid];
	}
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i)
		{
			temp[tempIndex] += temp[tempIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (tempIndex == 0)
	{
		cache[blockIdx.x] = temp[0];
	}
}

__global__ void ComputeSigma2(double *cache, double *mu_x, double *mu_y, double *b1, double *transform, double *sigma2, double np, int num)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid == 0)
	{
		*sigma2 = 0;
		for (int i = 0; i < num; i++)
		{
			*sigma2 += cache[i];
		}
		for (int i = 0; i < 3; i++)
		{
			*sigma2 += -np * mu_x[i] * mu_x[i] -
				(b1[i * 3] * transform[i] + b1[i * 3 + 1] * transform[3 + i] + b1[i * 3 + 2] * transform[6 + i]);
		}
		*sigma2 /= np * 3;
	}
}

__global__ void ComputeRes(double* source, double* transform, double* translation, double* result, int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < sourows && j < 3)
	{
		result[i * 3 + j] = source[i * 3] * transform[j*3] + source[i * 3 + 1] * transform[j*3 + 1] + source[i * 3 + 2] * transform[j*3 + 2] + translation[j];
	}	
}

/*__global__ void Denormalize(double *result, double *target, double *transform, double *translation, double *tar_means, double *sou_means, double scale, int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < 3 && j == 0)
	{
		translation[i] = scale*translation[i] + tar_means[i];
		for (int tid = 0; tid < 3; tid++)
		{
			translation[i] -= transform[i * 3 + tid] * sou_means[tid];
		}
	}
	if (i < sourows && j < 3)
	{
		result[i * 3 + j] = result[i * 3 + j] * scale + tar_means[j];
	}
}*/

cudaError_t Compute(myMatrix* target, myMatrix* source, double* result, double* transform, double* translation, double sigma2, double outliers, double tolerance, int max_iter)
{
	//define the initial value
	double ntol = tolerance + 10.0;
	double l = 0.0;
	int iter = 0;
	double res_sigma2 = sigma2;

	int tarrows = target->rows;
	int sourows = source->rows;
	
	dim3 tarnormblocks((tarrows + 31) / 32, 1);
	dim3 tarnormthreads(32, 3);
	dim3 sounormblocks((sourows + 31) / 32, 1);
	dim3 sounormthreads(32, 3);
	dim3 problocks((tarrows + 31) / 32, (sourows + 31) / 32);
	dim3 prothreads(32, 32);
	dim3 comblocks(1,1);
	dim3 comthreads(3,6);
	dim3 transblocks(1, 1);
	dim3 transthreads(3, 3);
	dim3 resblocks((sourows + 31) / 32,1);
	dim3 resthreads(32, 3);

	int num1 = (tarrows + 1023) / 1024;
	int num2 = (sourows + 1023) / 1024;

	//CPU memory allocation
	double *cache_tar = (double*)malloc(num1* sizeof(double));
	double *cache_sou = (double*)malloc(num2* sizeof(double));

	//GPU memory allocation
	cudaError_t cudaStatus;
	double *dev_target, *dev_source, *dev_result;
	cudaStatus = cudaMalloc((void**)&dev_target, tarrows*3*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_source, sourows*3*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_result, sourows*3*sizeof(double));
	cudaStatus = cudaMemcpy(dev_target, target->Mat, tarrows * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_source, source->Mat, sourows * 3 * sizeof(double), cudaMemcpyHostToDevice);
	

	double *dev_tar_means, *dev_sou_means;
	cudaStatus = cudaMalloc((void**)&dev_tar_means, 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_sou_means, 3 * sizeof(double));
	double *dev_cache_tar, *dev_cache_sou, *dev_sigma2;
	cudaStatus = cudaMalloc((void**)&dev_cache_tar, num1* sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_cache_sou, num2* sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_sigma2, sizeof(double));

	/*PointsMean << <1, 3 >> >(dev_target, dev_tar_means, tarrows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	PointsDis << <tarnormblocks, tarnormthreads >> >(dev_target, dev_tar_means, tarrows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	PointsScale << <num1, 1024 >> >(dev_target, dev_cache_tar, tarrows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpy(cache_tar, dev_cache_tar, num1 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		printf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	double scale_tar = 0;
	for (int i = 0; i < num1; i++)
	{
		scale_tar += cache_tar[i];
	}
	PointsMean << <1, 3 >> >(dev_source, dev_sou_means, sourows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	PointsDis << <sounormblocks, sounormthreads >> >(dev_source, dev_sou_means, sourows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	PointsScale << <num2, 1024 >> >(dev_source, dev_cache_sou, sourows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpy(cache_sou, dev_cache_sou, num2 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		printf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	double scale_sou = 0;
	for (int i = 0; i < num1; i++)
	{
		scale_sou += cache_tar[i];
	}

	double scale = std::max(scale_tar, scale_sou);
	PointsNorm << <tarnormblocks, tarnormthreads >> >(dev_target, scale, tarrows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	PointsNorm << <sounormblocks, sounormthreads >> >(dev_source, scale, sourows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}*/

	cudaStatus = cudaMemcpy(dev_result, source->Mat, sourows * 3 * sizeof(double), cudaMemcpyHostToDevice);

	double *dev_pt1, *dev_p1, *dev_px, *dev_l;
	double *dev_p, *dev_sp;
	cudaStatus = cudaMalloc((void**)&dev_pt1, tarrows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_p1, sourows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_px, sourows*3*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_l, tarrows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_p, tarrows*sourows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_sp, tarrows*sizeof(double));

	double *dev_mu_x, *dev_mu_y;
	double *dev_b1, *dev_b2, *dev_inv_b2;
	double *dev_transform, *dev_translation;
	cudaStatus = cudaMalloc((void**)&dev_mu_x, 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_mu_y, 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_b1, 3 * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_b2, 3 * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_inv_b2, 3 * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_transform, 3 * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_translation, 3 * sizeof(double));

	

	//loop caculation
	while (iter < max_iter && ntol > tolerance &&
		res_sigma2 > 10 * std::numeric_limits<double>::epsilon())
	{
		double ksig = -2.0 * res_sigma2;
		int cols = target->cols;
		double outlier_tmp =
			(outliers * sourows * std::pow(-ksig * M_PI, 0.5 * cols)) /
			((1 - outliers) * tarrows);
		
		//compute p
		Probability1 << <problocks, prothreads>> >(dev_target, dev_result, dev_p, ksig, tarrows, sourows);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		//compute sp and pt1
		Probability2 << <(tarrows + 31) / 32, 32 >> >(dev_p, dev_sp, dev_pt1, dev_l, outlier_tmp, tarrows, sourows);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		//compute px and p1
		Probability3 << <(sourows + 31) / 32, 32 >> >(dev_target, dev_p, dev_sp, dev_p1, dev_px, tarrows, sourows);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
        //compute l
		Computel << <num1, 1024 >> >(dev_l, dev_cache_tar, tarrows);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		cudaStatus = cudaMemcpy(cache_tar, dev_cache_tar, num1 * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		double pro_l = 0;
		for (int i = 0; i < num1; i++)
		{
			pro_l += cache_tar[i];
		}
		pro_l += cols * tarrows * std::log(res_sigma2) / 2;
		ntol = std::abs((pro_l - l) / pro_l);
		l = pro_l;
		//compute np
		ComputeNp<<<num2,1024>>>(dev_p1, dev_cache_sou, sourows);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		cudaStatus = cudaMemcpy(cache_sou, dev_cache_sou, num2 * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		double np = 0;
		for (int i = 0; i < num2; i++)
		{
			np += cache_sou[i];
		}
		//compute mu_x and mu_y
		ComputeMu<< <1, 6 >> >(dev_target, dev_source, dev_p1, dev_pt1, dev_mu_x, dev_mu_y, tarrows, sourows, np);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		//compute b1 and b2
		ComputeB<< <comblocks, comthreads >> >(dev_b1, dev_b2, dev_px, dev_p1, dev_mu_x, dev_mu_y, dev_source, np, sourows);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		//compute transform and translation
		ComputeTrans<< <transblocks, transthreads >> >(dev_b1, dev_b2, dev_inv_b2, dev_mu_x, dev_mu_y, dev_transform, dev_translation);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		// compute sigma2
		ComputeSigma1<< <num1, 1024 >> >(dev_target, dev_pt1, dev_cache_tar, tarrows);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		ComputeSigma2<< <1, 1 >> >(dev_cache_tar, dev_mu_x, dev_mu_y, dev_b1, dev_transform, dev_sigma2, np, num1);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		cudaStatus = cudaMemcpy(&res_sigma2, dev_sigma2, sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		//compute result points
		ComputeRes<< <resblocks, resthreads >> >(dev_source, dev_transform, dev_translation, dev_result, sourows);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}
		++iter;
	}

	//print the iterations, ntol and sigma2
	printf("iter = %d\n", iter);
	printf("ntol = %f\n",ntol);
	printf("res_sigma2 = %f\n",res_sigma2);

	/*Denormalize << <resblocks, resthreads >> >(dev_result, dev_target, dev_transform, dev_translation, dev_tar_means, dev_sou_means, scale, sourows);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}*/

	//return transform
	cudaStatus = cudaMemcpy(transform, dev_transform, 3 * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		printf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//return translation
	cudaStatus = cudaMemcpy(translation, dev_translation, 3  * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		printf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//return last result
	cudaStatus = cudaMemcpy(result, dev_result, sourows * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		printf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//free the memory
Error:
	free(cache_tar);
	free(cache_sou);

	cudaFree(dev_source);
	cudaFree(dev_target);
	cudaFree(dev_pt1);
	cudaFree(dev_p1);
	cudaFree(dev_px);
	cudaFree(dev_l);
	cudaFree(dev_p);
	cudaFree(dev_sp);
	cudaFree(dev_result);
	cudaFree(dev_b1);
	cudaFree(dev_b2);
	cudaFree(dev_cache_tar);
	cudaFree(dev_inv_b2);
	cudaFree(dev_mu_x);
	cudaFree(dev_mu_y);
	cudaFree(dev_transform);
	cudaFree(dev_translation);
	cudaFree(dev_sigma2);
	cudaFree(dev_cache_sou);
	cudaFree(dev_tar_means);
	cudaFree(dev_sou_means);

	return cudaStatus;
}