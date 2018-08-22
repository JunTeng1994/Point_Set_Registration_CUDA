
#include "kernel.cuh"

__global__ void init(double *m_g, double *m_w, double *source, double k, int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < sourows && j < sourows)
	{
		double razn = pow(source[3 * i] - source[3 * j], 2) + pow(source[3 * i + 1] - source[3 * j + 1], 2) +
			pow(source[3 * i + 2] - source[3 * j + 2], 2);
		m_g[i*sourows + j] = exp(razn / k);
		if (j < 3)
		{
			m_w[i * 3 + j] = 0;
		}
	}
}

__global__ void probability1(double* target, double* source, double* p,
	double ksig, int tarrows, int sourows)
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

__global__ void probability2(double* p, double* sp, double* pt1, double* temp_l, double outlier_tmp, int tarrows, int sourows)
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
		temp_l[i] = -log(sp[i]);
	}
}

__global__ void probability3(double* target, double* p, double* sp, double* p1, double* p1_max, double* px, double* correspondence,
	int tarrows, int sourows)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (j < sourows)
	{
		double temp;
		p1_max[j] = 0;
		px[j * 3] = 0;
		px[j * 3 + 1] = 0;
		px[j * 3 + 2] = 0;
		p1[j] = 0;
		correspondence[j] = 0;
		for (int i = 0; i < tarrows; i++)
		{
			temp = p[i*sourows + j] / sp[i];
			p1[j] += temp;
			px[j * 3] += target[i * 3] * temp;
			px[j * 3 + 1] += target[i * 3 + 1] * temp;
			px[j * 3 + 2] += target[i * 3 + 2] * temp;
			if (temp > p1_max[j])
			{
				correspondence[j] = i;
				p1_max[j] = temp;
			}
		}
	}
}

__global__ void probability4(double *temp_l, double *l, double res_sigma2, int tarrows)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid == 0){
		*l = 0;
		for (int i = 0; i < tarrows; i++)
		{
			*l += temp_l[i];
		}
		*l += 3 * tarrows * log(res_sigma2) / 2;
	}
}

__global__ void modify1(double *m_g, double *m_w, double *mid1, int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < 3 && j < sourows)
	{
		mid1[i*sourows + j] = 0;
		for (int tid = 0; tid < sourows; tid++)
		{
			mid1[i*sourows + j] += m_w[tid * 3 + i] * m_g[tid*sourows + j];
		}
	}
}
__global__ void modify2(double *m_w, double *mid1, double *mid2, int sourows, double *l, double lambda)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < 3)
	{
		mid2[tid] = 0;
		for (int i = 0; i < sourows; i++)
		{
			mid2[tid] += mid1[tid*sourows + i] * m_w[i * 3 + tid];
		}
	}
	__syncthreads();

	if (tid == 0)
	{
		*l += lambda / 2.0 * (mid2[0]+mid2[1]+mid2[2]);
	}
}

__global__ void com_result1(double *A, double *b, double *p1, double *px, double *m_g, double *source, double lambda, double sigma2,int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < sourows && j < sourows)
	{
		if (i == j){
			A[j * sourows + i] = p1[i] * m_g[i*sourows + j] + lambda*sigma2;
		}
		else{
			A[j * sourows + i] = p1[i] * m_g[i*sourows + j];
		}
		if (j < 3)
		{
			b[j * sourows + i] = px[i * 3 + j] - p1[i] * source[i * 3 + j];
		}
	}
}

__global__ void com_result2(double *source, double *result, double *m_g, double *m_w, double *x, int sourows)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < sourows && j < 3)
	{
		double temp = 0;
		for (int tid = 0; tid < sourows; tid++)
		{
			temp += m_g[i*sourows + tid] * x[j * sourows + tid];
		}
		result[i * 3 + j] = source[i * 3 + j] + temp;
		m_w[i * 3 + j] = x[j*sourows + i];
	}
}

__global__ void com_result3(double* target, double *result, double* pt1, double *p1, double* cache, int tarrows, int sourows)
{
	__shared__ double temp[256];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tempIndex = threadIdx.x;
	temp[tempIndex] = 0;
	if (tid < tarrows + sourows)
	{
		if (tid < tarrows){
			temp[tempIndex] = (pow(target[tid * 3], 2) + pow(target[tid * 3 + 1], 2) + pow(target[tid * 3 + 2], 2))* pt1[tid];
		}
		else{
			int i = tid - tarrows;
			temp[tempIndex] = (pow(result[i * 3], 2) + pow(result[i * 3 + 1], 2) + pow(result[i * 3 + 2], 2))* p1[i];
		}
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

__global__ void com_result4(double *result, double *cache, double *px, double *sigma2, double np, int num, int sourows)
{
	int tid = threadIdx.x;
	__shared__ double temp[3];
	if (tid < 3)
	{
		temp[tid] = 0;
		for (int i = 0; i < sourows; i++)
		{
			temp[tid] += px[i * 3 + tid] * result[i * 3 + tid];
		}
	}
	__syncthreads();
	if (tid == 0)
	{
		*sigma2 = 0;
		for (int i = 0; i < num; i++)
		{
			*sigma2 += cache[i];
		}
  		*sigma2 += -2 * (temp[0] + temp[1] + temp[2]);
		*sigma2 /= np * 3;
		*sigma2 = abs(*sigma2);
	}
}

cudaError_t ResultCompute(PointsMatrix* target, PointsMatrix* source, double* result,
	double m_sigma2, double m_outliers, double m_tolerance, int max_iter, double m_beta, double m_lambda)
{
	int tarrows = target->rows;
	int sourows = source->rows;
	double k = -2.0 * m_beta * m_beta;
	double ntol = m_tolerance + 10.0;
	double l = 0.0;
	int iter = 0;
	double res_sigma2 = m_sigma2;

	dim3 initblocks((sourows + 31) / 32, (sourows + 31) / 32);
	dim3 initthreads(32, 32);
	dim3 problocks1((tarrows + 31) / 32, (sourows + 31) / 32);
	dim3 prothreads1(32, 32);
	dim3 modblocks1(3, (sourows + 31) / 32);
	dim3 modthreads1(1, 32);
	dim3 resblocks1((sourows + 31) / 32, (sourows + 31) / 32);
	dim3 resthreads1(32, 32);
	dim3 resblocks2((sourows + 31) / 32, 1);
	dim3 resthreads2(32, 3);

	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	int m = sourows;
	int lda = m;
	int ldb = m;
	int nrhs = target->cols;
	int  lwork = 0;
	int info_gpu = 0;
	const double one = 1;

	double *dev_target, *dev_source, *dev_result;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&dev_target, tarrows * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_source, sourows * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_result, sourows * 3 * sizeof(double));
	cudaStatus = cudaMemcpy(dev_target, target->Mat, tarrows * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_source, source->Mat, sourows * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_result, source->Mat, sourows * 3 * sizeof(double), cudaMemcpyHostToDevice);

	double *dev_m_g,*dev_m_w;
	cudaStatus = cudaMalloc((void**)&dev_m_g,sourows*sourows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_m_w, sourows * 3 * sizeof(double));

	double *dev_pt1, *dev_p1, *dev_px, *dev_correspondence, *dev_temp_l;
	double *dev_p, *dev_sp, *dev_p1_max, *dev_l;
	cudaStatus = cudaMalloc((void**)&dev_pt1, tarrows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_p1, sourows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_px, sourows * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_correspondence, sourows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_temp_l, tarrows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_p, tarrows*sourows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_sp, tarrows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_p1_max, sourows*sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_l, sizeof(double));

	double *dev_mid1, *dev_mid2;
	cudaStatus = cudaMalloc((void**)&dev_mid1, sourows * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_mid2, 3 * sizeof(double));

	double *dev_A, *dev_b, *dev_x;
	cudaStatus = cudaMalloc((void**)&dev_A, sourows * sourows * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_b, sourows * 3 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_x, sourows * 3 * sizeof(double));

	double *d_tau = NULL; 
	int *devInfo = NULL;
	double *d_work = NULL;
	cudaStatus = cudaMalloc((void**)&d_tau, sizeof(double)* m);
	cudaStatus = cudaMalloc((void**)&devInfo, sizeof(int));

	double *p1 = (double*)malloc(sourows*sizeof(double));

	double *dev_cache, *dev_sigma2;
	cudaStatus = cudaMalloc((void**)&dev_cache, ((tarrows + sourows + 255) / 256)* sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_sigma2, sizeof(double));

	init << <initblocks, initthreads >> >(dev_m_g, dev_m_w, dev_source, k, sourows);
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

	/*double *m_g = (double*)malloc(sourows*sourows*sizeof(double));
	cudaStatus = cudaMemcpy(m_g, dev_m_g, sourows*sourows*sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		printf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	printf("m_g = %f\n", m_g[1]);*/
	while (iter < max_iter && ntol > m_tolerance &&
		res_sigma2 > 10 * std::numeric_limits<double>::epsilon())
	{
		double ksig = -2.0 * res_sigma2;
		int cols = target->cols;
		double outlier_tmp =
			(m_outliers * source->rows * std::pow(-ksig * M_PI, 0.5 * cols)) /
			((1 - m_outliers) * target->rows);

		//º∆À„probabilities
		probability1 << <problocks1, prothreads1 >> >(dev_target, dev_result, dev_p, ksig, tarrows, sourows);
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
		probability2 << <(tarrows + 31) / 32, 32 >> >(dev_p, dev_sp, dev_pt1, dev_temp_l, outlier_tmp, tarrows, sourows);
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
		probability3 << <(sourows + 31) / 32, 32 >> >(dev_target, dev_p, dev_sp, dev_p1, dev_p1_max, dev_px, dev_correspondence, tarrows, sourows);
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
		probability4 << <1, 1 >> >(dev_temp_l, dev_l, res_sigma2, tarrows);
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
		

		modify1 << <modblocks1, modthreads1 >> >(dev_m_g, dev_m_w, dev_mid1, sourows);
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
		modify2 << <1, 3 >> >(dev_m_w, dev_mid1, dev_mid2, sourows, dev_l, m_lambda);
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
		double pro_l = 0;
		cudaStatus = cudaMemcpy(&pro_l, dev_l, sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		printf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
		}

		ntol = std::abs((pro_l - l) / pro_l);
		l = pro_l;
		
		com_result1 << <resblocks1, resthreads1 >> >(dev_A, dev_b, dev_p1, dev_px, dev_m_g, dev_source, m_lambda, res_sigma2, sourows);
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

		cusolver_status = cusolverDnCreate(&cusolverH);
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

		cublas_status = cublasCreate(&cublasH);
		assert(CUBLAS_STATUS_SUCCESS == cublas_status);

		// step 3: query working space of geqrf and ormqr
		cusolver_status = cusolverDnDgeqrf_bufferSize(
			cusolverH,
			m,
			m,
			dev_A,
			lda,
			&lwork);
		assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

		cudaStatus = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
		assert(cudaSuccess == cudaStatus);

		// step 4: compute QR factorization
		cusolver_status = cusolverDnDgeqrf(
			cusolverH,
			m,
			m,
			dev_A,
			lda,
			d_tau,
			d_work,
			lwork,
			devInfo);
		cudaStatus = cudaDeviceSynchronize();
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
		assert(cudaSuccess == cudaStatus);

		// check if QR is good or not
		cudaStatus = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStatus);

		//printf("after geqrf: info_gpu = %d\n", info_gpu);
		assert(0 == info_gpu);

		// step 5: compute Q^T*B
		cusolver_status = cusolverDnDormqr(
			cusolverH,
			CUBLAS_SIDE_LEFT,
			CUBLAS_OP_T,
			m,
			nrhs,
			m,
			dev_A,
			lda,
			d_tau,
			dev_b,
			ldb,
			d_work,
			lwork,
			devInfo);
		cudaStatus = cudaDeviceSynchronize();
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
		assert(cudaSuccess == cudaStatus);

		// check if QR is good or not
		cudaStatus = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStatus);

		//printf("after ormqr: info_gpu = %d\n", info_gpu);
		assert(0 == info_gpu);

		// step 6: compute x = R \ Q^T*B
		cublas_status = cublasDtrsm(
			cublasH,
			CUBLAS_SIDE_LEFT,
			CUBLAS_FILL_MODE_UPPER,
			CUBLAS_OP_N,
			CUBLAS_DIAG_NON_UNIT,
			m,
			nrhs,
			&one,
			dev_A,
			lda,
			dev_b,
			ldb);
		cudaStatus = cudaDeviceSynchronize();
		assert(CUBLAS_STATUS_SUCCESS == cublas_status);
		assert(cudaSuccess == cudaStatus);

		cudaStatus = cudaMemcpy(dev_x, dev_b, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost);
		assert(cudaSuccess == cudaStatus);

		
		if (d_work) cudaFree(d_work);

		if (cublasH) cublasDestroy(cublasH);
		if (cusolverH) cusolverDnDestroy(cusolverH);

		/*double *x = (double*)malloc(sourows*3*sizeof(double));
		cudaStatus = cudaMemcpy(x, dev_x, sourows*3*sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		printf("x = %f\n",x[sourows]);*/

		com_result2 << <resblocks2, resthreads2 >> >(dev_source, dev_result, dev_m_g, dev_m_w, dev_x, sourows);
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

		/*cudaStatus = cudaMemcpy(result, dev_result, sourows * 3 * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		printf("result = %f\n", result[0]);*/

		int num = (tarrows + sourows + 255) / 256;
		com_result3 << <num, 256 >> >(dev_target, dev_result, dev_pt1, dev_p1, dev_cache, tarrows, sourows);
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

		cudaStatus = cudaMemcpy(p1, dev_p1, sourows * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		double np = 0;
		for (int i = 0; i < source->rows; i++)
		{
			np += p1[i];
		}

		com_result4 << <1, 3 >> >(dev_result, dev_cache, dev_px, dev_sigma2, np, num, sourows);
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
		//printf("sigma2 = %f\n",res_sigma2);
		iter++;
	}
	printf("iter = %d\n", iter);
	printf("ntol = %f\n", ntol);
	printf("res_sigma2 = %f\n", res_sigma2);

	cudaStatus = cudaMemcpy(result, dev_result, source->rows * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		printf("%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}



Error:
	cudaFree(dev_source);
	cudaFree(dev_target);
	cudaFree(dev_result);
	cudaFree(dev_m_g);
	cudaFree(dev_A);
	cudaFree(dev_b);
	cudaFree(dev_cache);
	cudaFree(dev_correspondence);
	cudaFree(dev_l);
	cudaFree(dev_mid1);
	cudaFree(dev_mid2);
	cudaFree(dev_m_g);
	cudaFree(dev_m_w);
	cudaFree(dev_p);
	cudaFree(dev_p1);
	cudaFree(dev_p1_max);
	cudaFree(dev_pt1);
	cudaFree(dev_px);
	cudaFree(dev_result);
	cudaFree(dev_sigma2);
	cudaFree(dev_source);
	cudaFree(dev_sp);
	cudaFree(dev_target);
	cudaFree(dev_temp_l);
	cudaFree(dev_x);

	return cudaStatus;
}