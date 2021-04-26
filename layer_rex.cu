#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;
	srand (1);
	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		// h_bias[i] = 0.0f;

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			// h_weight[i][j] = 0.00f;
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);
	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}


__device__ float sigmoid(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_sigmoid(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		output[idx] = sigmoid(input[idx]);
		n++;

		if(n == cnt + 1) break;
	}
}

__global__ void makeError(float* err, float* output, unsigned int* Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	err[blockIdx.x * 10 + threadIdx.x] = ((Y[blockIdx.x] == threadIdx.x ? 1.0f : 0.0f) - output[blockIdx.x * 10 + threadIdx.x]);
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		// output[idx] += dt * grad[idx];
		atomicAdd(&output[idx], dt * grad[idx]);

		n++;
		if(n == cnt + 1) break;
	}
	
}

__global__ void fp_preact_c1(float input[batch_size][28][28], float preact[batch_size][6][24][24], float weight[6][5][5])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 5*5*6*24*24;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain){
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size) % 5);
		const int i2 = ((idx /= 5	) % 5);
		const int i3 = ((idx /= 5	) % 6);
		const int i4 = ((idx /= 6	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&preact[i0][i3][i4][i5], weight[i3][i1][i2] * input[i0][i4 + i1][i5 + i2]);
		
		n++;
		if(n == cnt + 1) break;
	}
}



__global__ void fp_bias_c1(float preact[batch_size][6][24][24], float bias[6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*24*24;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain){
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		// preact[i0][i1][i2][i3] += bias[i1];
		atomicAdd(&preact[i0][i1][i2][i3], bias[i1]);

		n++;
		if(n == cnt + 1) break;
	}
}


__global__ void fp_preact_c2(float input[batch_size][6][24][24], float preact[batch_size][6][12][12], float weight[6][2][2])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 2*2*6*12*12;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain){
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 2);
		const int i2 = ((idx /= 2	) % 2);
		const int i3 = ((idx /= 2	) % 6);
		const int i4 = ((idx /= 6	) % 12);
		const int i5 = ((idx /= 12	) % 12);

		atomicAdd(&preact[i0][i3][i4][i5], weight[i3][i1][i2] * input[i0][i3][i4 * 2 + i1][i5 * 2 + i2]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void fp_bias_c2(float preact[batch_size][6][12][12], float bias[6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*12*12;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size) % 6);
		const int i2 = ((idx /= 6	) % 12);
		const int i3 = ((idx /= 12	) % 12);

		// preact[i0][i1][i2][i3] += bias[i1];
		atomicAdd(&preact[i0][i1][i2][i3], bias[i1]);

		n++;
		if(n == cnt + 1) break;
	}
}


__global__ void fp_preact_c3(float input[batch_size][6][12][12], float preact[batch_size][6][6][6], float weight[6][2][2])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 2*2*6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size) % 2);
		const int i2 = ((idx /= 2	) % 2);
		const int i3 = ((idx /= 2	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i0][i3][i4][i5], weight[i3][i1][i2] * input[i0][i3][i4 * 2 + i1][i5 * 2 + i2]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void fp_bias_c3(float preact[batch_size][6][6][6], float bias[6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		// preact[i0][i1][i2][i3] += bias[i1];
		atomicAdd(&preact[i0][i1][i2][i3],bias[i1]);
		
		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void fp_preact_f(float input[batch_size][6][6][6], float preact[batch_size][10], float weight[10][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 10*6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i0][i1], weight[i1][i2][i3][i4] * input[i0][i2][i3][i4]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void fp_bias_f(float preact[batch_size][10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 10;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1) % batch_size);
		const int i1 = ((idx /= batch_size) % 10);
		// preact[i0][i1] += bias[i1];
		atomicAdd(&preact[i0][i1], bias[i1]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[batch_size][10], float p_output[batch_size][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 10*6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1) % batch_size);
		const int i1 = ((idx /= batch_size) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		
		atomicAdd(&d_weight[i1][i2][i3][i4], d_preact[i0][i1] * p_output[i0][i2][i3][i4]);
		//d_weight[i1][i2][i3][i4] = d_preact[i0][i1] * p_output[i0][i2][i3][i4];

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[batch_size][10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 10;

	// for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
	// 	bias[idx] += dt * d_preact[idx];
	// }

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1) % batch_size);
		const int i1 = ((idx /= batch_size) % 10);

		atomicAdd(&bias[i1], dt * d_preact[i0][i1]);
		//bias[i1] += dt * d_preact[i0][i1];

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_output_c3(float d_output[batch_size][6][6][6], float n_weight[10][6][6][6], float nd_preact[batch_size][10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 10*6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1) % batch_size);
		const int i1 = ((idx /= batch_size) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i0][i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i0][i1]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_preact_c3(float d_preact[batch_size][6][6][6], float d_output[batch_size][6][6][6], float preact[batch_size][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		const float o = sigmoid(preact[i0][i1][i2][i3]);

		d_preact[i0][i1][i2][i3] = d_output[i0][i1][i2][i3] * o * (1 - o);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_weight_c3(float d_weight[6][2][2], float d_preact[batch_size][6][6][6], float p_output[batch_size][6][12][12])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*2*2*6*6*6;
	const float d = pow(6.0f, 3.0f);

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 2);
		const int i3 = ((idx /= 2	) % 2);
		const int i4 = ((idx /= 2	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i0][i4][i5][i6] * p_output[i0][i4][i5 * 2 + i2][i6 * 2 + i3]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_bias_c3(float bias[1], float d_preact[batch_size][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*6*6;
	const float d = pow(6.0f, 3.0f);

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain){
		int idx = n * size + pos;
		const int i0 = ((idx /= 1) % batch_size);
		const int i1 = ((idx /= batch_size ) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		atomicAdd(&bias[0], dt * d_preact[i0][i1][i2][i3] / d);

		n++;
		if(n == cnt + 1) break;
	}
}



__global__ void bp_output_c2(float d_output[batch_size][6][12][12], float n_weight[6][2][2], float nd_preact[batch_size][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*2*2*6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain){
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 2);
		const int i3 = ((idx /= 2	) % 2);
		const int i4 = ((idx /= 2	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i0][i4][i5 * 2 + i2][i6 * 2 + i3], n_weight[i1][i2][i3] * nd_preact[i0][i4][i5][i6]);

		n++;
		if(n == cnt + 1) break;
	}
}


__global__ void bp_preact_c2(float d_preact[batch_size][6][12][12], float d_output[batch_size][6][12][12], float preact[batch_size][6][12][12])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*12*12;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 12);
		const int i3 = ((idx /= 12	) % 12);

		const float o = sigmoid(preact[i0][i1][i2][i3]);

		d_preact[i0][i1][i2][i3] = d_output[i0][i1][i2][i3] * o * (1 - o);

		n++;
		if(n == cnt + 1) break;
	}
	
}

__global__ void bp_weight_c2(float d_weight[6][2][2], float d_preact[batch_size][6][12][12], float p_output[batch_size][6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*2*2*12*12;
	const float d = pow(6.0f, 3.0f);

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain){
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 2);
		const int i3 = ((idx /= 2	) % 2);
		const int i4 = ((idx /= 2	) % 6);
		const int i5 = ((idx /= 6	) % 12);
		const int i6 = ((idx /= 12	) % 12);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i0][i4][i5][i6] * p_output[i0][i4][i5 * 2 + i2][i6 * 2 + i3]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_bias_c2(float bias[6], float d_preact[batch_size][6][12][12])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*12*12;
	const float d = pow(6.0f, 3.0f);

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 12);
		const int i3 = ((idx /= 12	) % 12);

		atomicAdd(&bias[i1], dt * d_preact[i0][i1][i2][i3] / d);

		n++;
		if(n == cnt + 1) break;
	}
}



__global__ void bp_output_c1(float d_output[batch_size][6][24][24], float n_weight[6][2][2], float nd_preact[batch_size][6][12][12])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*2*2*6*12*12;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 2);
		const int i3 = ((idx /= 2	) % 2);
		const int i4 = ((idx /= 2	) % 6);
		const int i5 = ((idx /= 6	) % 12);
		const int i6 = ((idx /= 12	) % 12);

		atomicAdd(&d_output[i0][i4][i5 * 2 + i2][i6 * 2 + i3], n_weight[i1][i2][i3] * nd_preact[i0][i4][i5][i6]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_preact_c1(float d_preact[batch_size][6][24][24], float d_output[batch_size][6][24][24], float preact[batch_size][6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*24*24;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain){
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		const float o = sigmoid(preact[i0][i1][i2][i3]);

		d_preact[i0][i1][i2][i3] = d_output[i0][i1][i2][i3] * o * (1 - o);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[batch_size][6][24][24], float p_output[batch_size][28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*5*5*24*24;
	const float d = pow(24.0f, 2.0f);

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 5);
		const int i3 = ((idx /= 5	) % 5);
		const int i4 = ((idx /= 5	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i0][i1][i4][i5] * p_output[i0][i4 + i2][i5 + i3] / d);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void bp_bias_c1(float bias[6], float d_preact[batch_size][6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*24*24;
	const float d = pow(24.0f, 2.0f);

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		atomicAdd(&bias[i1], dt * d_preact[i0][i1][i2][i3] / d);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void fp_add_res(float preact1[batch_size][6][6][6],float preact2[batch_size][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int N = batch_size * 6*6*6;
	const float d = pow(6.0f, 3.0f);

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		atomicAdd(&preact1[i0][i1][i2][i3], preact2[i0][i1][i2][i3] + preact1[i0][i1][i2][i3] );

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void fp_preact_r(float input[batch_size][6][24][24], float preact[batch_size][6][6][6], float weight[1][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 4*4*6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 4);
		const int i2 = ((idx /= 4	) % 4);
		const int i3 = ((idx /= 4	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i0][i3][i4][i5], weight[0][i1][i2] * input[i0][i3][i4 * 4 + i1][i5 * 4 + i2]);

		n++;
		if(n == cnt + 1) break;
	}
}

__global__ void fp_bias_r(float preact[batch_size][6][6][6], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = batch_size * 6*6*6;

	int cnt = N / size;
	int remain = N % size;

	int n = 0;
	while(n < cnt || pos < remain) {
		int idx = n * size + pos;
		const int i0 = ((idx /= 1	) % batch_size);
		const int i1 = ((idx /= batch_size	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		// preact[i0][i1][i2][i3] += bias[0];
		atomicAdd(&preact[i0][i1][i2][i3],bias[0]);

		n++;
		if(n == cnt + 1) break;
	}
}
