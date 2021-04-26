#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <omp.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static std::vector<Layer*> l_input;
static std::vector<Layer*> l_c1;
static std::vector<Layer*> l_c2;
static std::vector<Layer*> l_c3;
static std::vector<Layer*> l_f;
static std::vector<Layer*> l_r;

float* l_c1_weight;
float* l_c2_weight;
float* l_c3_weight;
float* l_f_weight;
float* l_r_weight;
float* l_c1_bias;
float* l_c2_bias;
float* l_c3_bias;
float* l_f_bias;
float* l_r_bias;

int deviceCount = 0;

static void learn();
static int* classify(float input[batch_size][28][28], int tid);
static void test(int tid);
static double forward_pass(float input[batch_size][28][28], int tid);
static double back_pass(int tid);

__global__ void weight_update(float* dest, float* weight, int N, int device_cnt)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	
	if(pos < N * device_cnt){
		int idx = pos % N;
		atomicAdd(&dest[idx], weight[pos]);
	}
}

__global__ void weight_average(float* weight, int N, int device_cnt)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	
	if(pos < N){
		weight[pos] /= device_cnt;
	}
}

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	fprintf(stdout ,"begin\n");
	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}

	loaddata();
	learn();

	fprintf(stdout, "begin test");
	test(0);

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(float input[batch_size][28][28], int tid)
{

	l_input[tid]->clear();
	l_c1[tid]->clear();
	l_c2[tid]->clear();
	l_c3[tid]->clear();
	l_f[tid]->clear();
	l_r[tid]->clear();
	clock_t start, end;
	start = clock();

	l_input[tid]->setOutput((float *)input);

	fp_preact_c1<<<2048, 1024>>>((float (*)[28][28])l_input[tid]->output, (float (*)[6][24][24])l_c1[tid]->preact, (float (*)[5][5])l_c1[tid]->weight);
	fp_bias_c1<<<2048, 1024>>>((float (*)[6][24][24])l_c1[tid]->preact, l_c1[tid]->bias);
	apply_sigmoid<<<2048, 1024>>>(l_c1[tid]->preact, l_c1[tid]->output, l_c1[tid]->O);

	fp_preact_r<<<2048, 1024>>>((float (*)[6][24][24])l_c1[tid]->preact, (float (*)[6][6][6])l_r[tid]->preact, (float (*)[4][4])l_r[tid]->weight);
	fp_bias_r<<<2048, 1024>>>((float (*)[6][6][6])l_r[tid]->preact, l_r[tid]->bias);

	fp_preact_c2<<<2048, 1024>>>((float (*)[6][24][24])l_c1[tid]->output, (float (*)[6][12][12])l_c2[tid]->preact, (float (*)[2][2])l_c2[tid]->weight);
	fp_bias_c2<<<2048, 1024>>>((float (*)[6][12][12])l_c2[tid]->preact, l_c2[tid]->bias);
	apply_sigmoid<<<2048, 1024>>>(l_c2[tid]->preact, l_c2[tid]->output, l_c2[tid]->O);

	fp_preact_c3<<<2048, 1024>>>((float (*)[6][12][12])l_c2[tid]->output, (float (*)[6][6][6])l_c3[tid]->preact, (float (*)[2][2])l_c3[tid]->weight);
	fp_bias_c3<<<2048, 1024>>>((float (*)[6][6][6])l_c3[tid]->preact, l_c3[tid]->bias);

	fp_add_res<<<2048, 1024>>>((float (*)[6][6][6])l_c3[tid]->preact, (float (*)[6][6][6])l_r[tid]->preact);
	
	apply_sigmoid<<<2048, 1024>>>(l_c3[tid]->preact, l_c3[tid]->output, l_c3[tid]->O);
	

	fp_preact_f<<<2048, 1024>>>((float (*)[6][6][6])l_c3[tid]->output, (float (*)[10])l_f[tid]->preact, (float (*)[6][6][6])l_f[tid]->weight);
	fp_bias_f<<<2048, 1024>>>((float (*)[10])l_f[tid]->preact, l_f[tid]->bias);
	apply_sigmoid<<<2048, 1024>>>(l_f[tid]->preact, l_f[tid]->output, l_f[tid]->O);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass(int tid)
{
	// fprintf(stdout, "start backward\n");
	//fprintf(stdout, "\n here \n");
	clock_t start, end;

	start = clock();

	bp_weight_f<<<2048, 1024>>>((float (*)[6][6][6])l_f[tid]->d_weight, (float (*)[10])l_f[tid]->d_preact, (float (*)[6][6][6])l_c3[tid]->output);
	bp_bias_f<<<2048, 1024>>>(l_f[tid]->bias, (float (*)[10])l_f[tid]->d_preact);
	
	bp_output_c3<<<2048, 1024>>>((float (*)[6][6][6])l_c3[tid]->d_output, (float (*)[6][6][6])l_f[tid]->weight, (float (*)[10])l_f[tid]->d_preact);
	bp_preact_c3<<<2048, 1024>>>((float (*)[6][6][6])l_c3[tid]->d_preact, (float (*)[6][6][6])l_c3[tid]->d_output, (float (*)[6][6][6])l_c3[tid]->preact);
	bp_weight_c3<<<2048, 1024>>>((float (*)[2][2])l_c3[tid]->d_weight, (float (*)[6][6][6])l_c3[tid]->d_preact, (float (*)[6][12][12])l_c2[tid]->output);
	bp_bias_c3<<<2048, 1024>>>(l_c3[tid]->bias, (float (*)[6][6][6])l_c3[tid]->d_preact);

	
	bp_output_c2<<<2048, 1024>>>((float (*)[6][12][12])l_c2[tid]->d_output, (float (*)[2][2])l_c3[tid]->weight, (float (*)[6][6][6])l_c3[tid]->d_preact);
	bp_preact_c2<<<2048, 1024>>>((float (*)[6][12][12])l_c2[tid]->d_preact, (float (*)[6][12][12])l_c2[tid]->d_output, (float (*)[6][12][12])l_c2[tid]->preact);
	bp_weight_c2<<<2048, 1024>>>((float (*)[2][2])l_c2[tid]->d_weight, (float (*)[6][12][12])l_c2[tid]->d_preact, (float (*)[6][24][24])l_c1[tid]->output);
	bp_bias_c2<<<2048, 1024>>>(l_c2[tid]->bias, (float (*)[6][12][12])l_c2[tid]->d_preact);

	
	bp_output_c1<<<2048, 1024>>>((float (*)[6][24][24])l_c1[tid]->d_output, (float (*)[2][2])l_c2[tid]->weight, (float (*)[6][12][12])l_c2[tid]->d_preact);
	bp_preact_c1<<<2048, 1024>>>((float (*)[6][24][24])l_c1[tid]->d_preact, (float (*)[6][24][24])l_c1[tid]->d_output, (float (*)[6][24][24])l_c1[tid]->preact);
	bp_weight_c1<<<2048, 1024>>>((float (*)[5][5])l_c1[tid]->d_weight, (float (*)[6][24][24])l_c1[tid]->d_preact, (float (*)[28][28])l_input[tid]->output);
	bp_bias_c1<<<2048, 1024>>>(l_c1[tid]->bias, (float (*)[6][24][24])l_c1[tid]->d_preact);


	apply_grad<<<2048, 1024>>>(l_f[tid]->weight, l_f[tid]->d_weight, l_f[tid]->M * l_f[tid]->N);
	apply_grad<<<2048, 1024>>>(l_c2[tid]->weight, l_c2[tid]->d_weight, l_c2[tid]->M * l_c2[tid]->N);
	apply_grad<<<2048, 1024>>>(l_c1[tid]->weight, l_c1[tid]->d_weight, l_c1[tid]->M * l_c1[tid]->N);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;

}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn()
{
	// train_cnt = 60000;
	cudaGetDeviceCount(&deviceCount);
	omp_set_num_threads(deviceCount);

	cublasHandle_t blas[deviceCount];
	l_input = std::vector<Layer*>(deviceCount);
	l_c1 = std::vector<Layer*>(deviceCount);
	l_c2 = std::vector<Layer*>(deviceCount);
	l_c3 = std::vector<Layer*>(deviceCount);
	l_f = std::vector<Layer*>(deviceCount);
	l_r = std::vector<Layer*>(deviceCount);

	float err;
	int iter = 20;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");


	#pragma omp parallel num_threads(deviceCount) 
	{
		int i = omp_get_thread_num();
		cudaSetDevice(i);
		l_input[i] = new Layer(0, 0, 28*28 * batch_size);
		l_c1[i] = new Layer(5*5, 6, 24*24*6 * batch_size);
		l_c2[i] = new Layer(2*2, 6, 12*12*6 * batch_size);
		l_c3[i] = new Layer(2*2, 6, 6*6*6 * batch_size);
		l_f[i] = new Layer(6*6*6, 10, 10 * batch_size);
		l_r[i] = new Layer(4*4,1,6*6*6 * batch_size);
		cublasCreate(&blas[i]);

		if(i == 0){
			l_c1_weight = new float[5*5*6*(deviceCount-1)];
			l_c2_weight = new float[2*2*6*(deviceCount-1)];
			// l_c3_weight = new float[2*2*6*(deviceCount-1)];
			l_f_weight = new float[6*6*6*10*(deviceCount-1)];
			// l_r_weight = new float[4*4*1*(deviceCount-1)];

			l_c1_bias = new float[6*(deviceCount-1)];
			l_c2_bias = new float[6*(deviceCount-1)];
			// l_c3_bias = new float[6*(deviceCount-1)];
			l_f_bias = new float[10*(deviceCount-1)];
			// l_r_bias = new float[1*(deviceCount-1)];
			
			cudaMalloc(&l_c1_weight, sizeof(float) * 5*5*6*(deviceCount-1));
			cudaMalloc(&l_c2_weight, sizeof(float) * 2*2*6*(deviceCount-1));
			// cudaMalloc(&l_c3_weight, sizeof(float) * 2*2*6*(deviceCount-1));
			cudaMalloc(&l_f_weight, sizeof(float) * 6*6*6*10*(deviceCount-1));
			// cudaMalloc(&l_r_weight, sizeof(float) * 4*4*1*(deviceCount-1));
			cudaMalloc(&l_c1_bias, sizeof(float) * 6*(deviceCount-1));
			cudaMalloc(&l_c2_bias, sizeof(float) * 6*(deviceCount-1));
			// cudaMalloc(&l_c3_bias, sizeof(float) * 6*(deviceCount-1));
			cudaMalloc(&l_f_bias, sizeof(float) * 10*(deviceCount-1));
			// cudaMalloc(&l_r_bias, sizeof(float) * 1*(deviceCount-1));
		}
	}

	cudaDeviceSynchronize();

	
	auto start_time = std::chrono::steady_clock::now();
	while (iter < 0 || iter-- > 0) {
		#pragma omp parallel num_threads(deviceCount) 
		{
			err = 0.0f;
			int tid = omp_get_thread_num();
			cudaSetDevice(tid);
			unsigned int* Y;
			cudaMalloc(&Y, sizeof(unsigned int) * batch_size);
			int batch_cnt = train_cnt / batch_size;
			for (int q = 0; q < batch_cnt; q+=deviceCount) {
				float tmp_err;
				int p = q + tid;
				float input[batch_size][28][28];
				unsigned int Y_host[batch_size] = {0};

				for(int k = 0; k < batch_size; k++){
					for (int i = 0; i < 28; ++i) {
						for (int j = 0; j < 28; ++j) {
							input[k][i][j] = (train_set[p * batch_size + k].data)[i][j];
						}
					}
					Y_host[k] = train_set[p * batch_size + k].label;
				}
				time_taken += forward_pass(input, tid);

				l_f[tid]->bp_clear();
				l_c2[tid]->bp_clear();
				l_c1[tid]->bp_clear();
				l_c3[tid]->bp_clear();
				

				// cudaMemset(Y, 0, sizeof(unsigned int) * batch_size);
				cudaMemcpy(Y, Y_host, sizeof(unsigned int) * batch_size, cudaMemcpyHostToDevice);
				makeError<<<batch_size, 10>>>(l_f[tid]->d_preact, l_f[tid]->output, Y, 10 * batch_size);
		
				cublasSnrm2(blas[tid], 10 * batch_size, l_f[tid]->d_preact, 1, &tmp_err);
				err += tmp_err;

				time_taken += back_pass(tid);
				// fprintf(stdout, "device %d, finish iter %d \n", tid, p);
			}
		}

		if(deviceCount > 0){
			#pragma omp parallel num_threads(deviceCount) 
			{
				int tid = omp_get_thread_num();
				cudaSetDevice(tid);

				if(tid != 0){
					cudaMemcpyPeer(&l_c1_weight[(tid-1) * l_c1[tid]->M * l_c1[tid]->N], 0, l_c1[tid]->weight, tid, sizeof(float) * l_c1[tid]->M * l_c1[tid]->N);
					cudaMemcpyPeer(&l_c2_weight[(tid-1) * l_c2[tid]->M * l_c2[tid]->N], 0, l_c2[tid]->weight, tid, sizeof(float) * l_c2[tid]->M * l_c2[tid]->N);
					// cudaMemcpyPeer(&l_c3_weight[(tid-1)* l_c3[tid]->M * l_c3[tid]->N], 0, l_c3[tid]->weight, tid, sizeof(float) * l_c3[tid]->M * l_c3[tid]->N);
					cudaMemcpyPeer(&l_f_weight[(tid-1) * l_f[tid]->M * l_f[tid]->N], 0, l_f[tid]->weight, tid, sizeof(float) * l_f[tid]->M * l_f[tid]->N);
					// // cudaMemcpyPeer(&l_r_weight[(tid-1) * l_r[tid]->M * l_r[tid]->N], 0, l_r[tid]->weight, tid, sizeof(float) * l_r[tid]->M * l_r[tid]->N);

					cudaMemcpyPeer(&l_c1_bias[(tid-1) * l_c1[tid]->N], 0, l_c1[tid]->bias, tid, sizeof(float) * l_c1[tid]->N);
					cudaMemcpyPeer(&l_c2_bias[(tid-1) * l_c2[tid]->N], 0, l_c2[tid]->bias, tid, sizeof(float) * l_c2[tid]->N);
					// cudaMemcpyPeer(&l_c3_bias[(tid-1) * l_c3[tid]->N], 0, l_c3[tid]->bias, tid, sizeof(float) * l_c3[tid]->N);
					cudaMemcpyPeer(&l_f_bias[(tid-1) * l_f[tid]->N], 0, l_f[tid]->bias, tid, sizeof(float) * l_f[tid]->N);
					// cudaMemcpyPeer(&l_r_bias[(tid-1) * l_r[tid]->N], 0, l_r[tid]->bias, tid, sizeof(float) * l_r[tid]->N);
				}

				#pragma omp barrier
				if(tid == 0){
					weight_update<<<2048, 1024>>>(l_c1[tid]->weight, &l_c1_weight[tid * l_c1[tid]->M * l_c1[tid]->N], l_c1[tid]->M * l_c1[tid]->N, deviceCount-1);
					weight_update<<<2048, 1024>>>(l_c2[tid]->weight, &l_c2_weight[tid * l_c2[tid]->M * l_c2[tid]->N], l_c2[tid]->M * l_c2[tid]->N, deviceCount-1);
					// weight_update<<<2048, 1024>>>(l_c3[tid]->weight, &l_c3_weight[tid * l_c3[tid]->M * l_c3[tid]->N], l_c3[tid]->M * l_c3[tid]->N, deviceCount-1);
					weight_update<<<2048, 1024>>>(l_f[tid]->weight, &l_f_weight[tid * l_f[tid]->M * l_f[tid]->N], l_f[tid]->M * l_f[tid]->N, deviceCount-1);
					// // weight_update<<<2048, 1024>>>(l_r[tid]->weight, &l_r_weight[tid * l_r[tid]->M * l_r[tid]->N], l_r[tid]->M * l_r[tid]->N, deviceCount-1);

					weight_update<<<2048, 1024>>>(l_c1[tid]->bias, &l_c1_bias[tid * l_c1[tid]->N], l_c1[tid]->N, deviceCount-1);
					weight_update<<<2048, 1024>>>(l_c2[tid]->bias, &l_c2_bias[tid * l_c2[tid]->N], l_c2[tid]->N, deviceCount-1);
					// weight_update<<<2048, 1024>>>(l_c3[tid]->bias, &l_c3_bias[tid * l_c3[tid]->N], l_c3[tid]->N, deviceCount-1);
					weight_update<<<2048, 1024>>>(l_f[tid]->bias, &l_f_bias[tid * l_f[tid]->N], l_f[tid]->N, deviceCount-1);
					// // weight_update<<<2048, 1024>>>(l_r[tid]->bias, &l_r_bias[tid * l_r[tid]->N], l_r[tid]->N, deviceCount-1);

					weight_average<<<2048, 1024>>>(l_c1[tid]->weight, l_c1[tid]->M * l_c1[tid]->N, deviceCount);
					weight_average<<<2048, 1024>>>(l_c2[tid]->weight, l_c2[tid]->M * l_c2[tid]->N, deviceCount);
					// weight_average<<<2048, 1024>>>(l_c3[tid]->weight, l_c3[tid]->M * l_c3[tid]->N, deviceCount);
					weight_average<<<2048, 1024>>>(l_f[tid]->weight, l_f[tid]->M * l_f[tid]->N, deviceCount);
					// // weight_average<<<2048, 1024>>>(l_r[tid]->weight, l_r[tid]->M * l_r[tid]->N, deviceCount);

					weight_average<<<2048, 1024>>>(l_c1[tid]->bias, l_c1[tid]->N, deviceCount);
					weight_average<<<2048, 1024>>>(l_c2[tid]->bias, l_c2[tid]->N, deviceCount);
					// weight_average<<<2048, 1024>>>(l_c3[tid]->bias, l_c3[tid]->N, deviceCount);
					weight_average<<<2048, 1024>>>(l_f[tid]->bias, l_f[tid]->N, deviceCount);
					// weight_average<<<2048, 1024>>>(l_r[tid]->bias, l_r[tid]->N, deviceCount);
					
					for(int j = 1; j < deviceCount; j++){
						cudaMemcpyPeer(l_c1[j]->weight, j, l_c1[tid]->weight, tid, sizeof(float) * l_c1[tid]->M * l_c1[tid]->N);
						cudaMemcpyPeer(l_c2[j]->weight, j, l_c2[tid]->weight, tid, sizeof(float) * l_c2[tid]->M * l_c2[tid]->N);
						// cudaMemcpyPeer(l_c3[j]->weight, j, l_c3[tid]->weight, tid, sizeof(float) * l_c3[tid]->M * l_c3[tid]->N);
						cudaMemcpyPeer(l_f[j]->weight, j, l_f[tid]->weight, tid, sizeof(float) * l_f[tid]->M * l_f[tid]->N);
						// // cudaMemcpyPeer(l_r[j]->weight, j, l_r[tid]->weight, tid, sizeof(float) * l_r[tid]->M * l_r[tid]->N);

						cudaMemcpyPeer(l_c1[j]->bias, j, l_c1[tid]->bias, tid, sizeof(float) * l_c1[tid]->N);
						cudaMemcpyPeer(l_c2[j]->bias, j, l_c2[tid]->bias, tid, sizeof(float) * l_c2[tid]->N);
						// cudaMemcpyPeer(l_c3[j]->bias, j, l_c3[tid]->bias, tid, sizeof(float) * l_c3[tid]->N);
						cudaMemcpyPeer(l_f[j]->bias, j, l_f[tid]->bias, tid, sizeof(float) * l_f[tid]->N);
						// cudaMemcpyPeer(l_r[j]->bias, j, l_r[tid]->bias, tid, sizeof(float) * l_r[tid]->N);
					}
				}
			}
		}
		

		fprintf(stdout, "\n finish iter %d \n", iter);
		err /= train_cnt;
		double accuracy = 100 - double(err) * 100.0;
		fprintf(stdout, "accuracy: %.2lf%% , time_on_gpu: %lf sec\n", accuracy, time_taken);


		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
	}



	auto end_time = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end_time - start_time;
	double seconds = diff.count();

	fprintf(stdout, "\n Time - %lf s\n", seconds);
}


// Returns label of given data (0-9)
static int* classify(float input[batch_size][28][28], int tid)
{
	float res[batch_size * 10];

	forward_pass(input, tid);

	int* max = new int[batch_size]{0};

	cudaMemcpy(&res[0], l_f[tid]->output, sizeof(float) * 10 * batch_size, cudaMemcpyDeviceToHost);

	
	for(int j = 0; j < batch_size; j++){
		for (int i = 0; i < 10; i++) {
			if (res[10 * j + max[j]] < res[10 * j + i]) {
				max[j] = i;
			}
		}
	}
	
	return max;
}

// Perform forward propagation of test data
static void test(int tid)
{
	cudaSetDevice(tid);
	int error = 0;
	
	int batch_cnt = test_cnt / batch_size;
	for(int p = 0; p < batch_cnt; ++p){
		float input[batch_size][28][28];
		for(int k = 0; k < batch_size; ++k){
			for (int i = 0; i < 28; ++i) {
				for (int j = 0; j < 28; ++j) {
					input[k][i][j] = (test_set[batch_size * p + k].data)[i][j];
				}
			}
		}

		int* max = classify(input, tid);
		for (int i = 0; i < batch_size; ++i) {
			if (max[i] != test_set[batch_size * p + i].label) {
				++error;
			}
		}
	}
	
	double err_percent = double(error) / double(test_cnt) * 100.0;
	fprintf(stdout, "Error Rate: %.2lf%% , accuracy: %.2lf%%\n",err_percent,100-err_percent);
}
