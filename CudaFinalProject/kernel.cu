#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "opencv2/opencv.hpp"

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <cuda_runtime_api.h>

const char *photoFilename = "data/photo.jpg";
const char *frameFilename = "data/frame.jpg";
const char *resultFilename = "data/result.jpg";

__global__ void ZoomKernel(const double,const double, const uchar3 *origin, uchar3 *result);

int main(){
	cv::Mat photo,smallphoto, frame, framegray, result;
	photo = cv::imread(photoFilename);
	frame = cv::imread(frameFilename);
	framegray = cv::Mat(frame.size(), CV_8UC1, cv::Scalar::all(0));
	smallphoto = cv::Mat(cv::Size(350, 350), photo.type(), cv::Scalar::all(0));

	size_t d_smallphoto_size = smallphoto.cols*smallphoto.rows * sizeof(uchar3);
	uchar3 *d_smallphoto = NULL;
	cudaMalloc(&d_smallphoto, d_smallphoto_size);

	size_t d_photo_size = photo.cols*photo.rows*sizeof(uchar3);
	uchar3 *d_photo = NULL;
	cudaMalloc(&d_photo, d_photo_size);
	cudaMemcpy(d_photo, photo.data, d_photo_size, cudaMemcpyHostToDevice);
	dim3 dimBlock(32, 32);
	dim3 dimGrid(photo.rows / dimBlock.x, photo.cols / dimBlock.y);



	cvNamedWindow("233");
	cv::imshow("233", photo);
	cv::waitKey();
	return 0;
}

__global__ void ZoomKernel(double scale_x, double scale_y, uchar3 *origin, uchar3 *result){

}




