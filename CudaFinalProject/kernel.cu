// Includes CUDA
#include <cuda_runtime.h>
// Utilities and timing functions
#include <cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;


const char *photoFilename = "data/photo.jpg";
const char *frameFilename = "data/frame.jpg";
const char *resultFilename = "data/result.jpg";

__global__ void ZoomKernel(double scale_x, double scale_y, uchar3 *origin, uchar3 *result, int width, int height, int size){
	unsigned int col = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int row = threadIdx.y + blockDim.y*blockIdx.y;

	int sy = (int)row * scale_y;
	sy = (sy < (height - 1)) ? sy : (height - 1);

	int sx = (int)col * scale_x;
	sx = (sx < (width - 1)) ? sx : (width - 1);

	result[row*size + col].x = origin[sy*width + sx].x;
	result[row*size + col].y = origin[sy*width + sx].y;
	result[row*size + col].z = origin[sy*width + sx].z;
}

void Zoom(Mat photo, Mat smallphoto){
	size_t d_smallphoto_size = smallphoto.cols*smallphoto.rows * sizeof(uchar3);
	uchar3 *d_smallphoto = NULL;
	cudaMalloc(&d_smallphoto, d_smallphoto_size);

	size_t d_photo_size = photo.cols*photo.rows*sizeof(uchar3);
	uchar3 *d_photo = NULL;
	cudaMalloc(&d_photo, d_photo_size);
	cudaMemcpy(d_photo, photo.data, d_photo_size, cudaMemcpyHostToDevice);

	double scale_x = (double)photo.cols / smallphoto.cols;
	double scale_y = (double)photo.rows / smallphoto.rows;

	dim3 dimBlock(32, 32);
	dim3 dimGrid((smallphoto.rows + dimBlock.x - 1) / dimBlock.x, (smallphoto.cols + dimBlock.y - 1) / dimBlock.y);

	ZoomKernel << <dimBlock, dimGrid >> >(scale_x, scale_y, d_photo, d_smallphoto, photo.cols, photo.rows, smallphoto.cols);

	cudaMemcpy(smallphoto.data, d_smallphoto, d_smallphoto_size, cudaMemcpyDeviceToHost);

	cudaFree(d_smallphoto);
	cudaFree(d_photo);
}

__global__ void GrayKernel(uchar1 *result, uchar3 *origin, int width){
	unsigned int col = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int row = threadIdx.y + blockDim.y*blockIdx.y;

	result[row*width + col].x = origin[row*width + col].x*0.11 + origin[row*width + col].y*0.59 + origin[row*width + col].z*0.3;
}

void ConvertGray(Mat framegray, Mat frame){

	size_t d_framegray_size = framegray.cols*framegray.rows * sizeof(uchar1);
	uchar1 *d_framegray = NULL;
	cudaMalloc(&d_framegray, d_framegray_size);

	size_t d_frame_size = frame.cols*frame.rows*sizeof(uchar3);
	uchar3 *d_frame = NULL;
	cudaMalloc(&d_frame, d_frame_size);
	cudaMemcpy(d_frame, frame.data, d_frame_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(32, 32);
	dim3 dimGrid((frame.cols + dimBlock.x - 1) / dimBlock.x, (frame.rows + dimBlock.y - 1) / dimBlock.y);

	GrayKernel << <dimBlock, dimGrid >> >(d_framegray, d_frame, frame.cols);

	cudaMemcpy(framegray.data, d_framegray, d_framegray_size, cudaMemcpyDeviceToHost);

	cudaError_t err = cudaSuccess;
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
	}
	cudaFree(d_frame);
	cudaFree(d_framegray);
}

__global__ void CoverKernel(uchar3 *result, uchar3 *smallphoto, uchar1 *gray, int photowidth, int framewidth, int startX, int startY){
	unsigned int col = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int row = threadIdx.y + blockDim.y*blockIdx.y;

	if (col < 350 && row < 350){
		if (gray[startY*framewidth + startX + row*framewidth + col].x < 20){
			result[startY*framewidth + startX + row*framewidth + col].x = smallphoto[row*photowidth + col].x;
			result[startY*framewidth + startX + row*framewidth + col].y = smallphoto[row*photowidth + col].y;
			result[startY*framewidth + startX + row*framewidth + col].z = smallphoto[row*photowidth + col].z;
		}
	}
}

void cover(Mat frame, Mat smallphoto, Mat framegray,int startX,int startY,int width){
	size_t d_frame_size = frame.cols*frame.rows*sizeof(uchar3);
	uchar3 *d_frame = NULL;
	cudaMalloc(&d_frame, d_frame_size);
	cudaMemcpy(d_frame, frame.data, d_frame_size, cudaMemcpyHostToDevice);

	size_t d_framegray_size = framegray.cols*framegray.rows*sizeof(uchar1);
	uchar1 *d_framegray = NULL;
	cudaMalloc(&d_framegray, d_framegray_size);
	cudaMemcpy(d_framegray, framegray.data, d_framegray_size, cudaMemcpyHostToDevice);

	size_t d_smallphoto_size = smallphoto.cols*smallphoto.rows*sizeof(uchar3);
	uchar3 *d_smallphoto = NULL;
	cudaMalloc(&d_smallphoto, d_smallphoto_size);
	cudaMemcpy(d_smallphoto, smallphoto.data, d_smallphoto_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(32, 32);
	dim3 dimGrid((smallphoto.rows + dimBlock.x - 1) / dimBlock.x, (smallphoto.cols + dimBlock.y - 1) / dimBlock.y);

	CoverKernel << <dimBlock, dimGrid >> >(d_frame, d_smallphoto, d_framegray, smallphoto.cols, frame.cols, startX, startY);
	
	cudaMemcpy(frame.data, d_frame, d_frame_size, cudaMemcpyDeviceToHost);

	cudaFree(d_frame);
	cudaFree(d_framegray);
	cudaFree(d_smallphoto);
}

int main(){
	cv::Mat photo,smallphoto, frame, framegray, result;
	photo = cv::imread(photoFilename);
	frame = cv::imread(frameFilename,1);
	framegray = cv::Mat(frame.size(), CV_8UC1, cv::Scalar::all(0));
	smallphoto = cv::Mat(cv::Size(350, 350), photo.type(), cv::Scalar::all(0));

	Zoom(photo, smallphoto);
	ConvertGray(framegray, frame);
	cover(frame, smallphoto, framegray, 180, 125, 350);

	cvNamedWindow("gray");
	imshow("gray", framegray);

	cvNamedWindow("frame");
	imshow("frame", frame);

	cvNamedWindow("small");
	cv::imshow("small", smallphoto);

	cv::waitKey();
	return 0;
}