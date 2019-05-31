#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fftw3.h"
#include <windows.h>
#include <Eigen/Dense>
#include <iostream>  
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace Eigen;

#define COLS 3
#define ROWS 3

#pragma comment(lib, "libfftw3-3.lib") // double版本
//#pragma comment(lib, "libfftw3f-3.lib")// float版本
// #pragma comment(lib, "libfftw3l-3.lib")// long double版本

extern "C"    void iteration_mat1();

/**********************************主函数****************************************/
int main()
{

    fftw_complex*result_temp_din, *result_temp_out;
    fftw_plan p;

    result_temp_din = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*COLS*ROWS);
    result_temp_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*COLS*ROWS);
    cout << "fftw" << endl;

    for (size_t j = 0; j < ROWS; j++)
    {
        for (size_t i = 0; i < COLS; i++)
        {

            result_temp_din[i + j*COLS][0] = (i+1)*(j+1);
            cout << result_temp_din[i + j*COLS][0] << " ";
            result_temp_din[i + j*COLS][1] = 0;
        }
    }

    //forward fft  
    p = fftw_plan_dft_2d(ROWS, COLS, result_temp_din, result_temp_out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    fftw_execute(p);
    cout << endl;
    for (size_t j = 0; j < ROWS; j++)
    {
        for (size_t i = 0; i < COLS; i++)
        {
            cout << result_temp_out[i + j*COLS][0] << " ";//实部  
            cout << result_temp_out[i + j*COLS][1] << endl;//虚部  
        }
    }

    cout << "cuda" << endl;
    iteration_mat1();
    system("pause");
    return 0;
}