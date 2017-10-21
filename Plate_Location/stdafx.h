// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <direct.h> 
#include "string"
#include <atlstr.h>//CString
#include <opencv/cv.h>
#include <iostream>
#define MIN_PERCENT 0.19//车牌字符的最先灰度占比,经过多次测试，发现percentage < 0.19可以刚好滤除车牌中的噪声点包括前区和后区的中间点
#define MIN_CHAR_WIDTH 1//车牌字符的最小像素宽度（Char_Segment函数中）
#define MAX_CUT 15//最大的分割数（ProjectionCut函数中）
#define GRAY_THRESHOLD 20//灰度阈值（PxPercentage函数中运用）
#define MAX_CONTINUOUS_PX_NUM 11//11边框分离时检测横向连续像素为255的像素最大数目（DetectionChange函数）
/*#define reHeight 36
#define reWidth 136
#define reHeight1 40
#define reWidth1 150//140*/
#define reHeight 36
#define reWidth 136
#define reHeight1 45
#define reWidth1 150
using namespace std;
using namespace cv;

// TODO: 在此处引用程序需要的其他头文件
