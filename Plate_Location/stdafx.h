// stdafx.h : ��׼ϵͳ�����ļ��İ����ļ���
// ���Ǿ���ʹ�õ��������ĵ�
// �ض�����Ŀ�İ����ļ�
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
#define MIN_PERCENT 0.19//�����ַ������ȻҶ�ռ��,������β��ԣ�����percentage < 0.19���Ըպ��˳������е����������ǰ���ͺ������м��
#define MIN_CHAR_WIDTH 1//�����ַ�����С���ؿ�ȣ�Char_Segment�����У�
#define MAX_CUT 15//���ķָ�����ProjectionCut�����У�
#define GRAY_THRESHOLD 20//�Ҷ���ֵ��PxPercentage���������ã�
#define MAX_CONTINUOUS_PX_NUM 11//11�߿����ʱ��������������Ϊ255�����������Ŀ��DetectionChange������
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

// TODO: �ڴ˴����ó�����Ҫ������ͷ�ļ�
