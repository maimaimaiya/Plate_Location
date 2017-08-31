// Plate_Location.cpp : 定义控制台应用程序的入口点。
//
//////////////////////////////////////////////////////////////////////////
// Name:	    Plate_Locate Header
// Version:		1.0
// Date:			2017-3-23
// MDate:		2017-4-05
// Author:	    帅鹏举
// Desciption:  
// Defines CPlateLocate
// 修改时间：2017-4-14
// 修改内容：基于颜色空间的定位
//////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Location.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <io.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>

using namespace std;
using namespace cv;
#define NORM_WIDTH 600
#define NORM_HEIGHT 800


int main()
{
	ifstream ifs("./src/Img_data.txt");
	ofstream outs("./src/error.txt");
	string temp;
	string tempFirst = "./src/test_Img/";
	string outfile = "./src/Img_out/";
	string m_name;
	char tempname[30];
	int Sum = 0;
	int Success_Num = 0;
	time_t nowclock, overclock;
	nowclock = clock();
	while (!ifs.eof())
	{
		getline(ifs, temp);
		//temp = "京A89106.jpg";
		if (temp.size() <= 4)
			continue;
		Sum++;
		strcpy(tempname, temp.c_str());
		printf("待检测车牌：%s\n", tempname);
		m_name = tempFirst + temp;
		Mat srcImage = imread(m_name);
		Mat dstImg;
		//粗略的尺寸归一
		dstImg.create(NORM_WIDTH, NORM_HEIGHT, 16);
		resize(srcImage, dstImg, dstImg.size(), 0, 0, INTER_CUBIC);
		CLocation test(srcImage,temp);
		//test.Color_Contour();
		test.PreTreatment(srcImage);
		//test.CannyDetection();
		test.SobelDetection();
		//if(test.Color_Contour())
		if (test.ColorFeatureExtraction())
		{
			Success_Num++;
			printf("成功\n");
		}
		else if (test.Color_Contour())
		{
			Success_Num++;
			printf("成功\n");
		}
		else
		{
			outs << temp << endl;
			imwrite("./src/失败/" + temp, srcImage);
			printf("失败\n");
		}
	}
	overclock = clock();
	printf("识别数量：%d张 成功识别：%d张 识别率：%.2lf%% ;时间：%d\n", Sum, Success_Num, Success_Num*(1.0) / Sum * 100,int(overclock - nowclock));
	ifs.close();
	outs.close();
    return 0;
}

