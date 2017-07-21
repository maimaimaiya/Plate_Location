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


int main()
{
	ifstream ifs("./src/data.txt");
	string temp;
	string tempFirst = "./src/general_test/";
	string m_name;
	char tempname[30];
	int Sum = 0;
	int Success_Num = 0;
	time_t nowclock, overclock;
	nowclock = clock();
	while (!ifs.eof())
	{
		getline(ifs, temp);
		//temp = "湘A0PQ76.jpg";
		if (temp.size() <= 4)
			continue;
		Sum++;
		strcpy(tempname, temp.c_str());
		printf("待检测车牌：%s\n", tempname);
		m_name = tempFirst + temp;
		Mat srcImage = imread(m_name);
		CLocation test(srcImage,temp);
		//test.Color_Contour();
		test.PreTreatment(srcImage);
		//test.CannyDetection();
		test.SobelDetection();
		if (test.ColorFeatureExtraction())
		{
			Success_Num++;
		}
	}
	overclock = clock();
	printf("识别数量：%d张 成功识别：%d张 识别率：%.2lf%% ;时间：%d\n", Sum, Success_Num, Success_Num*(1.0) / Sum * 100,int(overclock - nowclock));
    return 0;
}

