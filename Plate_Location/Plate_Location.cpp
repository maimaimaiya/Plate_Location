// Plate_Location.cpp : �������̨Ӧ�ó������ڵ㡣
//
//////////////////////////////////////////////////////////////////////////
// Name:	    Plate_Locate Header
// Version:		1.0
// Date:			2017-3-23
// MDate:		2017-4-05
// Author:	    ˧����
// Desciption:  
// Defines CPlateLocate
// �޸�ʱ�䣺2017-4-14
// �޸����ݣ�������ɫ�ռ�Ķ�λ
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
		//temp = "��A89106.jpg";
		if (temp.size() <= 4)
			continue;
		Sum++;
		strcpy(tempname, temp.c_str());
		printf("����⳵�ƣ�%s\n", tempname);
		m_name = tempFirst + temp;
		Mat srcImage = imread(m_name);
		Mat dstImg;
		//���Եĳߴ��һ
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
			printf("�ɹ�\n");
		}
		else if (test.Color_Contour())
		{
			Success_Num++;
			printf("�ɹ�\n");
		}
		else
		{
			outs << temp << endl;
			imwrite("./src/ʧ��/" + temp, srcImage);
			printf("ʧ��\n");
		}
	}
	overclock = clock();
	printf("ʶ��������%d�� �ɹ�ʶ��%d�� ʶ���ʣ�%.2lf%% ;ʱ�䣺%d\n", Sum, Success_Num, Success_Num*(1.0) / Sum * 100,int(overclock - nowclock));
	ifs.close();
	outs.close();
    return 0;
}

