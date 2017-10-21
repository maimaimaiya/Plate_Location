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
// 修改时间：2017-10-21
// 修改内容：加入字符分割代码
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
#include "HistProj.h"
#include "Region_Seg.h"
using namespace std;
using namespace cv;
#define NORM_WIDTH 600
#define NORM_HEIGHT 800
int filenum = 0, ans = 0;
int success = 0, success1 = 0, success2 = 0;

bool Char_seg(string filename,IplImage *Srcimg)
{
	Region_Seg seg;
	IplImage* Im = seg.Char_PreSeg1(Srcimg);
	HistProj hist;
	int num = 0, num1 = 0, num2 = 0;
	num = seg.Char_Seg(filename, Im);
	if (num == 7)
	{
		success++; //cout << success << endl;
	}
	else
	{
		//cout << filename << endl;
		cout << "分割方式1失败" << endl;
		//num1=test1.Char_Segment(Im, filename);
		IplImage* Im1 = seg.Char_PreSeg(Srcimg);
		num2 = hist.Char_Segment1(Im1, filename);
		if (num2 != 7)
		{
			cout << "分割方式2失败" << endl;
			num1 = hist.Char_Segment(Im, filename);
		}
		else
		{
			success1++;
			cout << success << endl;
			num1 = hist.Char_Segment(Im1, filename);
			if (num1 != 7)cout << "分割方式3失败" << endl;
			else {
				success2++;
			}
		}

		//Im = test.Char_PreSeg(Srcimg);
		//num1= test1.Char_Segment(Im, filename);

	}
	if (num < 3)
	{
		//IplImage* Im1 = test.Char_PreSeg1(Srcimg);
		//int num2=test.Char_Seg(filename, Im1);
		ans++;
	}//if (num2 == 7) success++;
	cvWaitKey(0);
	cvReleaseImage(&Srcimg);
	cvReleaseImage(&Im);
	
	return true;
}


int main()
{
	ifstream ifs("./src/data.txt");
	ofstream outs("./src/error.txt");
	string temp;
	string tempFirst = "./src/general_test/";
	string outfile = "./src/out/";
	string m_name;
	char tempname[30];
	int Sum = 0;
	int Success_Num = 0;
	time_t nowclock, overclock;
	nowclock = clock();
	while (!ifs.eof())
	{
		getline(ifs, temp);
		//temp = "沪A51V39.jpg";
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
			printf("定位成功\n");			
			IplImage imgTmp = test.GetResultImage();
			IplImage *input = cvCloneImage(&imgTmp);
			Char_seg(tempname,input);

		}
		else if (test.Color_Contour())
		{
			Success_Num++;
			printf("定位成功\n");
			IplImage imgTmp = test.GetResultImage();
			IplImage *input = cvCloneImage(&imgTmp);
			Char_seg(tempname,input);
		}
		else
		{
			outs << temp << endl;
			imwrite("./src/失败/" + temp, srcImage);
			printf("失败\n");
		}
	}
	overclock = clock();
	printf("定位数量：%d张 成功识别：%d张 识别率：%.2lf%% ;时间：%d\n", Sum, Success_Num, Success_Num*(1.0) / Sum * 100,int(overclock - nowclock));
	printf("未分割%d张\n", ans);
	printf("分割数量：%d张 成功识别：%d张 %d张 %d张 识别率：%.2lf%% \n", Success_Num, success, success1, success2, (float)100 * (success + success1 + success2)*1.0 / Sum );
	ifs.close();
	outs.close();
    return 0;
}

