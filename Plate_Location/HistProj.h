#pragma once
#include "stdafx.h"
class HistProj
{
public:
	HistProj();
	~HistProj();
	int Char_Segment(IplImage* Srcimg, std::string filename);
	int Char_Segment1(IplImage* Srcimg, std::string filename);
private:
	IplImage *m_Grayimg = NULL;//保存灰度图
	IplImage *m_Binimg = NULL;//保存二值图
	void ProjectionCalculate(Mat& mat1, int* vArr);
	int** ProjectionCut(int* vArr, int width, int& number);
	//int** verProjection_cut(int* vArr, int width, int& number);
	float PxPercentage(Mat& mat1);
	int Thresholding(Mat TempDst, IplImage*&dst);
};




