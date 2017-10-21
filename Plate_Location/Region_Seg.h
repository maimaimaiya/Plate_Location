#pragma once
#include "stdafx.h"
class Region_Seg
{
public:
	Region_Seg();
	IplImage* Char_PreSeg(IplImage *m_img);
	int Char_Seg(std::string filename, IplImage* img);
	void borderCut(cv::Mat& mat1, cv::Mat& dst);
	IplImage* Region_Seg::Char_PreSeg1(IplImage *m_img);
	~Region_Seg();
private:
	void Thresholding(IplImage* &img_g);
	IplImage* synthetic(IplImage *logimg, IplImage *oustimg);
};

