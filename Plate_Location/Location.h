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
#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
class CLocation
{
public:
	//！传入资源与文件名
	CLocation(Mat src,string name);
	~CLocation();
	//！预处理 高斯模糊及灰度化
	void PreTreatment(Mat src);
	//！Sobel轮廓检测
	void SobelDetection();
	//！Canny轮廓检测
	void CannyDetection();
	//！求取Canny上下阈值
	void AdaptiveFindThreshold(const CvArr* image, double *low, double *high, int aperture_size);
	void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high);
	
	//！形态判断
	bool verifySizes(RotatedRect mr);
	//! 结果车牌显示
	Mat showResultMat(Mat src, Size rect_size, Point2f center, int index);

	//! PlateLocate所用常量
	static const int DEFAULT_GAUSSIANBLUR_SIZE = 5;
	static const int SOBEL_SCALE = 1;
	static const int SOBEL_DELTA = 0;
	static const int SOBEL_DDEPTH = CV_16S;
	static const int SOBEL_X_WEIGHT = 1;
	static const int SOBEL_Y_WEIGHT = 0;
	static const int DEFAULT_MORPH_SIZE_WIDTH = 17;
	static const int DEFAULT_MORPH_SIZE_HEIGHT = 3;
	//！ 颜色特征提取
	bool ColorFeatureExtraction(); 
	//！ 二值化
	Mat Binarization(Mat src, double mean, double aimWeight);
	//! 形态学操作
	Mat Morphological(Mat src);
	//！区域筛选
	bool ContourSearch(Mat src);
	//！垂直投影
	double VerticalProjection(Mat src);
	//! 轮廓跟踪
	void ContourTracking(Mat src);
	int ContourMarking(int x_start, int y_start,Mat src);
	bool Color_Contour();
	bool Blue_Judge(int x, int y, Mat &temp);
	bool White_Judge(int x, int y, Mat &temp);
	void DetectionChange(Mat & mat1, Mat & dst);
	void ProjectionCalculate(Mat & mat1, int * vArr);
	int ** ProjectionCut(int * vArr, int width, int & numofcut);
	Mat GetResultImage();
	void SetResultImage(Mat img);
protected:

	//！颜色特征值数组
	double **ColorFeture; 
	//! 连接操作所用变量
	int m_MorphSizeWidth;
	int m_MorphSizeHeight;

	//! verifySize所用变量
	float m_error;
	float m_aspect;
	int m_verifyMin;
	int m_verifyMax;

	//! showResultMat所用常量
	static const int WIDTH = 136;
	static const int HEIGHT = 36;
	static const int TYPE = CV_8UC3;

	//! 角度判断所用常量
	static const int DEFAULT_ANGLE = 30;

	//! 角度判断所用变量
	int m_angle;

	//! verifySize所用常量
	static const int DEFAULT_VERIFY_MIN = 1;
	static const int DEFAULT_VERIFY_MAX = 100;

	//! 提取颜色特征值所用常量
	
	
	//! 提取颜色特征值所用变量
	int m_featureNum;
	double m_featureSum;
	double m_featureMean;
	double m_aimWeight;

	//! 目标点常量
	//double m_aimWeight;

	//! 车牌规则度常量
	static const int DEFAULT_CHAR_NUM = 7;

	//! 最大周长
	static const int MAX_PERIMETER = 2000;
	static const int MAX_CNTR = 10000;
	static const int MAX_BRIGHTNESS = 255;
	static const int GRAY = 128;
	static const int MAX_COLS = 1500;
	static const int MAX_ROWS = 1500;

private:
	Mat m_srcImg;
	Mat m_dstImg;
	Mat ResultImage;
	//！宽高
	int m_cols;
	int m_rows;
	//！字符统计数组
	int *m_Projection;
	int m_CharNum;
	//！车牌规则度
	double m_RuleDegree;
	//！文件输出
	string m_ImgName;
	char m_OutAddress[30];
	string m_OutAddressFirst;
	//！调试模式，true开始调试，false关闭调试
	bool m_deBug;
	//! 偏移量
	int Freeman[8][2] = {  
		{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },
		{ -1, 0 },{ -1,  1 },{ 0,  1 },{ 1,  1 } };
	int chain_code[MAX_CNTR];
	int **blue;
	int **white;
	//int blue[3000][3000] = { 0 };
	//int white[3000][3000] = { 0 };
	Mat srcBlur;

	int Color_Mark[MAX_COLS][MAX_ROWS];
	int Color_HSV[MAX_COLS][MAX_ROWS][3];
};

