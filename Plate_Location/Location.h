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
#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
class CLocation
{
public:
	//��������Դ���ļ���
	CLocation(Mat src,string name);
	~CLocation();
	//��Ԥ���� ��˹ģ�����ҶȻ�
	void PreTreatment(Mat src);
	//��Sobel�������
	void SobelDetection();
	//��Canny�������
	void CannyDetection();
	//����ȡCanny������ֵ
	void AdaptiveFindThreshold(const CvArr* image, double *low, double *high, int aperture_size);
	void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high);
	
	//����̬�ж�
	bool verifySizes(RotatedRect mr);
	//! ���������ʾ
	Mat showResultMat(Mat src, Size rect_size, Point2f center, int index);

	//! PlateLocate���ó���
	static const int DEFAULT_GAUSSIANBLUR_SIZE = 5;
	static const int SOBEL_SCALE = 1;
	static const int SOBEL_DELTA = 0;
	static const int SOBEL_DDEPTH = CV_16S;
	static const int SOBEL_X_WEIGHT = 1;
	static const int SOBEL_Y_WEIGHT = 0;
	static const int DEFAULT_MORPH_SIZE_WIDTH = 17;
	static const int DEFAULT_MORPH_SIZE_HEIGHT = 3;
	//�� ��ɫ������ȡ
	bool ColorFeatureExtraction(); 
	//�� ��ֵ��
	Mat Binarization(Mat src, double mean, double aimWeight);
	//! ��̬ѧ����
	Mat Morphological(Mat src);
	//������ɸѡ
	bool ContourSearch(Mat src);
	//����ֱͶӰ
	double VerticalProjection(Mat src);
	//! ��������
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

	//����ɫ����ֵ����
	double **ColorFeture; 
	//! ���Ӳ������ñ���
	int m_MorphSizeWidth;
	int m_MorphSizeHeight;

	//! verifySize���ñ���
	float m_error;
	float m_aspect;
	int m_verifyMin;
	int m_verifyMax;

	//! showResultMat���ó���
	static const int WIDTH = 136;
	static const int HEIGHT = 36;
	static const int TYPE = CV_8UC3;

	//! �Ƕ��ж����ó���
	static const int DEFAULT_ANGLE = 30;

	//! �Ƕ��ж����ñ���
	int m_angle;

	//! verifySize���ó���
	static const int DEFAULT_VERIFY_MIN = 1;
	static const int DEFAULT_VERIFY_MAX = 100;

	//! ��ȡ��ɫ����ֵ���ó���
	
	
	//! ��ȡ��ɫ����ֵ���ñ���
	int m_featureNum;
	double m_featureSum;
	double m_featureMean;
	double m_aimWeight;

	//! Ŀ��㳣��
	//double m_aimWeight;

	//! ���ƹ���ȳ���
	static const int DEFAULT_CHAR_NUM = 7;

	//! ����ܳ�
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
	//�����
	int m_cols;
	int m_rows;
	//���ַ�ͳ������
	int *m_Projection;
	int m_CharNum;
	//�����ƹ����
	double m_RuleDegree;
	//���ļ����
	string m_ImgName;
	char m_OutAddress[30];
	string m_OutAddressFirst;
	//������ģʽ��true��ʼ���ԣ�false�رյ���
	bool m_deBug;
	//! ƫ����
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

