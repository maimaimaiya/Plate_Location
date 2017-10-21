#include "stdafx.h"
#include "HistProj.h"
#include "Region_Seg.h"
#include "io.h"
#include "fstream"
#include "vector"
#include "shlwapi.h"
#include <sstream>
#pragma comment(lib,"shlwapi.lib")
using namespace std;
HistProj::HistProj()
{
}
HistProj::~HistProj()
{
	cvDestroyAllWindows();
	cvReleaseImage(&m_Grayimg);
	cvReleaseImage(&m_Binimg);
}
int HistProj::Char_Segment(IplImage* Srcimg, std::string filename)
{//中值滤波  参考：一种基于边缘信息的车牌字符分割算法
	cv::Mat src = cv::cvarrToMat(Srcimg);
	cv::Mat dst;
	cv::medianBlur(src, dst, 3);
	//blur(src, dst, Size(3, 3), Point(-1, -1));
	IplImage* img_gray1 = &IplImage(dst);
	//cvShowImage("src1", img_gray1);
	//去边框
	cv::Mat img_3 = cv::cvarrToMat(img_gray1);
	     Region_Seg  s;
		s.borderCut(img_3, img_3);//去边框
	cv::Point2f center = cv::Point2f(img_3.cols / 2, img_3.rows / 2);  // 旋转中心
	cv::Mat rotateMat = getRotationMatrix2D(center, 180, 1);//旋转矩阵
	cv::warpAffine(img_3, img_3, rotateMat, img_3.size());//这是我的一个意外发现，目前没找到原因，就是我也不知道为什么在经
	cv::warpAffine(img_3, img_3, rotateMat, img_3.size());//过两次翻转（理论上相当于没有改变什么）后，分割正确率提高了。
														  //cv::imshow("边框分离", img_3);
	Srcimg = &IplImage(img_3);
	//cvShowImage("2",img);
	//cout << Srcimg->width << " " << Srcimg->height << endl;
	IplImage*  dstimg = cvCreateImage(cv::Size(reWidth1, reHeight1), Srcimg->depth, Srcimg->nChannels);
	cvCopyMakeBorder(Srcimg, dstimg, cvPoint((reWidth1 - (Srcimg->width)) / 2, (reHeight1 - (Srcimg->height)) / 2), IPL_BORDER_CONSTANT);
	//归一化处理
	int  charHeight =40, charWidth = 30;//要保存的单个字符图的size
	int *vArr = new int[reWidth1];//创建投影数组
	int **pic_Arr;//创建投影分割结果的起始点与结束点的记录数组
				  //其中第二维只有两个数，第1个数表示开始位置，第2个数表示结束位置。
	Mat img_4= cv::cvarrToMat(dstimg);
	//归一化图片的大小。  便于处理
	resize(img_4, img_4, Size(reWidth1, reHeight1));//归一化为re_Width*re_Height大小
	ProjectionCalculate(img_4, vArr);//计算投影数组
	int pic_ArrNumber;
	pic_Arr = ProjectionCut(vArr, reWidth1, pic_ArrNumber);//进行分割
	Mat img_5;
	IplImage pI_1 = img_4;
	IplImage pI_2;
	CvScalar s1;
	int seg_num = 0;
	char numchar[10];
	for (int i = 0; i < pic_ArrNumber; i++)
	{
		int pic_width = pic_Arr[i][1] - pic_Arr[i][0];// 字符宽度 
		if (pic_width <= MIN_CHAR_WIDTH&&pic_Arr[i][1] <= reWidth1 / 2 && pic_ArrNumber <= 3)
		{
			continue;
		}
		img_5 = cv::Mat(reHeight1, pic_Arr[i][1] - pic_Arr[i][0], CV_8UC1, 1);
		pI_2 = img_5;
		//创建文件夹
		stringstream ss(filename);
		while (getline(ss, filename, '/')) //以/为间隔分割filename的内容,存入filename。这里最后得到的就是不含路径的文件名
			;
		stringstream ss1(filename);
		getline(ss1, filename, '.');//去掉后缀
		CString str = ("rst1//" + filename).c_str();
		if (!PathIsDirectory(str))//创建文件夹，有则创建
		{
			CreateDirectory(str, NULL);
		}
		for (int j = 0; j < reHeight1; j++)
		{
			for (int k = pic_Arr[i][0]; k < pic_Arr[i][1]; k++)
			{
				s1 = cvGet2D(&pI_1, j, k);
				cvSet2D(&pI_2, j, k - pic_Arr[i][0], s1);
			}
		}
		float percentage = PxPercentage(img_5);
		if (percentage < MIN_PERCENT)
		{
			continue;
		}
		/////调整输出图片的大小。
		Mat img_w;
		// 初始化输入参数
		int top = (int)abs(charHeight - img_5.rows) / 2;
		int bottom = top;
		int left = (int)abs(charWidth - img_5.cols) / 2;
		int right = left;
		copyMakeBorder(img_5, img_w, top, bottom, left, right, IPL_BORDER_CONSTANT, 0);//0填充扩展边缘
		resize(img_w, img_w, Size(charWidth, charHeight));
		char name[25] = { 0 };
		seg_num++;
		sprintf_s(name, "%d.jpg", seg_num);//存储图像
		//cout<< "./rst/" + filename + "/" + name  + ".jpg" <<endl;
		cv::imwrite("./rst1/" + filename + "/" +name + ".jpg", img_w);
		//imwrite(name, img_w);
		//cvNamedWindow("1");
		//imshow("1", img_w);
	
	}cv::imwrite("./rst1/" + filename + "/" + filename + ".jpg", img_4);
	delete[] vArr;
	free(pic_Arr);
	return  seg_num;
}
int HistProj::Char_Segment1(IplImage* Srcimg, std::string filename)
{//中值滤波  参考：一种基于边缘信息的车牌字符分割算法
	cv::Mat src = cv::cvarrToMat(Srcimg);
	cv::Mat dst;
	cv::medianBlur(src, dst, 3);
	//blur(src, dst, Size(3, 3), Point(-1, -1));
	IplImage* img_gray1 = &IplImage(dst);
	//cvShowImage("src1", img_gray1);
	//去边框

	cv::Mat img_3 = cv::cvarrToMat(img_gray1);
	Region_Seg  s;
	s.borderCut(img_3, img_3);//去边框
	cv::Point2f center = cv::Point2f(img_3.cols / 2, img_3.rows / 2);  // 旋转中心
	cv::Mat rotateMat = getRotationMatrix2D(center, 180, 1);//旋转矩阵
	cv::warpAffine(img_3, img_3, rotateMat, img_3.size());//这是我的一个意外发现，目前没找到原因，就是我也不知道为什么在经
	cv::warpAffine(img_3, img_3, rotateMat, img_3.size());//过两次翻转（理论上相当于没有改变什么）后，分割正确率提高了。
														  //cv::imshow("边框分离", img_3);
	Srcimg = &IplImage(img_3);
	//cvShowImage("2",img);
	//cout << Srcimg->width << " " << Srcimg->height << endl;
	IplImage*  dstimg = cvCreateImage(cv::Size(reWidth1, reHeight1), Srcimg->depth, Srcimg->nChannels);
	cvCopyMakeBorder(Srcimg, dstimg, cvPoint((reWidth1 - (Srcimg->width)) / 2, (reHeight1 - (Srcimg->height)) / 2), IPL_BORDER_CONSTANT);
	//归一化处理
	int  charHeight = 40, charWidth = 30;//要保存的单个字符图的size
	int *vArr = new int[reWidth1];//创建投影数组
	int **pic_Arr;//创建投影分割结果的起始点与结束点的记录数组
				  //其中第二维只有两个数，第1个数表示开始位置，第2个数表示结束位置。
	Mat img_4 = cv::cvarrToMat(dstimg);
	//归一化图片的大小。  便于处理
	resize(img_4, img_4, Size(reWidth1, reHeight1));//归一化为re_Width*re_Height大小
	ProjectionCalculate(img_4, vArr);//计算投影数组
	int pic_ArrNumber;
	pic_Arr = ProjectionCut(vArr, reWidth1, pic_ArrNumber);//进行分割
	Mat img_5;
	IplImage pI_1 = img_4;
	IplImage pI_2;
	CvScalar s1;
	int seg_num = 0;
	char numchar[10];
	for (int i = 0; i < pic_ArrNumber; i++)
	{
		int pic_width = pic_Arr[i][1] - pic_Arr[i][0];// 字符宽度 
		if ((pic_width <= 2))
		{
			continue;
		}
		img_5 = cv::Mat(reHeight1, pic_Arr[i][1] - pic_Arr[i][0], CV_8UC1, 1);
		pI_2 = img_5;
		//创建文件夹
		stringstream ss(filename);
		while (getline(ss, filename, '/')) //以/为间隔分割filename的内容,存入filename。这里最后得到的就是不含路径的文件名
			;
		stringstream ss1(filename);
		getline(ss1, filename, '.');//去掉后缀
		CString str = ("rst//" + filename).c_str();
		if (!PathIsDirectory(str))//创建文件夹，有则创建
		{
			CreateDirectory(str, NULL);
		}
		for (int j = 0; j < reHeight1; j++)
		{
			for (int k = pic_Arr[i][0]; k < pic_Arr[i][1]; k++)
			{
				s1 = cvGet2D(&pI_1, j, k);
				cvSet2D(&pI_2, j, k - pic_Arr[i][0], s1);
			}
		}
		float percentage = PxPercentage(img_5);
		if (percentage < MIN_PERCENT)
		{
			continue;
		}
		/////调整输出图片的大小。
		Mat img_w;
		// 初始化输入参数
		int top = (int)abs(charHeight - img_5.rows) / 2;
		int bottom = top;
		int left = (int)abs(charWidth - img_5.cols) / 2;
		int right = left;
		copyMakeBorder(img_5, img_w, top, bottom, left, right, IPL_BORDER_CONSTANT, 0);//0填充扩展边缘
		resize(img_w, img_w, Size(charWidth, charHeight));
		char name[25] = { 0 }; 
		seg_num++;
		sprintf_s(name, "%d.jpg", seg_num);//存储图像
									 //cout<< "./rst/" + filename + "/" + name  + ".jpg" <<endl;
		cv::imwrite("./rst/" + filename + "/" + name + ".jpg", img_w);
		
		//imwrite(name, img_w);
		//cvNamedWindow("1");
		//imshow("1", img_w);

	}cv::imwrite("./rst/" + filename + "/" + filename + ".jpg", img_4);
	delete[] vArr;
	free(pic_Arr);
	return  seg_num;
}

void HistProj::ProjectionCalculate(Mat& mat1, int* vArr)
{
	IplImage pI_1 = mat1;
	CvScalar s1;
	int i, j;

	for (i = 0; i < mat1.cols; i++)
	{
		vArr[i] = 0;
	}

	for (i = 0; i < mat1.rows; i++)
	{
		for (j = 0; j < mat1.cols; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			if (s1.val[0] >50)//20
			{
				vArr[j]++;
			}
		}
	}
}
int** HistProj::ProjectionCut(int* vArr, int width, int& numofcut)
{
	int i, flag = 0;
	numofcut = 0;
	int threshold = 2;//投影阈值
	int **pic_cut = (int**)malloc(MAX_CUT * sizeof(int *));
	for (i = 0; i < width - 1; i++)
	{
		if ((vArr[i] <= threshold) && (vArr[i + 1] > threshold))
		{
			pic_cut[numofcut] = (int*)malloc(2 * sizeof(int));
			pic_cut[numofcut][0] = i;
			flag = 1;
		}
		else if ((vArr[i] > threshold) && (vArr[i + 1] <= threshold) && (flag != 0))
		{
			pic_cut[numofcut][1] = i + 1;
			numofcut++;
			if (numofcut >= MAX_CUT)
			{
				break;
			}
			flag = 0;
		}
	}
	return pic_cut;
}

float HistProj::PxPercentage(Mat& mat1)
{
	IplImage pI_1 = mat1;
	CvScalar s1;
	int width = mat1.rows;
	int height = mat1.cols;
	int i, j;
	float sum = 0, allSum = 0, Percent;
	for (i = 0; i < width; i++)
	{
		for (j = 0; j < height; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			if (s1.val[0]>GRAY_THRESHOLD)
			{
				sum++;
			}
			allSum++;
		}
	}
	Percent = sum / allSum;
	return Percent;
}

/******************************************************************
Function: Thresholding(Mat TempDst, IplImage*&dst)
Description:  二值化
Calls:
Called By: Char_Segment(IplImage* Srcimg)
intput:	param@1 ：Mat TempDst		输入的二值图
param@2 ：IplImage*& dst	引用返回二值化后的图像
Output:
Return:IplImage*& dst		引用返回二值化后的图像
******************************************************************/
int HistProj::Thresholding(Mat TempDst, IplImage*&dst)
{
	int graySum = 0;
	int max = 0;
	for (int i = 1; i < TempDst.rows; i++)
	{
		for (int j = 1; j < TempDst.cols; j++)
		{
			graySum += (int)TempDst.at<uchar>(i, j);
			max = (int)TempDst.at<uchar>(i, j) > max ? (int)TempDst.at<uchar>(i, j) : max;
		}
	}
	int grayMean = graySum / (TempDst.rows* TempDst.cols);//求灰度值均值
														  //对比度增强 将灰度值小于均值的元素赋0，其余的拉伸
	for (int i = 1; i < TempDst.rows; i++)
	{
		for (int j = 1; j < TempDst.cols; j++)
		{
			if ((int)TempDst.at<uchar>(i, j) < grayMean)
				TempDst.at<uchar>(i, j) = 0;
			else
			{
				TempDst.at<uchar>(i, j) = (int)TempDst.at<uchar>(i, j)*((int)TempDst.at<uchar>(i, j) - grayMean) / (255 - grayMean);
			}
		}
	}
	// 转为二值图    
	cvThreshold(m_Grayimg, dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);	// OTSU法二值化
	return 0;
}