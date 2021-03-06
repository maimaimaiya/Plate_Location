//////////////////////////////////////////////////////////////////////////
// Name:			Location.cpp
// Version:		1.0
// Date:			2017-3-23
// MDate:		2017-4-05
// Author:	    帅鹏举
// Desciption:  
// Defines CPlateLocate
// 修改时间：2017-10-21
// 修改内容：showResultMat中修改在原图的裁剪尺寸，使结果裁剪的更大
//////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Location.h"
const float DEFAULT_ERROR = 0.6;
const float DEFAULT_ASPECT = 3.75;
const double WEIGHT_ONE = 0.33748;
const double WEIGHT_TWO = 0.66252;
const double AIM_WEIGHT = 0.7;
const double DEGREE_WEIGHT = 0.6;
#define AREASIZE 5
#define MIN_CHAR_WIDTH 1//车牌字符的最小像素宽度（Char_Segment函数中）
#define MAX_CUT 15//最大的分割数（ProjectionCut函数中）
#define MAX_CONTINUOUS_PX_NUM 11//边框分离时检测横向连续像素为255的像素最大数目（DetectionChange函数）

CLocation::CLocation(Mat src, string name)
{
	/*m_srcImg.create(600, 800, TYPE);
	resize(src, m_srcImg, m_srcImg.size(), 0, 0, INTER_CUBIC);*/
	m_srcImg = src.clone();
	m_deBug = 0;
	m_ImgName = name;
	m_dstImg = m_srcImg.clone();
	m_MorphSizeWidth = DEFAULT_MORPH_SIZE_WIDTH;
	m_MorphSizeHeight = DEFAULT_MORPH_SIZE_HEIGHT;
	m_error = DEFAULT_ERROR;
	m_aspect = DEFAULT_ASPECT;
	m_angle = DEFAULT_ANGLE;
	m_verifyMin = DEFAULT_VERIFY_MIN;
	m_verifyMax = DEFAULT_VERIFY_MAX;
	m_cols = src.cols;
	m_rows = src.rows;
	ColorFeture = new double *[m_rows];
	for (int i = 0; i < m_rows; i++)
		ColorFeture[i] = new double[m_cols]();
	m_featureNum = 0;
	m_featureSum = 0;
	m_featureMean = 0;
	m_aimWeight = 0;
	m_CharNum = 0;
	m_RuleDegree = 0;
	//输出储存地址
	m_OutAddressFirst = "./src/out/";
	//地址+文件名
	strcpy(m_OutAddress, (m_OutAddressFirst + m_ImgName).c_str());
	m_aimWeight = AIM_WEIGHT;
	blue = new int *[m_rows];
	white = new int *[m_rows];
	for (int i = 0; i < m_rows; i++)
	{
		white[i] = new int[m_cols]();
		blue[i] = new int[m_cols]();
	}
}


CLocation::~CLocation()
{
}

void CLocation::PreTreatment(Mat src)
{
	Mat dst_Blur, dst_Gray;
	GaussianBlur(src, dst_Blur, Size(3, 3), 0, 0, BORDER_DEFAULT);			 //高斯模糊
	cvtColor(dst_Blur, dst_Gray, CV_BGR2GRAY);								 //灰度化
	m_dstImg = dst_Gray;
}

///Sobel轮廓检测
void CLocation::SobelDetection()
{
	Mat src = m_dstImg;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat grad;
	int scale = SOBEL_SCALE;
	int delta = SOBEL_DELTA;
	int ddepth = SOBEL_DDEPTH;
	/// X 梯度  
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );  
	//Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.  
	Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	/// Y 梯度    
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );  
	Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	/// Total Gradient (approximate)  
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	m_dstImg = grad;

	threshold(m_dstImg, m_dstImg, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY); //二值化

	if (m_deBug)
	{
		imshow("Sobel二值化", m_dstImg);
		cvWaitKey(0);
	}
	Mat element = getStructuringElement(MORPH_RECT, Size(25, 2)); //设置横向25*2矩形模版

	morphologyEx(m_dstImg, m_dstImg, MORPH_CLOSE, element);			//形态学处理

	if (m_deBug)
	{
		imshow("Sobel后形态学", m_dstImg);
		cvWaitKey(0);
	}
}

///Canny轮廓检测
void CLocation::CannyDetection()
{
	IplImage imgTmp = m_dstImg;
	IplImage* pImg = cvCloneImage(&imgTmp);
	IplImage* pCannyImg = NULL;

	double low_thresh = 0.0;
	double high_thresh = 0.0;

	pCannyImg = cvCreateImage(cvGetSize(pImg), IPL_DEPTH_8U, 1);

	CvMat *dx = (CvMat*)pImg;
	CvMat *dy = (CvMat*)pCannyImg;
	if (low_thresh == 0.0 && high_thresh == 0.0)
	{
		AdaptiveFindThreshold(pImg, &low_thresh, &high_thresh, 3);
		cout << "low_thresh:  " << low_thresh << endl;
		cout << "high_thresh: " << high_thresh << endl;
	}
	cvCanny(pImg, pCannyImg, low_thresh, high_thresh, 3);
	m_dstImg = cvarrToMat(pCannyImg, true);

	IplImage qImg = IplImage(m_dstImg); // cv::Mat -> IplImage
	cvSaveImage("./src/out/Canny.jpg", &qImg);

	imshow("canny", m_dstImg);
	cvWaitKey(0);
}


//！求取Canny上下阈值
void CLocation::AdaptiveFindThreshold(const CvArr* image, double *low, double *high, int aperture_size)
{
	Mat src = cvarrToMat(image, true);
	const int cn = src.channels();
	Mat dx(src.rows, src.cols, CV_16SC(cn));
	Mat dy(src.rows, src.cols, CV_16SC(cn));

	Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
	Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

	CvMat _dx = dx, _dy = dy;
	_AdaptiveFindThreshold(&_dx, &_dy, low, high);

}

// 仿照matlab，自适应求高低两个门限                                              
void CLocation::_AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high)
{
	CvSize size;
	IplImage *imge = 0;
	int i, j;
	CvHistogram *hist;
	int hist_size = 255;
	float range_0[] = { 0,256 };
	float* ranges[] = { range_0 };
	double PercentOfPixelsNotEdges = 0.7;
	size = cvGetSize(dx);
	imge = cvCreateImage(size, IPL_DEPTH_32F, 1);
	// 计算边缘的强度, 并存于图像中                                          
	float maxv = 0;
	for (i = 0; i < size.height; i++)
	{
		const short* _dx = (short*)(dx->data.ptr + dx->step*i);
		const short* _dy = (short*)(dy->data.ptr + dy->step*i);
		float* _image = (float *)(imge->imageData + imge->widthStep*i);
		for (j = 0; j < size.width; j++)
		{
			_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
			maxv = maxv < _image[j] ? _image[j] : maxv;

		}
	}
	if (maxv == 0) {
		*high = 0;
		*low = 0;
		cvReleaseImage(&imge);
		return;
	}

	// 计算直方图                                                            
	range_0[1] = maxv;
	hist_size = (int)(hist_size > maxv ? maxv : hist_size);
	hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(&imge, hist, 0, NULL);
	int total = (int)(size.height * size.width * PercentOfPixelsNotEdges);
	float sum = 0;
	int icount = hist->mat.dim[0].size;

	float *h = (float*)cvPtr1D(hist->bins, 0);
	for (i = 0; i < icount; i++)
	{
		sum += h[i];
		if (sum > total)
			break;
	}
	// 计算高低门限                                                          
	*high = (i + 1) * maxv / hist_size;
	*low = *high * 0.4;
	cvReleaseImage(&imge);
	cvReleaseHist(&hist);

}


//! 显示最终生成的车牌图像，便于判断是否成功进行了旋转。
Mat CLocation::showResultMat(Mat src, Size rect_size, Point2f center, int index)
{
	Mat img_crop;
	rect_size.width += 20;
	rect_size.height += 10;  //修改在原图上裁剪的尺寸
	getRectSubPix(src, rect_size, center, img_crop);

	Mat resultResized;

	resultResized.create(HEIGHT, WIDTH, TYPE);

	resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);

	return resultResized;
}

// Defines CPlateLocate
// 修改时间：2017-4-14
// 修改内容：颜色特征提取

//！ 颜色特征提取
bool CLocation::ColorFeatureExtraction()
{
	Mat src = m_srcImg.clone(); //深拷贝
	int cPointB, cPointG, cPointR;
	for (int i = 1; i < m_rows; i++)
	{
		for (int j = 1; j < m_cols; j++)
		{
			cPointB = src.at<Vec3b>(i, j)[0];
			cPointG = src.at<Vec3b>(i, j)[1];
			cPointR = src.at<Vec3b>(i, j)[2];

			int dif_B_R = cPointB - cPointR; //计算蓝色特征
			int dif_B_G = cPointB - cPointG;
			if (dif_B_G > 0 && dif_B_R > 0)
			{
				ColorFeture[i][j] = WEIGHT_ONE * dif_B_R + WEIGHT_TWO * dif_B_G;
				if (ColorFeture[i][j] > 0)
				{
					m_featureNum++;   //统计非0的特征点数量
					m_featureSum += ColorFeture[i][j] / 1000; //计算所有特征值总和
				}
			}
			else
				ColorFeture[i][j] = 0;
		}
	}
	if (m_featureNum > 0)
	{
		m_featureMean = m_featureSum / m_featureNum; //计算特征均值
		m_featureMean *= 1000;
	}
	int m_aimNum = 0;
	for (int i = 1; i < m_rows; i++)
	{
		for (int j = 1; j < m_cols; j++)
		{
			if (ColorFeture[i][j] > m_featureMean && ColorFeture[i][j] > 0) //求取目标点数量
				m_aimNum++;
		}
	}

	m_aimWeight = m_aimNum *1.0 / m_featureNum; //求取目标点权值
	//cout << m_aimWeight << " " << m_aimNum << " " << m_featureNum << endl;
	Mat dst = src;
	dst = Binarization(dst, m_featureMean, m_aimWeight); //根据目标区域二值化

	//imshow("二值化", dst);
	//cvWaitKey(0);
	if (m_deBug)
	{
		imshow("颜色+Sobel二值化", dst);
 		cvWaitKey(0);
	}

	dst = Morphological(dst); //形态学处理

	if (m_deBug)
	{
		imshow("颜色+Sobel二值化后形态学", dst);
		cvWaitKey(0);
	}
	//ContourTracking(dst);

	cvtColor(dst, dst, CV_BGR2GRAY);

	if (ContourSearch(dst)) //筛选区域
		return true;
	else
		return false;
}

//！根据目标区域二值化
Mat CLocation::Binarization(Mat src, double mean, double aimWeight)
{
	Mat dst = src;
	double thresh = mean*(m_aimWeight + (1 - m_aimWeight) / aimWeight); // 目标点
	//cout << thresh << endl;
	//二值化
	for (int i = 1; i < m_rows; i++)
	{
		for (int j = 1; j < m_cols; j++)
		{
			//若大于该阈值且在边缘检测后的结果内则赋值255
			if (ColorFeture[i][j] > thresh && (int)m_dstImg.at<uchar>(i, j) == 255)
			{
				dst.at<Vec3b>(i, j)[0] = 255;
				dst.at<Vec3b>(i, j)[1] = 255;
				dst.at<Vec3b>(i, j)[2] = 255;
				ColorFeture[i][j] = 1;
			}
			else
			{
				dst.at<Vec3b>(i, j)[0] = 0;
				dst.at<Vec3b>(i, j)[1] = 0;
				dst.at<Vec3b>(i, j)[2] = 0;
				ColorFeture[i][j] = 0;
			}
		}
	}
	return dst;
}

//！形态学处理
Mat CLocation::Morphological(Mat src)
{
	//imshow("形态学前", src);
	//cvWaitKey(0);

	Mat element = getStructuringElement(MORPH_RECT, Size(25, 2)); //先横向25*2
	//形态学处理
	morphologyEx(src, src, MORPH_CLOSE, element);

	//imshow("第一次形态学", src);
	//cvWaitKey(0);

	element = getStructuringElement(MORPH_RECT, Size(5, 2)); //横向5*2
	//形态学处理
	morphologyEx(src, src, MORPH_OPEN, element);

	//imshow("第二次形态学", src);
	//cvWaitKey(0);

	element = getStructuringElement(MORPH_RECT, Size(2, 5)); //纵向2*5
	//形态学处理
	morphologyEx(src, src, MORPH_OPEN, element);

	//imshow("第三次形态学", src);
	//cvWaitKey(0);

	return src;
}

bool CLocation::ContourSearch(Mat src)
{
	//Find 轮廓 of possibles plates
	vector< vector< Point> > contours;
	findContours(src,
		contours, // a vector of contours
		CV_RETR_EXTERNAL, // 提取外部轮廓
		CV_CHAIN_APPROX_NONE); // 每个轮廓的所有像素

	vector<vector<Point> >::iterator itc = contours.begin();

	vector<RotatedRect> rects;

	int t = 0;
	while (itc != contours.end())
	{
		//利用minAreaRect()返回最小外接矩形
		RotatedRect mr = minAreaRect(Mat(*itc));
	
		//判断是否满足宽高比 不满足则删除
		if (!verifySizes(mr))
		{
			itc = contours.erase(itc);
		}
		else
		{
			++itc;
			rects.push_back(mr);
		}
	}
	int k = 1;
	vector<Mat> resultVec;
	for (int i = 0; i < rects.size(); i++)
	{
		RotatedRect minRect = rects[i];
		if (verifySizes(minRect))
		{
			//倾斜校正 可以进一步筛出部分图片
			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			Size rect_size = minRect.size;
			if (r < 1)
			{
				angle = 90 + angle;
				swap(rect_size.width, rect_size.height);
			}
			//如果抓取的方块旋转超过m_angle角度，则不是车牌，放弃处理
			if (angle - m_angle < 0 && angle + m_angle > 0)
			{
				//Create and rotate image
				Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
				Mat img_rotated;
				//仿射
				warpAffine(m_srcImg, img_rotated, rotmat, m_dstImg.size(), CV_INTER_CUBIC);

				//Mat resultMat(img_rotated, minRect);
				Mat resultMat;
				resultMat = showResultMat(img_rotated, rect_size, minRect.center, k++);
				resultVec.push_back(resultMat);
			}
		}
	}

	double DegreeMax = -1;
	int LastKey = -1;
	for (int k = 0; k < resultVec.size(); k++)
	{

		Mat TempDst;/*(HEIGHT, WIDTH, CV_8UC1, Scalar(0));
		for (int row = 0; row < HEIGHT; row++)
		{
			for (int col = 0; col < WIDTH; col++)
			{
				if (Blue_Judge(row, col, resultVec[k]))
				{
					TempDst.at<uchar>(row, col) = 255;
				}
			}
		}*/
		cvtColor(resultVec[k], TempDst, CV_BGR2GRAY);
	//	dilate(TempDst, TempDst,Size(3,3));//膨胀
	//	erode(TempDst, TempDst);//腐蚀
		//threshold(TempDst, TempDst, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY); //二值化
	//	imshow("da前", resultVec[k]);
		//waitKey(0);
		int graySum = 0;
		for (int i = 1; i < TempDst.rows; i++)
		{
			for (int j = 1; j < TempDst.cols; j++)
			{
				graySum += (int)TempDst.at<uchar>(i, j);
			}
		}

		int grayMean = graySum / (TempDst.rows* TempDst.cols);//求灰度值均值


		//对比度增强 将灰度值小于均值的元素赋0
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
		
		threshold(TempDst, TempDst, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY); //二值化*/
		if (m_deBug)
		{
			imshow("da", TempDst);
			waitKey(0);
		}
		// 进行车牌特征值计算
 		double Degree = VerticalProjection(TempDst);

		//车牌对应最大的特征值
		if (Degree > DegreeMax)
		{
			DegreeMax = Degree;
			LastKey = k;
		}
	}
	if (m_deBug)
	{
		if (resultVec.size() != 0)
		{
			imshow("result", resultVec[LastKey]);
   			cvWaitKey(0); 
		}
	}
	//imshow("Last", resultVec[LastKey]);
	if ((DegreeMax != -1 || LastKey != -1 )/*&& DegreeMax>0.2*/)
	{
		imwrite(m_OutAddress, resultVec[LastKey]);
		//IplImage qImg = IplImage(resultVec[LastKey]); // cv::Mat -> IplImage
		//cvSaveImage(m_OutAddress, &qImg);
		SetResultImage(resultVec[LastKey]);
		vector< vector< Point> >().swap(contours);
		vector< RotatedRect >().swap(rects);
		vector< Mat >().swap(resultVec);
		return true;
	}
	vector< vector< Point> >().swap(contours);
	vector< RotatedRect >().swap(rects);
	vector< Mat >().swap(resultVec);
	return false;


}

//! 对minAreaRect获得的最小外接矩形，用纵横比进行判断
bool CLocation::verifySizes(RotatedRect mr)
{
	float error = m_error;
	//Spain car plate size: 52x11 aspect 4,7272
	//China car plate size: 440mm*140mm，aspect 3.142857
	float aspect = m_aspect;
	//Set a min and max area. All other patchs are discarded
	//int min= 1*aspect*1; // minimum area
	//int max= 2000*aspect*2000; // maximum area
	int min = 44 * 14 * m_verifyMin; // minimum area
	int max = 44 * 14 * m_verifyMax; // maximum area
									 //Get only patchs that match to a respect ratio.
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;

	int area = mr.size.height * mr.size.width;
	float r = (float)mr.size.width / (float)mr.size.height;
	if (r < 1)
	{
		r = (float)mr.size.height / (float)mr.size.width;
	}

	if ((area < min || area > max) || (r < rmin || r > rmax))
	{
		return false;
	}
	else
	{
		return true;
	}
}

double CLocation::VerticalProjection(Mat src)
{
	/*DetectionChange(src, src);//去边框
	Point2f center = Point2f(src.cols / 2, src.rows / 2);  // 旋转中心
	Mat rotateMat = getRotationMatrix2D(center, 180, 1);//旋转矩阵
	warpAffine(src, src, rotateMat, src.size());//这是我的一个意外发现，目前没找到原因，就是我也不知道为什么在经
	warpAffine(src, src, rotateMat, src.size());//过两次翻转（理论上相当于没有改变什么）后，分割正确率提高了。
	//imshow("边框分离", src);
	cvWaitKey(0);
	int reHeight = 30, reWidth = 100;//归一化处理的size
	int *vArr = new int[reWidth];//创建投影数组

	int **pic_Arr;//创建投影分割结果的起始点与结束点的记录数组
				  //其中第二维只有两个数，第1个数表示开始位置，第2个数表示结束位置。
	Mat img_4;
	//归一化图片的大小。  便于处理
	resize(src, img_4, Size(reWidth, reHeight));//归一化为re_Width*re_Height大小
	ProjectionCalculate(img_4, vArr);//计算投影数组
	int pic_ArrNumber;
	pic_Arr = ProjectionCut(vArr, reWidth, pic_ArrNumber);//进行分割

	Mat img_5;
	IplImage pI_1 = img_4;
	IplImage pI_2;
	CvScalar s1;
	int seg_num = 0;

	char numchar[10];
	Mat seg_perimg[10];
	/*if (pic_ArrNumber < 7)
	{
		delete[] vArr;
		free(pic_Arr);
		return pic_ArrNumber;
	}
	int LastNum = pic_ArrNumber;
	int TempNum = 0;
	double wid = reWidth / 10;
	for (int i = 0; i < pic_ArrNumber; i++)
	{
		int pic_width = pic_Arr[i][1] - pic_Arr[i][0];// 字符宽度 
		/*if (pic_width <= MIN_CHAR_WIDTH&&pic_Arr[i][1] <= reWidth / 2 && pic_ArrNumber <= 3)
		{
			LastNum--;
			continue;
		}
		if (pic_width < wid / 4)
			LastNum--;
		else if (pic_width < wid / 3 || pic_width > wid * 1.5)
			TempNum++;
	}
	*/


	m_Projection = new int[src.cols]();
	//memset(m_Projection, 0, sizeof(m_Projection));
	for (int j = 1; j < src.cols; j++)
	{
		//舍弃上下20% 只取中间60%做统计
		for (int i = 1 + 0.2*src.rows; i < src.rows*(1 - 0.20); i++)
		{
			if ((int)src.at<uchar>(i, j) > 0)
			{
				m_Projection[j]++;
			}
		}
		//cout << m_Projection[j];
	}
	//cout << endl;
	bool CharState = false;
	int *m_CharWidth = new int[src.cols]();
	m_CharNum = 0;
	//统计字符个数
	int Zero_num = 1;
	for (int i = 0; i < src.cols; i++)
	{
		if (m_Projection[i] != 0 && !CharState)
		{
			CharState = true;
		}
		else if (m_Projection[i] == 0 && CharState && (Zero_num >= 3||i>=src.cols-3))
		{
			m_CharWidth[m_CharNum] -= Zero_num;
			m_CharNum++;
			CharState = false;
			Zero_num = 1;
		}
		else if (m_Projection[i] == 0 && CharState)
			Zero_num++;
		else if (m_Projection[i] != 0 && CharState)
			Zero_num = 0;
		if (CharState)
			m_CharWidth[m_CharNum]++;
	}
	double wid = src.cols / 10;
	int LastNum = m_CharNum;
	double TempNum = 0;
	for (int i = 0; i < m_CharNum; i++)
	{
		if (m_CharWidth[i] < wid / 4)
			LastNum--;
		else if (m_CharWidth[i] < wid / 3 || m_CharWidth[i] > wid * 1.5)
			TempNum++;
		//cout << m_CharWidth[i] << " ";
	}
	
  	if (LastNum < 5)
		m_RuleDegree = 0;
	else
	{
		//计算待选区域规则度
		//	1 - 0.6*(abs(7 - LastNum)) / max(7, LastNum) - 0.4*(TempNum / LastNum);
		m_RuleDegree = 1 - DEGREE_WEIGHT*(abs(DEFAULT_CHAR_NUM - LastNum) *1.0 / max(DEFAULT_CHAR_NUM, LastNum)) - (1 - DEGREE_WEIGHT)*(TempNum *1.0 / LastNum);
	}
	//cout << LastNum << " " << TempNum << " "<<m_RuleDegree<< endl;
	return m_RuleDegree;
}

//////////////////////////////////////////////////////////////////////////
// Author:	    帅鹏举
// 修改时间：2017-7-10
// 修改内容：轮廓跟踪
//////////////////////////////////////////////////////////////////////////

void CLocation::ContourTracking(Mat src)
{
	int num;
	int fill_value;
	int xs, ys;
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<Vec3b>(i, j)[0] = 0;
			dst.at<Vec3b>(i, j)[1] = 0;
			dst.at<Vec3b>(i, j)[2] = 0;
		}
	}
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if ((int)src.at<uchar>(y, x) == MAX_BRIGHTNESS)
			{
				num = ContourMarking(y, x, src);
				if (num > MAX_PERIMETER)
					fill_value = MAX_BRIGHTNESS;
				else fill_value = GRAY;
				xs = x;  ys = y;
				//src.at<uchar>(ys, xs) = 0;
				src.at<Vec3b>(ys, xs)[0] = 0;
				src.at<Vec3b>(ys, xs)[1] = 0;
				src.at<Vec3b>(ys, xs)[2] = 0;
				dst.at<Vec3b>(ys, xs)[0] = fill_value;
				dst.at<Vec3b>(ys, xs)[1] = fill_value;
				dst.at<Vec3b>(ys, xs)[2] = fill_value;
				if (num > 1) {
					for (int i = 0; i < num - 1; i++) {
						xs = xs + Freeman[chain_code[i]][0];
						ys = ys + Freeman[chain_code[i]][1];
						src.at<Vec3b>(ys, xs)[0] = 0;
						src.at<Vec3b>(ys, xs)[1] = 0;
						src.at<Vec3b>(ys, xs)[2] = 0;
						dst.at<Vec3b>(ys, xs)[0] = fill_value;
						dst.at<Vec3b>(ys, xs)[1] = fill_value;
						dst.at<Vec3b>(ys, xs)[2] = fill_value;
					}
				}
			}

		}
	}
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			if ((int)dst.at<uchar>(y, x) == GRAY) {
				dst.at<Vec3b>(y, x)[0] = 0;
				dst.at<Vec3b>(y, x)[1] = 0;
				dst.at<Vec3b>(y, x)[2] = 0;
				for (int i = 0; i < 8; i++) {
					xs = x + Freeman[i][0];
					ys = y + Freeman[i][1];
					if (xs >= 0 && xs <= src.cols &&
						ys >= 0 && ys <= src.rows &&
						(int)dst.at<uchar>(ys, xs) == MAX_BRIGHTNESS)
						dst.at<Vec3b>(y, x)[0] = MAX_BRIGHTNESS;
						dst.at<Vec3b>(y, x)[1] = MAX_BRIGHTNESS;
						dst.at<Vec3b>(y, x)[2] = MAX_BRIGHTNESS;
				}
			}
		}
	}
	imshow("轮廓跟踪", dst);
	cvWaitKey(0);
}

int CLocation::ContourMarking(int x_start, int y_start, Mat src)
{
	int  x, y;             /* 輪郭線上の現在の注目画素の座標 */  /*当前的目标象素的上轮廓的坐标*/
	int xs, ys;             /* 注目画素の周囲の探索点の座標   */  /*搜索点的感兴趣的像素周围的坐标*/
	int code, num;          /* 輪郭点のチェーンコード, 総数   */  /*轮廓点的链码，总数*/
	int i, counter, detect; /* 制御変数など                   */  /*制约变量*/
	counter = 0;			/* 孤立点のチェック */  /*检查孤立点*/
	for (i = 0; i < 8; i++) {
		xs = x_start + Freeman[i][0];
		ys = y_start + Freeman[i][1];
		if (xs >= 0 && xs <= src.cols && ys >= 0 && ys <= src.rows
			&& (int)src.at<uchar>(ys, xs) == MAX_BRIGHTNESS) counter++;
	}
	if (counter == 0) num = 1;  /* (x_start,y_start)は孤立点 */ /*起点是孤立点*/
	else {
		/* 探索開始 */
		num = -1;   x = x_start;    y = y_start;    code = 0;
		do {
			detect = 0; /* 次の点をみつけたとき１にする */
						/* 初期探索方向の決定 */   /*初始方向的确定*/
			code = code - 3;   if (code < 0) code = code + 8;
			do {
				xs = x + Freeman[code][0];
				ys = y + Freeman[code][1];
				if (xs >= 0 && xs <= src.cols && ys >= 0 &&
					ys <= src.rows &&
					(int)src.at<uchar>(ys, xs) == MAX_BRIGHTNESS) {
					detect = 1;  /* 次の点を検出 */  /*下一个点的检测*/
					num++;
					if (num > MAX_CNTR) {
						printf("輪郭線の画素数 > %d\n", MAX_CNTR);
						exit(1);
					}
					chain_code[num] = code;
					x = xs;  y = ys;
				}
				code++;  if (code > 7) code = 0;
			} while (detect == 0);
		} while (x != x_start || y != y_start); /* 開始点の検出まで */  /*检测起点*/
		num = num + 2;  /* chain_code[ ]の添え字とのずれの修正 */  /*chain_code[]的下标偏差的修改*/
	}
	return(num);
}


bool CLocation::Color_Contour()
{
	
	//Mat dst = m_srcImg.clone();
	Mat dst(m_rows, m_cols, CV_8UC1, Scalar(0));
	GaussianBlur(m_srcImg, srcBlur, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//cvtColor(src, srcHSV, CV_BGR2HSV);
	int H, S, V;

	//一次遍历统计出每AREASiZE*AREASIZE区域大小内的白色或蓝色像素点个数
	for (int i = AREASIZE; i < m_rows; i++)
	{
		for (int j = AREASIZE; j < m_cols; j++)
		{
			bool blue_status = false;
			bool white_status = false;

			if (Blue_Judge(i, j,srcBlur))
				blue[i][j] = blue[i - 1][j] + blue[i][j - 1] - blue[i-1][j-1] + 1;
			else
				blue[i][j] = blue[i - 1][j] + blue[i][j - 1] - blue[i - 1][j - 1];


			if (White_Judge(i, j,srcBlur))
				white[i][j] = white[i - 1][j] + white[i][j - 1] - white[i - 1][j - 1] + 1;
			else
				white[i][j] = white[i - 1][j] + white[i][j - 1] - white[i - 1][j - 1];


			if (blue[i][j] - blue[i][j - AREASIZE] - blue[i - AREASIZE][j] + blue[i - AREASIZE][j - AREASIZE]>0) // i-AREASIZE/2, j-AREASIZE/2 点的蓝色点数量
				blue_status = true;


			if (white[i][j] - white[i][j - AREASIZE] - white[i - AREASIZE][j] + white[i - AREASIZE][j - AREASIZE]>0) // i-AREASIZE/2, j-AREASIZE/2 点的蓝色点数量
				white_status = true;


			if (blue_status&&white_status)
			{
				Color_Mark[i][j] = 1;
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				Color_Mark[i][j] = 0;
			}
		}
	}
	
	if (m_deBug)
	{
		imshow("test0", dst);
		cvWaitKey(0);
	}
	//cvtColor(dst, dst, CV_RGB2GRAY);
	//threshold(dst, dst, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	Mat element = getStructuringElement(MORPH_RECT, Size(25, 5)); //设置横向25*2矩形模版
	morphologyEx(dst, dst, MORPH_CLOSE, element);			//形态学处理

	for (int i = 1; i < m_rows; i++)
	{
		for (int j = 1; j < m_cols; j++)
		{
			if ((int)dst.at<uchar>(i, j) == 255 && 255 == (int)m_dstImg.at<uchar>(i, j))
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}
	dst = Morphological(dst);
	if (m_deBug)
	{
		imshow("形态学", dst);
		cvWaitKey(0);
	}

	if (ContourSearch(dst)) //筛选区域
		return true;
	else
		return false;

}

/**
* Mat &temp 需要判断的图像
* x 需要判断点的x坐标
* y 需要判断点的y坐标
* return 该点是否是蓝色像素点
*/
bool CLocation::Blue_Judge(int x, int y, Mat &temp)
{
	double b = (double)temp.at<Vec3b>(x, y)[0];
	double g = (double)temp.at<Vec3b>(x, y)[1];
	double r = (double)temp.at<Vec3b>(x, y)[2];
	double h;
	double numerator = (r - g + r - b) / 2;
	double denominator = sqrt(pow((r - g), 2) + (r - b)*(g - b));
	if (denominator == 0)
		h = 0;
	else
		h = acos(numerator / denominator) * 180 / 3.14;
	if (b > g)
		h = 360 - h;
	double s = 1 - (double)3.0*min(r, min(g, b)) / (r + g + b);
	double i = (r + g + b) / 3;
	//if(h>180&&h<300 && b > 50)
	if (b *1.0> 1.4*g*1.0&&b*1.0 > 1.4*r*1.0&&b > 50)
		return true;
	return false;
}

/**
* Mat &temp 需要判断的图像
* x 需要判断点的x坐标
* y 需要判断点的y坐标
* return 该点是否是白色像素点
*/
bool CLocation::White_Judge(int x, int y,Mat &temp)
{
	double b = (double)temp.at<Vec3b>(x, y)[0];
	double g = (double)temp.at<Vec3b>(x, y)[1];
	double r = (double)temp.at<Vec3b>(x, y)[2];
	double h;
	double numerator = (r - g + r - b) / 2;
	double denominator = sqrt(pow((r - g), 2) + (r - b)*(g - b));
	if (denominator == 0)
		h = 0;
	else
		h = acos(numerator / denominator) * 180 / 3.14;
	if (b > g)
		h = 360 - h;
	double s = 1 - (double)3.0*min(r, min(g, b)) / (r + g + b);
	double i = (r + g + b) / 3;
	if ((i >= 200 && i<250) || (s <= 0.25&&s >= 0)) //白色点判断
													//double S = b + g + r;
													//if (b*1.0 < 0.4*S&&g*1.0 < 0.4*S&&r*1.0 < 0.4*S&&S>150)
		return true;
	return false;
}

/******************************************************************
Function: detectionChange(Mat& mat1, Mat& dst, int number)
Description:  分离去除边框
Calls:
Called By: Char_Segment(IplImage* Srcimg)
intput:	param@1 :Mat& mat1		输入二值化后图像
param@2 :Mat& dst		输出分离去除边框后的图像
Output:
Return: Mat& dst 输出分离去除边框后的图像
*******************************************************************/
void CLocation::DetectionChange(Mat& mat1, Mat& dst)
{
	IplImage pI_1 = mat1, pI_2;
	CvScalar s1, s2;
	int height = mat1.rows;
	int width = mat1.cols;
	int sum_1 = 0, sum_2 = 0, width_1 = 0, width_2 = 0;
	int i, j;
	for (i = 0; i < height; i++)
	{
		sum_1 = 0;
		sum_2 = 0;
		for (j = 0; j < width - 1; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			s2 = cvGet2D(&pI_1, i, j + 1);
			if (((int)s1.val[0]) != ((int)s2.val[0]))
			{
				sum_1++;
				sum_2 = 0;
			}
			else
			{
				sum_2++;
			}
			if (sum_2 > width / 5)//保证可以检测到超过5个轮廓
			{
				sum_1 = 0;
				break;
			}
		}
		width_1 = i;
		if (sum_1 >= MAX_CONTINUOUS_PX_NUM)
		{
			break;
		}
	}

	for (i = height - 1; i > 0; i--)
	{
		sum_1 = 0;
		sum_2 = 0;
		for (j = 0; j < width - 1; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			s2 = cvGet2D(&pI_1, i, j + 1);
			if (((int)s1.val[0]) != ((int)s2.val[0]))
			{
				sum_1++;
				sum_2 = 0;
			}
			else
			{
				sum_2++;
			}
			if (sum_2 > width/5)
			{
				sum_1 = 0;
				break;
			}
		}
		width_2 = i;
		if (sum_1 >= MAX_CONTINUOUS_PX_NUM)
		{
			break;
		}
	}
	if (width_2 <= width_1)
	{
		width_2 = height - 1;
	}
	dst = cv::Mat(width_2 - width_1 + 1, width, CV_8UC1, 1);
	pI_2 = dst;
	for (i = width_1; i <= width_2; i++)
	{
		for (j = 0; j < width; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			cvSet2D(&pI_2, i - width_1, j, s1);
		}
	}
}


/******************************************************************
Function: ProjectionCalculate(Mat& mat1, int* vArr, int number)
Description:  计算投影数组
Calls:
Called By: Char_Segment(IplImage* Srcimg)
intput:	param@1 :Mat& mat1		输入二值化后图像
param@2 :int* vArr		返回的投影数组
param@3 :int number
Output:
Return: vArr 输出投影数组
******************************************************************/
void CLocation::ProjectionCalculate(Mat& mat1, int* vArr)
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
			if (s1.val[0] > 20)
			{
				vArr[j]++;
			}
		}
	}
}

/******************************************************************
Function: ProjectionCut(int* vArr, int width, int& numofcut)
Description:  计算返回分割数组，第二维0为开始点，1为结束点横坐标
Calls:
Called By: Char_Segment(IplImage* Srcimg)
intput:	param@1 :int* vArr		投影数组
param@2 :int width		数组长度（即图片宽度）
param@3 :int& numofcut	返回分割数目
Output:
Return: int& numofcut		引用返回分割数目
int** pic_cut		直接返回分割数组
******************************************************************/
int** CLocation::ProjectionCut(int* vArr, int width, int& numofcut)
{
	int i, flag = 0;
	numofcut = 0;
	int threshold = 1;//投影阈值
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

void CLocation::SetResultImage(Mat img)
{
	ResultImage = img;
}

Mat CLocation::GetResultImage()
{
	return ResultImage;
}
