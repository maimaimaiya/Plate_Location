#include "stdafx.h"
#include "Region_Seg.h"
#include "io.h"
#include "fstream"
#include "vector"
#include "shlwapi.h"
#include <sstream>
#pragma comment(lib,"shlwapi.lib")
struct node
{
	int x = 0;
	int y = 0;
	int width = 0;
	int height = 0;
}dp[10];
bool cmp(node min1, node max1)
{
	return min1.x < max1.x;
}

Region_Seg::Region_Seg()
{
}
IplImage* Region_Seg::Char_PreSeg(IplImage *m_img)
{
	IplImage* img_gray = cvCreateImage(cvGetSize(m_img), IPL_DEPTH_8U, 1);
	if (m_img->nChannels == 3)
	{
		cvCvtColor(m_img, img_gray, CV_BGR2GRAY);
	}
	else
	{
		img_gray = cvCloneImage(m_img);
	}
	//cvCvtColor(m_img, img_gray, CV_BGR2GRAY);//�ҶȻ�
	Thresholding(img_gray);//��ֵ��
	//cvThreshold(img_gray, img_gray,0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);	// OTSU����ֵ��
	IplImage *lgimg= cvCreateImage(cvGetSize(img_gray), img_gray->depth, img_gray->nChannels);
	IplImage *img_gray1= cvCreateImage(cvGetSize(img_gray), img_gray->depth, img_gray->nChannels);
	cvThreshold(img_gray, lgimg, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);	// OTSU����ֵ��
	cvCanny(img_gray, img_gray, 61, 61 * 3, 3);
	img_gray1 =synthetic(lgimg, img_gray);
	//cvShowImage("test", img_gray1);
	//cvWaitKey(0);
	cvDilate(img_gray1, img_gray1);//����
	cvErode(img_gray1, img_gray1);//��ʴ
	//cvShowImage("test", img_gray1);
	//cvWaitKey(0);
	IplImage* img = cvCreateImage(cv::Size(reWidth, reHeight), img_gray1->depth, img_gray1->nChannels);
	cvResize(img_gray1, img);//��һ��Ϊre_Width*re_Height��С
	//cout << img->width << " " << img->height << endl;
	//cvShowImage("test", img_gray);	
	//cvWaitKey(0);
	return img_gray1;
}
IplImage* Region_Seg::Char_PreSeg1(IplImage *m_img)
{
	IplImage* img_gray = cvCreateImage(cvGetSize(m_img), IPL_DEPTH_8U, 1);
	if (m_img->nChannels == 3)
	{
		cvCvtColor(m_img, img_gray, CV_BGR2GRAY);
	}
	else
	{
		img_gray = cvCloneImage(m_img);
	}
	cvCvtColor(m_img, img_gray, CV_BGR2GRAY);//�ҶȻ�
	Thresholding(img_gray);//��ֵ��
	//cvCanny(img_gray, img_gray, 61, 61 * 3, 3);
	cvThreshold(img_gray, img_gray, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);	// OTSU����ֵ��
	cvDilate(img_gray, img_gray);//����
	cvErode(img_gray, img_gray);//��ʴ
	IplImage* img = cvCreateImage(cv::Size(reWidth, reHeight), IPL_DEPTH_8U, 1);
	cvResize(img_gray, img);//��һ��Ϊre_Width*re_Height��С
							//cout << img->width << " " << img->height << endl;
							//cvShowImage("test", img_gray);
	return img_gray;
}
/*��ֵ��*/
void Region_Seg::Thresholding(IplImage* &img_g)
{
	//��Ҷ�ֵ��ֵ
	cv::Mat TempDst = cv::cvarrToMat(img_g);
	int graySum = 0;
	for (int i = 1; i < TempDst.rows; i++)
	{
		for (int j = 1; j < TempDst.cols; j++)
		{
			graySum += (int)TempDst.at<uchar>(i, j);
		}
	}
	int grayMean = graySum / (TempDst.rows* TempDst.cols);
	//�Աȶ���ǿ ���Ҷ�ֵС�ھ�ֵ��Ԫ�ظ�0
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

}
void Region_Seg::borderCut(cv::Mat& mat1, cv::Mat& dst)
{
	IplImage pI_1 = mat1, pI_2;
	CvScalar s1, s2;
	int height = mat1.rows;
	int width = mat1.cols;
	int sum_1 = 0, sum_2 = 0, width_1 = 0, width_2 = 0;
	int i, j;
	for (i = 0; i < height; i++)//һ��
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
			if (sum_2 > width / 5)//��֤���Լ�⵽����5������
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
			if (sum_2 > width)
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
	//dst = cv::Mat(width_2 - width_1 + 1, width, CV_8UC1, 1);
	dst = cv::Mat(width_2 - width_1 +1, width, CV_8UC1, 1);
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
IplImage* Region_Seg::synthetic(IplImage *logimg, IplImage *oustimg)
{
	IplImage* thrsold = cvCreateImage(cvGetSize(logimg), logimg->depth, 1);
	CvScalar s1, s2,s3;
	for (int i =1; i < logimg->width-1; i++)
	{
		for (int j = 1; j < logimg->height-1; j++)
		{
			s1 = cvGet2D(logimg,j,i);
			s2 = cvGet2D(oustimg,j,i);
			/*if (s1.val[0] > 0 && s2.val[0] == 0)
			{
				if(cvGet2D(logimg, j - 1, i - 1).val[0] > 0 && cvGet2D(logimg, j, i - 1).val[0] > 0
					&& cvGet2D(logimg, j + 1, i).val[0] > 0 && cvGet2D(logimg, j - 1, i).val[0] > 0 && cvGet2D(logimg, j + 1, i).val[0] > 0
					&& cvGet2D(logimg, j - 1, i + 1).val[0] > 0 && cvGet2D(logimg, j, i + 1).val[0] > 0 && cvGet2D(logimg, j + 1, i + 1).val[0] > 0) //logimgΪǰ����
		      	cvSet2D(thrsold, j, i, s1);
				else cvSet2D(thrsold, j, i, s2);
			}
			else if (s1.val[0] == 0 && s2.val[0] > 0)
			{ 
				if(cvGet2D(oustimg, j - 1, i - 1).val[0]>0 && cvGet2D(oustimg, j, i - 1).val[0]>0
				&& cvGet2D(oustimg, j + 1, i).val[0]>0 && cvGet2D(oustimg, j - 1, i).val[0]>0 && cvGet2D(oustimg, j + 1, i).val[0]>0
				&& cvGet2D(oustimg, j - 1, i + 1).val[0]>0 && cvGet2D(oustimg, j, i + 1).val[0]>0 && cvGet2D(oustimg, j + 1, i + 1).val[0]>0) //oustimgΪǰ����
				cvSet2D(thrsold, j, i, s2);
				else cvSet2D(thrsold, j, i, s1);
			}*/
			if ((s1.val[0] == 0 && s2.val[0] == 0 )|| (s1.val[0] > 0 && s2.val[0] > 0))
			{
				cvSet2D(thrsold, j, i, s1);
				//s3=cvGet2D(thrsold, j, i);
			}
		}
	}
	return thrsold;
}
int Region_Seg::Char_Seg(std::string filename, IplImage* img)
{
	//��ֵ�˲�  �ο���һ�ֻ��ڱ�Ե��Ϣ�ĳ����ַ��ָ��㷨
	cv::Mat src = cv::cvarrToMat(img);
	cv::Mat dst;
	cv::medianBlur(src, dst, 3);
	//blur(src, dst, Size(3, 3), Point(-1, -1));
	IplImage* img_gray1 = &IplImage(dst);
	//cvShowImage("src1", img_gray1);
	//ȥ�߿�
	cv::Mat img_3 = cv::cvarrToMat(img_gray1);
	borderCut(img_3, img_3);//ȥ�߿�
	cv::Point2f center = cv::Point2f(img_3.cols / 2, img_3.rows / 2);  // ��ת����
	cv::Mat rotateMat = getRotationMatrix2D(center, 180, 1);//��ת����
	cv::warpAffine(img_3, img_3, rotateMat, img_3.size());//�����ҵ�һ�����ⷢ�֣�Ŀǰû�ҵ�ԭ�򣬾�����Ҳ��֪��Ϊʲô�ھ�
	cv::warpAffine(img_3, img_3, rotateMat, img_3.size());//�����η�ת���������൱��û�иı�ʲô���󣬷ָ���ȷ������ˡ�											  //cv::imshow("�߿����", img_3);
	img = &IplImage(img_3);
//	cvShowImage("2",img);
	//cvWaitKey(0);
	//cout << img->width << " " << img->height << endl;
	
	IplImage*  dstimg = cvCreateImage(cv::Size(reWidth1, reHeight1), img->depth, img->nChannels);
	cvCopyMakeBorder(img, dstimg, cvPoint((reWidth1 - (img->width)) / 2, (reHeight1 - (img->height)) / 2), IPL_BORDER_CONSTANT);
	//cvCopyMakeBorder(img, dstimg, cvPoint(0, (reHeight1 - (img->height)) / 2), IPL_BORDER_CONSTANT);
	IplImage*  dstimg1 = cvCreateImage(cvGetSize(dstimg), dstimg->depth, dstimg->nChannels);
	cvCopy(dstimg, dstimg1);
	//cvShowImage("dstimg1", dstimg1);
	//cvSaveImage(".\\resource\\dstimg.jpg", dstimg);
	//��ʼ�ָ�
	CvSeq* contours = NULL;
	CvMemStorage* storage = cvCreateMemStorage(0);
	int count = cvFindContours(dstimg1, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL);
	//printf("����������%d", count);
	int all = 0;
	int t = 0;
	for (CvSeq* c = contours; c != NULL; c = c->h_next)
	{	CvRect rc = cvBoundingRect(c, 0);
		if (rc.height < 0.32*dstimg1->height||
			(rc.x>((double)(dstimg1->width/8))*7 && rc.width<(double)(dstimg1->width /12)))
		continue;
		dp[all].x = rc.x;
		dp[all].width = rc.width;
		if (rc.y - 2 < 0) {
			dp[all].y = rc.y;
			dp[all].height = dstimg1->height - rc.y - 1;
		}
		else {
			dp[all].y = rc.y - 2;
			dp[all].height = dstimg1->height - rc.y + 3;
		}
		all++;
	}
	cvReleaseMemStorage(&storage);
	sort(dp, dp + all, cmp);
	int ans = all;

	dp[0].y = 0 ;
	dp[0].height = dstimg1->height - dp[0].y -2;
	for (int i = 0; i < ans; i++)
	{
		//cout << dp[i].x << " " << dp[i].y << " " << dp[i].height << " " << dp[i].width << endl;
		if (dp[i].width < dstimg1->width *0.063) {
			dp[i].x = dp[i].x - 4;
			dp[i].width = dp[i].width + 8;
		}
		else {
			dp[i].x = dp[i].x;
			dp[i].width = dp[i].width + 4;
		}
	if (dp[i].x - 2 < 0 || dp[i].x < 0)
		{
			dp[i].x = 0;
		}
		else
			dp[i].x -= 2;
		if (dp[i].y - 2 < 0)
			dp[i].y = 0;
		else
			dp[i].y -= 2;
		if (dp[i].y + dp[i].height>dstimg->height)
			dp[i].height = dstimg->height- dp[i].y-1;
	}//dp[ans - 1].width -= 1;//���һ��������߽�
	////�����ļ���
	stringstream ss(filename);
	while (std::getline(ss, filename, '/')) //��/Ϊ����ָ�filename������,����filename���������õ��ľ��ǲ���·�����ļ���
		;
	stringstream ss1(filename);
	std::getline(ss1, filename, '.');//ȥ����׺
	CString str = ("dst//" + filename).c_str();
	if (!PathIsDirectory(str))//�����ļ��У����򴴽�
	{
		CreateDirectory(str, NULL);
	}
	cv::imwrite("./dst/" + filename + "/" + filename + ".jpg", img_3);
	//int  charHeight = 20, charWidth = 20;//Ҫ����ĵ����ַ�ͼ��size
	for (int i = 0; i <ans; i++)
	{
		if (i>0 && dp[i].x - (dp[i - 1].x + dp[i - 1].width) < 0)
			dp[i - 1].width = dp[i].x - dp[i - 1].x - 1;
		CvSize size = cvSize(dp[i].width, dp[i].height);//�����С
		cvSetImageROI(dstimg, cvRect(dp[i].x, dp[i].y, size.width, size.height));
		IplImage* pDest = cvCreateImage(size, dstimg->depth, dstimg->nChannels);//����Ŀ��ͼ��
		cvCopy(dstimg, pDest); //����ͼ��
		cvResetImageROI(pDest);
		char numchar[10];
		sprintf(numchar, "%d", i);
		cv::Mat Dest = cv::cvarrToMat(pDest);
		cv::imwrite("./dst/" + filename + "/" + numchar + ".jpg", Dest);
		cvReleaseImage(&pDest);
	}
	//cv::imwrite("./dst/" + filename + "/" + filename + ".jpg", img_3);
	return ans;
}
Region_Seg::~Region_Seg()
{
	cvDestroyAllWindows();
}
