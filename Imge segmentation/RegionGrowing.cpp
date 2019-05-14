#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


/***************************************************************************************
Function:  ���������㷨
Input:     src ������ԭͼ�� pt ��ʼ������ th ��������ֵ����
Output:    ��ʵ�ʵ����ڵ����� ʵ�����ǰ�ɫ�����������Ǻ�ɫ
Description: �������������Ϊ��ɫ(255),����ɫΪ��ɫ(0)
Return:    NULL
Others:    NULL
***************************************************************************************/
void RegionGrow(cv::Mat& src, cv::Mat& matDst, cv::Point2i pt, int th)
{
	cv::Point2i ptGrowing;						//��������λ��
	int nGrowLable = 0;								//����Ƿ�������
	int nSrcValue = 0;								//�������Ҷ�ֵ
	int nCurValue = 0;								//��ǰ������Ҷ�ֵ
	matDst = cv::Mat::zeros(src.size(), CV_8UC1);	//����һ���հ��������Ϊ��ɫ
	//��������˳������
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	std::vector<cv::Point2i> vcGrowPt;						//������ջ
	vcGrowPt.push_back(pt);							//��������ѹ��ջ��
	matDst.at<uchar>(pt.y, pt.x) = 255;				//���������
	nSrcValue = src.at<uchar>(pt.y, pt.x);			//��¼������ĻҶ�ֵ
	
	while (!vcGrowPt.empty())						//����ջ��Ϊ��������
	{
		pt = vcGrowPt.back();						//ȡ��һ��������
		vcGrowPt.pop_back();

		//�ֱ�԰˸������ϵĵ��������
		for (int i = 0; i<8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//����Ƿ��Ǳ�Ե��
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//��ǰ��������ĻҶ�ֵ

			if (nGrowLable == 0)					//�����ǵ㻹û�б�����
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) < th)					//����ֵ��Χ��������
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//���Ϊ��ɫ
					vcGrowPt.push_back(ptGrowing);					//����һ��������ѹ��ջ��
				}
			}
		}
	}

}


void on_MouseHandle(int event, int x, int y, int flags, void* param){
	cv::Mat& src = *(cv::Mat*) param;
	cv::Mat src_gray, dst;
	if (src.channels() > 1)
		cv::cvtColor(src, src_gray, CV_RGB2GRAY);
	cv::Point2i  pt;
	switch (event)
	{
		//�������
	case cv::EVENT_LBUTTONDOWN:
	{
		//x:�� y:��						   
		pt=cv::Point2i(x, y);
		std::cout <<"���ӵ�λ�ã�"<< "(x,y)=" << "(" << x << "," << y << ")" << std::endl;
	}
	break;
	   //����ſ�
	char str[16];
	case cv::EVENT_LBUTTONUP:
	{
		cv::circle(src, cv::Point2i(x, y),2, cv::Scalar(0, 0, 255), -1,CV_AA);
		sprintf_s(str, "(%d,%d)", x, y);
		//cv::putText(src, str, cv::Point2i(x, y), 3, 1, cv::Scalar(150, 200,0), 2, 8);
		cv::namedWindow("dst", CV_WINDOW_NORMAL);//����һ��dst���� 
		pt = cv::Point2i(x, y);
	    RegionGrow(src_gray,dst,pt,40);  //��������
		cv::bitwise_and(src_gray, dst, dst); //������
		imshow("src", src);
		imshow("dst", dst);
	}
		break;
	}


}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\lung2.jpeg");
	if (src.empty()){
		return -1;
	}
	cv::namedWindow("src", CV_WINDOW_NORMAL);//����һ��img����
	cv::setMouseCallback("src", on_MouseHandle, (void*)&src);//���ûص����� 
	imshow("src", src);
	cv::waitKey(0);
}