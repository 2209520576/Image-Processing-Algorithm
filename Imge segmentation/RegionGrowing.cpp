#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


/***************************************************************************************
Function:  区域生长算法
Input:     src 待处理原图像 pt 初始生长点 th 生长的阈值条件
Output:    肺实质的所在的区域 实质区是白色，其他区域是黑色
Description: 生长结果区域标记为白色(255),背景色为黑色(0)
Return:    NULL
Others:    NULL
***************************************************************************************/
void RegionGrow(cv::Mat& src, cv::Mat& matDst, cv::Point2i pt, int th)
{
	cv::Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0;								//标记是否生长过
	int nSrcValue = 0;								//生长起点灰度值
	int nCurValue = 0;								//当前生长点灰度值
	matDst = cv::Mat::zeros(src.size(), CV_8UC1);	//创建一个空白区域，填充为黑色
	//生长方向顺序数据
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	std::vector<cv::Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push_back(pt);							//将生长点压入栈中
	matDst.at<uchar>(pt.y, pt.x) = 255;				//标记生长点
	nSrcValue = src.at<uchar>(pt.y, pt.x);			//记录生长点的灰度值
	
	while (!vcGrowPt.empty())						//生长栈不为空则生长
	{
		pt = vcGrowPt.back();						//取出一个生长点
		vcGrowPt.pop_back();

		//分别对八个方向上的点进行生长
		for (int i = 0; i<8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值

			if (nGrowLable == 0)					//如果标记点还没有被生长
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) < th)					//在阈值范围内则生长
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//标记为白色
					vcGrowPt.push_back(ptGrowing);					//将下一个生长点压入栈中
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
		//左键按下
	case cv::EVENT_LBUTTONDOWN:
	{
		//x:列 y:行						   
		pt=cv::Point2i(x, y);
		std::cout <<"种子点位置："<< "(x,y)=" << "(" << x << "," << y << ")" << std::endl;
	}
	break;
	   //左键放开
	char str[16];
	case cv::EVENT_LBUTTONUP:
	{
		cv::circle(src, cv::Point2i(x, y),2, cv::Scalar(0, 0, 255), -1,CV_AA);
		sprintf_s(str, "(%d,%d)", x, y);
		//cv::putText(src, str, cv::Point2i(x, y), 3, 1, cv::Scalar(150, 200,0), 2, 8);
		cv::namedWindow("dst", CV_WINDOW_NORMAL);//定义一个dst窗口 
		pt = cv::Point2i(x, y);
	    RegionGrow(src_gray,dst,pt,40);  //区域生长
		cv::bitwise_and(src_gray, dst, dst); //与运算
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
	cv::namedWindow("src", CV_WINDOW_NORMAL);//定义一个img窗口
	cv::setMouseCallback("src", on_MouseHandle, (void*)&src);//调用回调函数 
	imshow("src", src);
	cv::waitKey(0);
}