#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void getRobert_oper(cv::Mat& getRobert_oper1, cv::Mat& getRobert_oper2){
	//135°方向
	getRobert_oper1 = (cv::Mat_<float>(2, 2) << 1, 0, -1, 0);
	//45°方向
	getRobert_oper2 = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);

	//逆时针反转180°得到卷积核（这里反转之后与原来一样，为了严谨还是做这个操作）
	cv::flip(getRobert_oper1, getRobert_oper1, -1);
	cv::flip(getRobert_oper2, getRobert_oper2, -1);
}

void edge_Robert(cv::Mat& src, cv::Mat& dst1, cv::Mat& dst2, cv::Mat& dst,int ddepth, double delta = 0, int borderType = cv::BORDER_DEFAULT){
	//获取Robert算子
	cv::Mat getRobert_oper1;
	cv::Mat getRobert_oper2;
	getRobert_oper(getRobert_oper1, getRobert_oper2);

	//卷积得到135°方向边缘
	cv::filter2D(src, dst1, ddepth, getRobert_oper1, cv::Point(0, 0), delta, borderType);

	//卷积得到45°方向边缘
	cv::filter2D(src, dst2, ddepth, getRobert_oper2, cv::Point(1, 0), delta, borderType);

	//边缘强度（近似）
	cv::convertScaleAbs(dst1, dst1); //求绝对值并转为无符号8位图
	cv::convertScaleAbs(dst2, dst2);
	dst = dst1 + dst2;
}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig1025(a)(building_original).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1) cv::cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat dst, dst1, dst2;

	//注意：要采用CV_32F，因为有些地方卷积后为负数，若用8位无符号，则会导致这些地方为0
	edge_Robert(src, dst1, dst2, dst, CV_32F); 

	imshow("src", src);
	imshow("135°边缘", dst1);
	imshow("45°边缘", dst2);
	imshow("边缘强度", dst);
	cv::waitKey(0);
	return 0;
}