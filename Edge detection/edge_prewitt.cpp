#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void getPrewitt_oper(cv::Mat& getPrewitt_horizontal, cv::Mat& getPrewitt_vertical, cv::Mat& getPrewitt_Diagonal1,cv::Mat& getPrewitt_Diagonal2){
	//水平方向
	getPrewitt_horizontal = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	//垂直方向
	getPrewitt_vertical = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	//对角135°
	getPrewitt_Diagonal1 = (cv::Mat_<float>(3, 3) << 0, 1, 1, -1, 0, 1, -1, -1, 0);
	//对角45°
	getPrewitt_Diagonal2 = (cv::Mat_<float>(3, 3) << -1, -1, 0, -1, 0, 1, 0, 1, 1);

	//逆时针反转180°得到卷积核
	cv::flip(getPrewitt_horizontal, getPrewitt_horizontal, -1);
	cv::flip(getPrewitt_vertical, getPrewitt_vertical, -1);
	cv::flip(getPrewitt_Diagonal1, getPrewitt_Diagonal1, -1);
	cv::flip(getPrewitt_Diagonal2, getPrewitt_Diagonal2, -1);
}

void edge_Prewitt(cv::Mat& src, cv::Mat& dst1, cv::Mat& dst2, cv::Mat& dst3, cv::Mat& dst4, cv::Mat& dst, int ddepth, double delta = 0, int borderType = cv::BORDER_DEFAULT){
	//获取Prewitt算子
	cv::Mat getPrewitt_horizontal;
	cv::Mat getPrewitt_vertical;
	cv::Mat getPrewitt_Diagonal1;
	cv::Mat getPrewitt_Diagonal2;
	getPrewitt_oper(getPrewitt_horizontal, getPrewitt_vertical, getPrewitt_Diagonal1, getPrewitt_Diagonal2);

	//卷积得到水平方向边缘
	cv::filter2D(src, dst1, ddepth, getPrewitt_horizontal, cv::Point(-1, -1), delta, borderType);

	//卷积得到4垂直方向边缘
	cv::filter2D(src, dst2, ddepth, getPrewitt_vertical, cv::Point(-1, -1), delta, borderType);

	//卷积得到45°方向边缘
	cv::filter2D(src, dst3, ddepth, getPrewitt_Diagonal1, cv::Point(-1, -1), delta, borderType);

	//卷积得到135°方向边缘
	cv::filter2D(src, dst4, ddepth, getPrewitt_Diagonal2, cv::Point(-1, -1), delta, borderType);

	//边缘强度（近似）
	cv::convertScaleAbs(dst1, dst1); //求绝对值并转为无符号8位图
	cv::convertScaleAbs(dst2, dst2);





	cv::convertScaleAbs(dst3, dst3); //求绝对值并转为无符号8位图
	cv::convertScaleAbs(dst4, dst4);
	dst = dst1 + dst2 ;
	//std::cout << dst4 << std::endl;
}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\(embedded_square_noisy_512).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1) cv::cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat dst, dst1, dst2, dst3, dst4;

	//注意：要采用CV_32F，因为有些地方卷积后为负数，若用8位无符号，则会导致这些地方为0
	edge_Prewitt(src, dst1, dst2, dst3, dst4, dst, CV_32F);

	cv::namedWindow("src", CV_WINDOW_NORMAL);
	imshow("src", src);
	cv::namedWindow("水平边缘", CV_WINDOW_NORMAL);
	imshow("水平边缘", dst1);
	cv::namedWindow("垂直边缘", CV_WINDOW_NORMAL);
	imshow("垂直边缘", dst2);
	cv::namedWindow("45°边缘", CV_WINDOW_NORMAL);
	imshow("45°边缘", dst3);
	cv::namedWindow("135°边缘", CV_WINDOW_NORMAL);
	imshow("135°边缘", dst4);
	cv::namedWindow("边缘强度", CV_WINDOW_NORMAL);
	imshow("边缘强度", dst);
	cv::waitKey(0);
	return 0;
}