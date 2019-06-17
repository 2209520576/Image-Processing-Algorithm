#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat RGB2GRAY(cv::Mat src, bool accelerate=false){
	CV_Assert(src.channels()==3);
	cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);
	cv::Vec3b rgb;
	int r = src.rows;
	int c = src.cols;
	
	  for (int i = 0; i < r; ++i){
		 for (int j = 0; j < c; ++j){
			rgb = src.at<cv::Vec3b>(i, j);
			uchar B = rgb[0]; uchar G = rgb[1]; uchar R = rgb[2];
			if (accelerate = false){
				dst.at<uchar>(i, j) = R*0.299 + G*0.587 + B*0.114;
			}
			else{
				dst.at<uchar>(i, j) = (R * 4898 + G * 9618 + B * 1868) >> 14;
			}
		 }
	   }
	return dst;
}

int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\lena.jpg");

	if (src.empty()){
		return -1;
	}
	cv::Mat dst,dst1;

	//opencv自带
	double t2 = (double)cv::getTickCount(); //测时间
	cv::cvtColor(src, dst1, CV_RGB2GRAY);
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "Opencv_rgb2gray=" << time2 << " ms. " << std::endl << std::endl;

	//RGB2GRAY
	double t1 = (double)cv::getTickCount(); //测时间
	dst = RGB2GRAY(src, true);
	t1 = (double)cv::getTickCount() - t1;
	double time1 = (t1 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "My_rgb2gray=" << time1 << " ms. " << std::endl << std::endl;


	cv::namedWindow("src", CV_WINDOW_NORMAL);
	imshow("src", src);
	cv::namedWindow("My_rgb2gray", CV_WINDOW_NORMAL);
	imshow("My_rgb2gray", dst);
	cv::namedWindow("Opencv_rgb2gray", CV_WINDOW_NORMAL);
	imshow("Opencv_rgb2gray", dst1);
	cv::waitKey(0);
	return 0;

}