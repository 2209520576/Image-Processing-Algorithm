#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

///////////////////////////////
//x，y方向联合实现获取高斯模板
//////////////////////////////
void generateGaussMask(cv::Mat& Mask, cv::Size wsize, double sigma){
	Mask.create(wsize, CV_64F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	double sum = 0.0;
	double x, y;
	for (int i = 0; i < h; ++i){
		y = pow(i - center_h, 2);
		for (int j = 0; j < w; ++j){
			x = pow(j - center_w, 2);
			//因为最后都要归一化的，常数部分可以不计算，也减少了运算量
			double g = exp(-(x + y) / (2 * sigma*sigma));
			Mask.at<double>(i, j) = g;
			sum += g;
		}
	}
	Mask = Mask / sum;
}

////////////////////////////
//按二维高斯函数实现高斯滤波
///////////////////////////
void GaussianFilter(cv::Mat& src, cv::Mat& dst, cv::Mat window){
	int hh = (window.rows - 1) / 2;
	int hw = (window.cols - 1) / 2;
	dst = cv::Mat::zeros(src.size(), src.type());
	//边界填充
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REPLICATE);//边界复制

	//高斯滤波
	for (int i = hh; i < src.rows + hh; ++i){
		for (int j = hw; j < src.cols + hw; ++j){
			double sum[3] = { 0 };

			for (int r = -hh; r <= hh; ++r){
				for (int c = -hw; c <= hw; ++c){
					if (src.channels() == 1){
						sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j + c) * window.at<double>(r + hh, c + hw);
					}
					else if (src.channels() == 3){
						cv::Vec3b rgb = Newsrc.at<cv::Vec3b>(i + r, j + c);
						sum[0] = sum[0] + rgb[0] * window.at<double>(r + hh, c + hw);//B
						sum[1] = sum[1] + rgb[1] * window.at<double>(r + hh, c + hw);//G
						sum[2] = sum[2] + rgb[2] * window.at<double>(r + hh, c + hw);//R
					}
				}
			}

			for (int k = 0; k < src.channels(); ++k){
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k]>255)
					sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(i - hh, j - hw) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3)
			{
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i - hh, j - hw) = rgb;
			}

		}
	}

}

////////////////////////////
//DOG高斯差分
///////////////////////////
void DOG1(cv::Mat &src, cv::Mat &dst, cv::Size wsize, double sigma,double k=1.6){
	cv::Mat Mask1, Mask2, gaussian_dst1, gaussian_dst2;
	generateGaussMask(Mask1, wsize, k*sigma);//获取二维高斯滤波模板1
	generateGaussMask(Mask2, wsize, sigma);//获取二维高斯滤波模板2
	
	//高斯滤波
	GaussianFilter(src, gaussian_dst1, Mask1);
	GaussianFilter(src, gaussian_dst2, Mask2);

	dst = gaussian_dst1 - gaussian_dst2-1;

	cv::threshold(dst, dst, 0, 255, cv::THRESH_BINARY);
}


////////////////////////////////////////
//DOG高斯差分--使用opencv的GaussianBlur
////////////////////////////////////////
void DOG2(cv::Mat &src, cv::Mat &dst, cv::Size wsize, double sigma, double k = 5){
	cv::Mat gaussian_dst1, gaussian_dst2;
	//高斯滤波
	cv::GaussianBlur(src, gaussian_dst1, wsize, k*sigma);
	cv::GaussianBlur(src, gaussian_dst2, wsize, sigma);

	dst = gaussian_dst1 - gaussian_dst2;
	cv::threshold(dst, dst, 0, 255, cv::THRESH_BINARY);
}

int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig1025(a)(building_original).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1) cv::cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat edge1,edge2;
	DOG1(src, edge1, cv::Size(7, 7), 0.8);
	DOG2(src, edge2, cv::Size(7, 7), 0.8);
	cv::namedWindow("src", CV_WINDOW_NORMAL);
	imshow("src", src);
	cv::namedWindow("My_DOG", CV_WINDOW_NORMAL);
	imshow("My_DOG", edge1);

	cv::namedWindow("Opencv_DOG", CV_WINDOW_NORMAL);
	imshow("Opencv_DOG", edge2);
	cv::waitKey(0);
	return 0;
}