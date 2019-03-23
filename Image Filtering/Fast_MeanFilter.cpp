#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>

//////////////////////////////////////
//积分图-常规方法 
//由三个由三个位置的积分计算出来
//对于W*H图像：3*(W-1)*(H-1)次加减法
//////////////////////////////////////
void Im_integral(cv::Mat& src,cv::Mat& dst){
	int nr = src.rows;
	int nc = src.cols;
    dst = cv::Mat::zeros(nr + 1, nc + 1, CV_64F);
	for (int i = 1; i < dst.rows; ++i){
		for (int j = 1; j < dst.cols; ++j){
			double top_left = dst.at<double>(i - 1, j - 1);
			double top_right = dst.at<double>(i - 1, j);
			double buttom_left = dst.at<double>(i, j - 1);
			int buttom_right = src.at<uchar>(i-1, j-1);
			dst.at<double>(i, j) = buttom_right + buttom_left + top_right - top_left;
		}
	}
}

//////////////////////////////////////////
//积分图-优化方法
//由上方negral(i-1,j)加上当前行的和即可
//对于W*H图像：2*(W-1)*(H-1)次加减法
//比常规方法快1.5倍左右
/////////////////////////////////////////
void Fast_integral(cv::Mat& src, cv::Mat& dst){
	int nr = src.rows;
	int nc = src.cols;
	int sum_r = 0;
	dst = cv::Mat::zeros(nr + 1, nc + 1, CV_64F);
	for (int i = 1; i < dst.rows; ++i){  
		for (int j = 1, sum_r = 0; j < dst.cols; ++j){
			sum_r = src.at<uchar>(i-1 , j-1) + sum_r; //行累加
			dst.at<double>(i, j) = dst.at<double>(i-1, j)+sum_r;
		}
	}
}


void Fast_MeanFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize){

	//图像边界扩充
	if (wsize.height % 2 == 0 || wsize.width % 2 == 0){
		fprintf(stderr, "Please enter odd size!");
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//以边缘为轴，对称
	dst = cv::Mat::zeros(src.size(), src.type());
	
	//计算积分图
	cv::Mat inte;
	Fast_integral(Newsrc, inte);

	//均值滤波
	int mean = 0;
	for (int i = hh+1; i < src.rows + hh + 1;++i){  //积分图图像比原图（边界扩充后的）多一行和一列 
		for (int j = hw+1; j < src.cols + hw + 1; ++j){
			double top_left = inte.at<double>(i - hh - 1, j - hw-1);
			double top_right = inte.at<double>(i-hh-1,j+hw);
			double buttom_left = inte.at<double>(i + hh, j - hw- 1);
			double buttom_right = inte.at<double>(i+hh,j+hw);
			mean = (buttom_right - top_right - buttom_left + top_left) / wsize.area();
			dst.at<uchar>(i - hh-1, j - hw-1) = mean;
		}
	}

}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\28.jpg");
	if (src.empty()){
		return -1;
	}
	if (src.channels()>1)
	cvtColor(src, src, CV_RGB2GRAY);

	cv::Mat dst;
	double t2 = (double)cv::getTickCount(); //测时间
	Fast_MeanFilter(src, dst, cv::Size(15,15));//均值滤波
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "FASTmy_process=" << time2 << " ms. " << std::endl << std::endl;

	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::waitKey(0);

}