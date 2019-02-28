#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

void gamma_correction(cv::Mat& src, cv::Mat& dst, float K){
	uchar LUT[256];
	src.copyTo(dst);
	for (int i = 0; i < 256; i++){
		//float f = (i + 0.5f) / 255;
		float f = i  / 255.0;
		f = pow(f, K);
		//LUT[i] = cv::saturate_cast<uchar>(f*255.0f-0.5f);
		LUT[i] = cv::saturate_cast<uchar>(f*255.0);
	}
	
	if (dst.channels() == 1){
		cv::MatIterator_<uchar> it = dst.begin<uchar>();
		cv::MatIterator_<uchar> it_end = dst.end<uchar>();
		for (; it != it_end; ++it){
			*it = LUT[(*it)];
		}
	}
	else{
		cv::MatIterator_<cv::Vec3b> it = dst.begin<cv::Vec3b>();
		cv::MatIterator_<cv::Vec3b> it_end = dst.end<cv::Vec3b>();
		for (; it != it_end; ++it){
			(*it)[0] = LUT[(*it)[0]];
			(*it)[1] = LUT[(*it)[1]];
			(*it)[2] = LUT[(*it)[2]];
		}
	}

}

int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\(washed_out_aerial_image).tif");
	if (src.empty()){
		return -1;
	}
	cv::Mat dst;
	gamma_correction(src, dst, 3); //gamma±ä»»

	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::waitKey(0);
}
