#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int Max_Entropy(cv::Mat& src, cv::Mat& dst, int thresh, int p){
	const int Grayscale = 256;
	int Graynum[Grayscale] = { 0 };
	int r = src.rows;
	int c = src.cols;
	for (int i = 0; i < r; ++i){
		const uchar* ptr = src.ptr<uchar>(i);   
		for (int j = 0; j < c; ++j){   
			if (ptr[j] == 0)				//排除掉黑色的像素点
				continue;
			Graynum[ptr[j]]++;
		}
	}

	float probability = 0.0; //概率
	float max_Entropy = 0.0; //最大熵
	int totalpix = r*c;
	for (int i = 0; i < Grayscale; ++i){

		float HO = 0.0; //前景熵
		float HB = 0.0; //背景熵

	    //计算前景像素数
		int frontpix = 0;
		for (int j = 0; j < i; ++j){
			frontpix += Graynum[j];
		}
		//计算前景熵
		for (int j = 0; j < i; ++j){
			if (Graynum[j] != 0){
				probability = (float)Graynum[j] / frontpix;
				HO = HO + probability*log(1/probability);
			}
		}

		//计算背景熵
		for (int k = i; k < Grayscale; ++k){
			if (Graynum[k] != 0){
				probability = (float)Graynum[k] / (totalpix - frontpix);
				HB = HB + probability*log(1/probability);
			}
		}

		//计算最大熵
		if(HO + HB > max_Entropy){
			max_Entropy = HO + HB;
			thresh = i + p; 
		}
	}

	//阈值处理
	src.copyTo(dst);
	for (int i = 0; i < r; ++i){
		uchar* ptr = dst.ptr<uchar>(i);
		for (int j = 0; j < c; ++j){
			if (ptr[j]> thresh)
				ptr[j] = 255;
			else
				ptr[j] = 0;
		}
	}
	return thresh;
}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig0943(a)(dark_blobs_on_light_background).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1)
		cv::cvtColor(src, src, CV_RGB2GRAY);

	cv::Mat dst, dst2;
	int thresh = 0;
	double t2 = (double)cv::getTickCount();
	thresh = Max_Entropy(src, dst, thresh,-20); //Max_Entropy
	std::cout << "Mythresh=" << thresh << std::endl;
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "my_process=" << time2 << " ms. " << std::endl << std::endl;

	double  Otsu = 0;
	Otsu = cv::threshold(src, dst2, Otsu, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	std::cout << "Otsuthresh=" << Otsu << std::endl;
	

	cv::namedWindow("src", CV_WINDOW_NORMAL);
	cv::imshow("src", src);
	cv::namedWindow("dst", CV_WINDOW_NORMAL);
	cv::imshow("dst", dst);
	cv::namedWindow("dst2", CV_WINDOW_NORMAL);
	cv::imshow("dst2", dst2);
	//cv::imwrite("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Image Filtering\\MeanFilter\\TXT.jpg",dst);
	cv::waitKey(0);
}