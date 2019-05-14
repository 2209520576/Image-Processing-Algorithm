#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

enum adaptiveMethod{meanFilter,gaaussianFilter,medianFilter};

void AdaptiveThreshold(cv::Mat& src, cv::Mat& dst, double Maxval, int Subsize, double c, adaptiveMethod method = meanFilter){

	if (src.channels() > 1)
		cv::cvtColor(src, src, CV_RGB2GRAY);

	cv::Mat smooth;
	switch (method)
	{
	case  meanFilter:
		cv::blur(src, smooth, cv::Size(Subsize, Subsize));
		break;
	case gaaussianFilter:
		cv::GaussianBlur(src, smooth, cv::Size(Subsize, Subsize),0,0);
		break;
	case medianFilter:
		cv::medianBlur(src, smooth, Subsize);
		break;
	default:
		break;
	}

	smooth = smooth - c;
	
	//ãÐÖµ´¦Àí
	src.copyTo(dst);
	for (int r = 0; r < src.rows;++r){
		const uchar* srcptr = src.ptr<uchar>(r);
		const uchar* smoothptr = smooth.ptr<uchar>(r);
		uchar* dstptr = dst.ptr<uchar>(r);
		for (int c = 0; c < src.cols; ++c){
			if (srcptr[c]>smoothptr[c]){
				dstptr[c] = Maxval;
			}
			else
				dstptr[c] = 0;
		}
	}

}

int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig1049(a)(spot_shaded_text_image).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1)
		cv::cvtColor(src, src, CV_RGB2GRAY);

	cv::Mat dst, dst2;
	double t2 = (double)cv::getTickCount();
	AdaptiveThreshold(src, dst, 255, 21, 10, meanFilter);  //
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "my_process=" << time2 << " ms. " << std::endl << std::endl;


	cv::adaptiveThreshold(src, dst2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 21, 10);


	cv::namedWindow("src", CV_WINDOW_NORMAL);
	cv::imshow("src", src);
	cv::namedWindow("dst", CV_WINDOW_NORMAL);
	cv::imshow("dst", dst);
	cv::namedWindow("dst2", CV_WINDOW_NORMAL);
	cv::imshow("dst2", dst2);
	//cv::imwrite("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Image Filtering\\MeanFilter\\TXT.jpg",dst);
	cv::waitKey(0);
}