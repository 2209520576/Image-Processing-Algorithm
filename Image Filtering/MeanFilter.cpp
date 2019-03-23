#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void MeanFilater(cv::Mat& src,cv::Mat& dst,cv::Size wsize){
	//图像边界扩充:窗口的半径
	if (wsize.height % 2 == 0 || wsize.width % 2 == 0){
		fprintf(stderr,"Please enter odd size!" );
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//以边缘为轴，对称
	dst=cv::Mat::zeros(src.size(),src.type());

    //均值滤波
	int sum = 0;
	int mean = 0;
	for (int i = hh; i < src.rows + hh; ++i){
		for (int j = hw; j < src.cols + hw;++j){

			for (int r = i - hh; r <= i + hh; ++r){
				for (int c = j - hw; c <= j + hw;++c){
					sum = Newsrc.at<uchar>(r, c) + sum;
				}
			}
			mean = sum / (wsize.area());
			dst.at<uchar>(i-hh,j-hw)=mean;
			sum = 0;
			mean = 0;
		}
	}

}

int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\28.jpg");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1)
		cv::cvtColor(src,src,CV_RGB2GRAY);

	cv::Mat dst;
	cv::Mat dst1;
	cv::Size wsize(15,15);

	double t2 = (double)cv::getTickCount();
	MeanFilater(src, dst, wsize); //均值滤波
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "FASTmy_process=" << time2 << " ms. " << std::endl << std::endl;

	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::imwrite("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Image Filtering\\MeanFilter\\woman.jpg",dst);
	cv::waitKey(0);
}