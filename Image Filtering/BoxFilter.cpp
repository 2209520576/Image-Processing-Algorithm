#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


/////////////////////////////////////////
//�����ͼ-�Ż�����
//���Ϸ�negral(i-1,j)���ϵ�ǰ�еĺͼ���
//����W*Hͼ��2*(W-1)*(H-1)�μӼ���
//�ȳ��淽����1.5������
/////////////////////////////////////////
void Fast_integral(cv::Mat& src, cv::Mat& dst){
	int nr = src.rows;
	int nc = src.cols;
	int sum_r = 0;
	dst = cv::Mat::zeros(nr + 1, nc + 1, CV_64F);
	for (int i = 1; i < dst.rows; ++i){
		for (int j = 1, sum_r = 0; j < dst.cols; ++j){
			//���ۼӣ���Ϊ����ͼ�൱����ԭͼ�Ϸ���һ�У���߼�һ�У����Ի���ͼ��(1,1)��Ӧԭͼ(0,0),(i,j)��Ӧ(i-1,j-1)
			sum_r = src.at<uchar>(i - 1, j - 1) + sum_r; //���ۼ�
			dst.at<double>(i, j) = dst.at<double>(i - 1, j) + sum_r;
		}
	}
}


//////////////////////////////////
//�����˲�-��ֵ�˲������������
/////////////////////////////////
void BoxFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize, bool normalize){

	//ͼ��߽�����
	if (wsize.height % 2 == 0 || wsize.width % 2 == 0){
		fprintf(stderr, "Please enter odd size!");
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_CONSTANT);//�Ա�ԵΪ�ᣬ�Գ�
	dst = cv::Mat::zeros(src.size(), src.type());

	//�������ͼ
	cv::Mat inte;
	Fast_integral(Newsrc, inte);

	//BoxFilter
	double mean = 0;
	for (int i = hh + 1; i < src.rows + hh + 1; ++i){  //����ͼͼ���ԭͼ���߽������ģ���һ�к�һ�� 
		for (int j = hw + 1; j < src.cols + hw + 1; ++j){
			double top_left = inte.at<double>(i - hh - 1, j - hw - 1);
			double top_right = inte.at<double>(i - hh - 1, j + hw);
			double buttom_left = inte.at<double>(i + hh, j - hw - 1);
			double buttom_right = inte.at<double>(i + hh, j + hw);
			if (normalize == true)
				mean = (buttom_right - top_right - buttom_left + top_left) / wsize.area();
			else
				mean = buttom_right - top_right - buttom_left + top_left;
			
			//һ��Ҫ�����жϺ���������ת��
			if (mean < 0)
				mean = 0;
			else if (mean>255)
				mean = 255;
			dst.at<uchar>(i - hh - 1, j - hw - 1) = static_cast<uchar>(mean);

		}
	}

}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\lena.jpg");
	if (src.empty()){
		return -1;
	}
	if (src.channels()>1)
		cvtColor(src, src, CV_RGB2GRAY);

	//�Ա�BoxFilter����
	cv::Mat dst1;
	double t2 = (double)cv::getTickCount(); //��ʱ��
	BoxFilter(src, dst1, cv::Size(9, 9), true);//�����˲�
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "FASTmy_process=" << time2 << " ms. " << std::endl << std::endl;
	int ss = dst1.at<uchar>(10, 0);
	std::cout << ss << std::endl<<std::endl;

	//opencv�Դ�BoxFilter����
	cv::Mat dst2;
	double t1 = (double)cv::getTickCount(); //��ʱ��
	cv::boxFilter(src, dst2, -1, cv::Size(9, 9), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);//�����˲�
	t1 = (double)cv::getTickCount() - t1;
	double time1 = (t1 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "FASTmy_process=" << time1 << " ms. " << std::endl << std::endl;
	 ss = dst2.at<uchar>(10, 0);
	std::cout <<ss << std::endl;

	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst1");
	cv::imshow("dst1", dst1);
	cv::namedWindow("dst2");
	cv::imshow("dst2", dst2);
	cv::waitKey(0);

}
