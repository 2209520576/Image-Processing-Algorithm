#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>

//////////////////////////////////////
//����ͼ-���淽�� 
//������������λ�õĻ��ּ������
//����W*Hͼ��3*(W-1)*(H-1)�μӼ���
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
//����ͼ-�Ż�����
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
			sum_r = src.at<uchar>(i-1 , j-1) + sum_r; 
			dst.at<double>(i, j) = dst.at<double>(i-1, j)+sum_r;
		}
	}
}


//����ͼ���پ�ֵ�˲�
void Fast_MeanFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize){

	//ͼ��߽�����
	if (wsize.height % 2 == 0 || wsize.width % 2 == 0){
		fprintf(stderr, "Please enter odd size!");
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//�Ա�ԵΪ�ᣬ�Գ�
	dst = cv::Mat::zeros(src.size(), src.type());
	
	//�������ͼ
	cv::Mat inte;
	Fast_integral(Newsrc, inte);

	//��ֵ�˲�
	double mean = 0;
	for (int i = hh+1; i < src.rows + hh + 1;++i){  //����ͼͼ���ԭͼ���߽������ģ���һ�к�һ�� 
		for (int j = hw+1; j < src.cols + hw + 1; ++j){
			double top_left = inte.at<double>(i - hh - 1, j - hw-1);
			double top_right = inte.at<double>(i-hh-1,j+hw);
			double buttom_left = inte.at<double>(i + hh, j - hw- 1);
			double buttom_right = inte.at<double>(i+hh,j+hw);
			mean = (buttom_right - top_right - buttom_left + top_left) / wsize.area();
			
			//һ��Ҫ�����жϺ���������ת��
			if (mean < 0)
				mean = 0;
			else if (mean>255)
				mean = 255;
			dst.at<uchar>(i - hh - 1, j - hw - 1) = static_cast<uchar> (mean);
		}
	}

}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig0334(a)(hubble-original).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels()>1)
	cvtColor(src, src, CV_RGB2GRAY);

	cv::Mat dst;
	double t2 = (double)cv::getTickCount(); //��ʱ��
	Fast_MeanFilter(src, dst, cv::Size(151,151));//��ֵ�˲�
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "FASTmy_process=" << time2 << " ms. " << std::endl << std::endl;

	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::waitKey(0);

}