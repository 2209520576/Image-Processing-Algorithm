#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//�׳�
int factorial(int n){
	int fac = 1;
	//0�Ľ׳�
	if (n == 0)
		return fac;
	for (int i = 1; i <= n; ++i){
		fac *= i;
	}
	return fac;
}

//���Sobelƽ������
cv::Mat getSobelSmoooth(int wsize){
	int n = wsize - 1;
	cv::Mat SobelSmooothoper=cv::Mat::zeros(cv::Size(wsize,1),CV_32FC1);
	for (int k = 0; k <= n; k++){
		float *pt = SobelSmooothoper.ptr<float>(0);
		pt[k] = factorial(n) / (factorial(k)*factorial(n - k));
	}
	return SobelSmooothoper;
}

//���Sobel�������
cv::Mat getSobeldiff(int wsize){
	cv::Mat Sobeldiffoper = cv::Mat::zeros(cv::Size(wsize, 1), CV_32FC1);
	cv::Mat SobelSmoooth = getSobelSmoooth(wsize - 1);
	for (int k = 0; k < wsize; k++){
		if (k == 0)  
			Sobeldiffoper.at<float>(0, k) = 1;
		else if (k == wsize - 1)
			Sobeldiffoper.at<float>(0, k) = -1;
		else
			Sobeldiffoper.at<float>(0, k) = SobelSmoooth.at<float>(0, k) - SobelSmoooth.at<float>(0, k - 1);
	}      
	return Sobeldiffoper;
}

//���ʵ��
void conv2D(cv::Mat& src, cv::Mat& dst, cv::Mat kernel, int ddepth, cv::Point anchor = cv::Point(-1, -1), int delta = 0, int borderType = cv::BORDER_DEFAULT){
	cv::Mat  kernelFlip;
	cv::flip(kernel, kernelFlip, -1);
	cv::filter2D(src, dst, ddepth, kernelFlip, anchor, delta, borderType);
}


//�ɷ������������ȴ�ֱ����������ˮƽ������
void sepConv2D_Y_X(cv::Mat& src, cv::Mat& dst, cv::Mat kernel_Y, cv::Mat kernel_X, int ddepth, cv::Point anchor = cv::Point(-1, -1), int delta = 0, int borderType = cv::BORDER_DEFAULT){
	cv::Mat dst_kernel_Y;
	conv2D(src, dst_kernel_Y, kernel_Y, ddepth, anchor, delta, borderType); //��ֱ������
	conv2D(dst_kernel_Y, dst, kernel_X, ddepth, anchor, delta, borderType); //ˮƽ������
}

//�ɷ�������������ˮƽ����������ֱ������
void sepConv2D_X_Y(cv::Mat& src, cv::Mat& dst, cv::Mat kernel_X, cv::Mat kernel_Y, int ddepth, cv::Point anchor = cv::Point(-1, -1), int delta = 0, int borderType = cv::BORDER_DEFAULT){
	cv::Mat dst_kernel_X;
	conv2D(src, dst_kernel_X, kernel_X, ddepth, anchor, delta, borderType); //ˮƽ������
	conv2D(dst_kernel_X, dst, kernel_Y, ddepth, anchor, delta, borderType); //��ֱ������
}


//Sobel���ӱ�Ե���
//dst_X ��ֱ����
//dst_Y ˮƽ����
void Sobel(cv::Mat& src, cv::Mat& dst_X, cv::Mat& dst_Y, cv::Mat& dst, int wsize, int ddepth, cv::Point anchor = cv::Point(-1, -1), int delta = 0, int borderType = cv::BORDER_DEFAULT){

	cv::Mat SobelSmooothoper = getSobelSmoooth(wsize); //ƽ��ϵ��
	cv::Mat Sobeldiffoper = getSobeldiff(wsize); //���ϵ��

	//�ɷ������������ȴ�ֱ����ƽ������ˮƽ�����֡����õ���ֱ��Ե
	sepConv2D_Y_X(src, dst_X, SobelSmooothoper.t(), Sobeldiffoper, ddepth);

	//�ɷ�������������ˮƽ����ƽ������ֱ�����֡����õ�ˮƽ��Ե
	sepConv2D_X_Y(src, dst_Y, SobelSmooothoper, Sobeldiffoper.t(), ddepth);

	//��Եǿ�ȣ����ƣ�
	dst = abs(dst_X) + abs(dst_Y);
	cv::convertScaleAbs(dst, dst); //�����ֵ��תΪ�޷���8λͼ
	cv::convertScaleAbs(dst_X, dst_X); //�����ֵ��תΪ�޷���8λͼ
	cv::convertScaleAbs(dst_Y, dst_Y);
	//cv::pow(dst_X, 2.0, dst_X);
	//cv::pow(dst_Y, 2.0, dst_Y);
	//cv::sqrt(dst_X + dst_Y, dst); 
	//dst.convertTo(dst, CV_8UC1);
}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig1025(a)(building_original).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1) cv::cvtColor(src, src, CV_RGB2GRAY);
	int wsize = 3;
	cv::Mat Sobeldiffoper, SobelSmooothoper;
	SobelSmooothoper = getSobelSmoooth(wsize);
	Sobeldiffoper = getSobeldiff(wsize);

	//ע�⣺Ҫ����CV_32F����Ϊ��������Ϊ����������8λ�޷��ţ���ᵼ����Щ�ط�Ϊ0
	cv::Mat dst, dst_X, dst_Y;
	Sobel(src, dst_X, dst_Y, dst, wsize, CV_32FC1);
	cv::namedWindow("src", CV_WINDOW_NORMAL);
	imshow("src", src);
	cv::namedWindow("ˮƽ��Ե", CV_WINDOW_NORMAL);
	imshow("ˮƽ��Ե", dst_Y);
	cv::namedWindow("��ֱ��Ե", CV_WINDOW_NORMAL);
	imshow("��ֱ��Ե", dst_X);
	//cv::namedWindow("��Եǿ��", CV_WINDOW_NORMAL);
	imshow("��Եǿ��", dst);
	cv::namedWindow("��Եǿ��ȡ��-Ǧ������", CV_WINDOW_NORMAL);
	imshow("��Եǿ��ȡ��-Ǧ������",255-dst);

	std::cout <<"SobelSmooothoper: "<< SobelSmooothoper << std::endl;
	std::cout << "Sobeldiffoper: " << Sobeldiffoper << std::endl;
	cv::waitKey(0);
	return 0;
}
