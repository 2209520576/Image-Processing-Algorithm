#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
////////////////////sobel����/////////////////////////
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
	cv::Mat SobelSmooothoper = cv::Mat::zeros(cv::Size(wsize, 1), CV_32FC1);
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
}


//ȷ��һ����������Ƿ���ͼ����
bool checkInRang(int r,int c, int rows, int cols){
	if (r >= 0 && r < rows && c >= 0 && c < cols)
		return true;
	else
		return false;
}

//��ȷ����Ե��������ӳ���Ե
void trace(cv::Mat &edgeMag_noMaxsup, cv::Mat &edge, float TL,int r,int c,int rows,int cols){
	if (edge.at<uchar>(r, c) == 0){
		edge.at<uchar>(r, c) = 255;
		for (int i = -1; i <= 1; ++i){
			for (int j = -1; j <= 1; ++j){
				float mag = edgeMag_noMaxsup.at<float>(r + i, c + j);
				if (checkInRang(r + i, c + j, rows, cols) && mag >= TL)
					trace(edgeMag_noMaxsup, edge, TL, r + i, c + j, rows, cols);
			}
		}
	}
}

//Canny��Ե���
void Edge_Canny(cv::Mat &src, cv::Mat &edge, float TL, float TH, int wsize=3, bool L2graydient = false){
	int rows = src.rows;
	int cols = src.cols;

	//��˹�˲�
	cv::GaussianBlur(src,src,cv::Size(5,5),0.8);
	//sobel����
	cv::Mat dx, dy, sobel_dst;
	Sobel(src, dx, dy, sobel_dst, wsize, CV_32FC1);

	//�����ݶȷ�ֵ
	cv::Mat edgeMag;
	if (L2graydient = false)  edgeMag = abs(dx) + abs(dy); //����ֵ֮�ͽ���
	else if (L2graydient = true)  cv::magnitude(dx, dy, edgeMag); //��ƽ��

	//�����ݶȷ��� �Լ� �Ǽ���ֵ����
	cv::Mat edgeMag_noMaxsup = cv::Mat::zeros(rows, cols, CV_32FC1);
	for (int r = 1; r < rows - 1; ++r){
		for (int c = 1; c < cols - 1; ++c){
			float x = dx.at<float>(r, c);
			float y = dy.at<float>(r, c);
			float angle = std::atan2f(y, x) / CV_PI * 180; //��ǰλ���ݶȷ���
			float mag = edgeMag.at<float>(r, c);  //��ǰλ���ݶȷ�ֵ

			//�Ǽ���ֵ����
			//��ֱ��Ե--�ݶȷ���Ϊˮƽ����-3*3���������ҷ���Ƚ�
			if (abs(angle)<22.5 || abs(angle)>157.5){
				float left = edgeMag.at<float>(r, c - 1);
				float right = edgeMag.at<float>(r, c + 1);
				if (mag >= left && mag >= right)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}
		
			//ˮƽ��Ե--�ݶȷ���Ϊ��ֱ����-3*3���������·���Ƚ�
			if ((angle>=67.5 && angle<=112.5 ) || (angle>=-112.5 && angle<=-67.5)){
				float top = edgeMag.at<float>(r-1, c);
				float down = edgeMag.at<float>(r+1, c);
				if (mag >= top && mag >= down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}

			//+45���Ե--�ݶȷ���Ϊ����������-3*3�������������·���Ƚ�
			if ((angle>112.5 && angle<=157.5) || (angle>-67.5 && angle<=-22.5)){
				float right_top = edgeMag.at<float>(r - 1, c+1);
				float left_down = edgeMag.at<float>(r + 1, c-1);
				if (mag >= right_top && mag >= left_down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}


			//+135���Ե--�ݶȷ���Ϊ����������-3*3�������������Ϸ���Ƚ�
			if ((angle >=22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)){
				float left_top = edgeMag.at<float>(r - 1, c - 1);
				float right_down = edgeMag.at<float>(r + 1, c + 1);
				if (mag >= left_top && mag >= right_down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}
		}
	}

	//˫��ֵ������Ե����
	edge = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (int r = 1; r < rows - 1; ++r){
		for (int c = 1; c < cols - 1; ++c){
			float mag = edgeMag_noMaxsup.at<float>(r, c);
			//���ڸ���ֵ��Ϊȷ����Ե��
			if (mag >= TH)
				trace(edgeMag_noMaxsup, edge, TL, r, c, rows, cols);
			else if (mag < TL)
				edge.at<uchar>(r, c) = 0;
		}
	}
}

int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig1025(a)(building_original).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1) cv::cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat edge,dst;

	//Canny
	Edge_Canny(src, edge, 50,128);

	//opencv�Դ�Canny
	cv::Canny(src, dst, 50, 150);

	cv::namedWindow("src", CV_WINDOW_NORMAL);
	imshow("src", src);
	cv::namedWindow("My_canny", CV_WINDOW_NORMAL);
	imshow("My_canny", edge);
	cv::namedWindow("Opencv_canny", CV_WINDOW_NORMAL);
	imshow("Opencv_canny", dst);
	cv::waitKey(0);
	return 0;
}