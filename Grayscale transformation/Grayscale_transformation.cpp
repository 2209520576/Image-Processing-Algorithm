#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/*图像反转*（用指针访问像素）*/
void Image_inversion(cv::Mat& src, cv::Mat& dst){
	int nr = src.rows;
	int nc = src.cols*src.channels(); //列数 * 通道数= 每行元素的个数
	src.copyTo(dst);
	if (src.isContinuous() && dst.isContinuous()){  //判断图像连续性
		nr = 1;
		nc = src.rows*src.cols*src.channels(); //行数*列数 * 通道数= 一维数组的个数
	}

	for (int i = 0; i < nr; i++){
		   const uchar* srcdata = src.ptr <uchar>(i);  //采用指针访问像素，获取第i行的首地址
		   uchar* dstdata = dst.ptr <uchar>(i);
			for (int j = 0; j < nc; j++){
			dstdata[j] = 255 - srcdata[j]; //开始处理每个像素
		}
	}
}

/*对数变换方法1*(灰度图像和彩色图像都适用)*/
void LogTransform1(cv::Mat& src, cv::Mat& dst, double c){
	int nr = src.rows;
	int nc = src.cols*src.channels();
	src.copyTo(dst);
	dst.convertTo(dst, CV_64F);
	if (src.isContinuous() && dst.isContinuous()){  //判断图像连续性
		nr = 1;
		nc = src.rows*src.cols*src.channels(); //行数*列数 * 通道数= 一维数组的个数
	}

	for (int i = 0; i < nr; i++){
		const uchar* srcdata = src.ptr <uchar>(i);  //采用指针访问像素，获取第i行的首地址
		double* dstdata = dst.ptr <double>(i);
		for (int j = 0; j < nc; j++){
			dstdata[j] = c*log(double(1.0 + srcdata[j])); //开始处理每个像素
		}
	}
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX); //经过对比拉升（将像素值归一化到0-255）得到最终的图像
	dst.convertTo(dst, CV_8U);  //转回无符号8位图像
}

/*对数变换方法2*(适用于灰度图像)*/
cv::Mat LogTransform2(cv::Mat& src , double c){
	if (src.channels()>1)
		cv::cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat dst;
	src.copyTo(dst);
	dst.convertTo(dst,CV_64F);
	dst = dst + 1.0;
	cv::log(dst,dst);
	dst =c*dst;
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX); //经过对比拉升（将像素值归一化到0-255）得到最终的图像
	dst.convertTo(dst, CV_8U);  //转回无符号8位图像
	return dst;

}


/*分段线性变换——对比度拉伸*/
/*****************
三段线性变换
a<=b,c<=d
*****************/
void contrast_stretching(cv::Mat& src, cv::Mat& dst, double a, double b, double c, double d){
	src.copyTo(dst);
	dst.convertTo(dst, CV_64F);
	double min = 0, max = 0;
	cv::minMaxLoc(dst, &min, &max, 0, 0);
	int nr = dst.rows;
	int nc = dst.cols *dst.channels();
	if (dst.isContinuous ()){
		int nr = 1;
		int nc = dst.cols  * dst.rows * dst.channels();
	}
	for (int i = 0; i < nr; i++){
		double* ptr_dst = dst.ptr<double>(i);
		for (int j = 0; j < nc; j++){
			if (min <= ptr_dst[j] < a)
				ptr_dst[j] = (c / a)*ptr_dst[j];
			else if(a <= ptr_dst[j] < b)
				ptr_dst[j] = ((d-c)/(b-a))*ptr_dst[j];
			else if (b <= ptr_dst[j] < max )
				ptr_dst[j] = ((max - d) / (max - b))*ptr_dst[j];
		}
	}
	dst.convertTo(dst, CV_8U);  //转回无符号8位图像
}


/*bit平面分层*/
/*十进制数转二进制数*/
void num2Binary(int num, int b[8]){ 
	int i;
	for ( i = 0; i < 8; i++){
		b[i] = 0;
	}
	i = 0;
	while (num!=0){
		b[i] = num % 2;
		num = num / 2;
		i++;
	}
}

/***************************
num_bit - 指定bit平面
num_bit = 1~8
num_bit=1，即输出第1个Bit平面
****************************/
void Bitplane_stratification(cv::Mat& src,cv::Mat& B , int num_Bit){
	int b[8];//8个二进制bit位
	if (src.channels()>1)
		cv::cvtColor(src, src, CV_RGB2GRAY);
	B.create(src.size(), src.type()); 
	for (int i = 0; i < src.rows; i++){
		const uchar* ptr_src = src.ptr<uchar>(i);
		uchar* ptr_B = B.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++){
			num2Binary(ptr_src[j], b);
			ptr_B[j] = b[num_Bit - 1]*255;  //0和1灰度差别太小，乘255便于视觉观察
		}
	}
}



int main(){
	cv::Mat src =cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\(2nd_from_top).tif");
	if (src.empty()){
		return -1;
	}
	cv::Mat dst;
	//Image_inversion(src, dst); //图像反转
	//LogTransform1(src, dst, 10); //对数变换1
    //dst = LogTransform2(src,10); //对数变换2
	//contrast_stretching(src, dst, 20, 100, 30, 200); //分段函数变换
	Bitplane_stratification(src, dst, 8); //Bit平面分层

	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::waitKey(0);
}

