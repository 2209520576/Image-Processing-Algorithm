#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/*ͼ��ת*����ָ��������أ�*/
void Image_inversion(cv::Mat& src, cv::Mat& dst){
	int nr = src.rows;
	int nc = src.cols*src.channels(); //���� * ͨ����= ÿ��Ԫ�صĸ���
	src.copyTo(dst);
	if (src.isContinuous() && dst.isContinuous()){  //�ж�ͼ��������
		nr = 1;
		nc = src.rows*src.cols*src.channels(); //����*���� * ͨ����= һά����ĸ���
	}

	for (int i = 0; i < nr; i++){
		   const uchar* srcdata = src.ptr <uchar>(i);  //����ָ��������أ���ȡ��i�е��׵�ַ
		   uchar* dstdata = dst.ptr <uchar>(i);
			for (int j = 0; j < nc; j++){
			dstdata[j] = 255 - srcdata[j]; //��ʼ����ÿ������
		}
	}
}

/*�����任����1*(�Ҷ�ͼ��Ͳ�ɫͼ������)*/
void LogTransform1(cv::Mat& src, cv::Mat& dst, double c){
	int nr = src.rows;
	int nc = src.cols*src.channels();
	src.copyTo(dst);
	dst.convertTo(dst, CV_64F);
	if (src.isContinuous() && dst.isContinuous()){  //�ж�ͼ��������
		nr = 1;
		nc = src.rows*src.cols*src.channels(); //����*���� * ͨ����= һά����ĸ���
	}

	for (int i = 0; i < nr; i++){
		const uchar* srcdata = src.ptr <uchar>(i);  //����ָ��������أ���ȡ��i�е��׵�ַ
		double* dstdata = dst.ptr <double>(i);
		for (int j = 0; j < nc; j++){
			dstdata[j] = c*log(double(1.0 + srcdata[j])); //��ʼ����ÿ������
		}
	}
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX); //�����Ա�������������ֵ��һ����0-255���õ����յ�ͼ��
	dst.convertTo(dst, CV_8U);  //ת���޷���8λͼ��
}

/*�����任����2*(�����ڻҶ�ͼ��)*/
cv::Mat LogTransform2(cv::Mat& src , double c){
	if (src.channels()>1)
		cv::cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat dst;
	src.copyTo(dst);
	dst.convertTo(dst,CV_64F);
	dst = dst + 1.0;
	cv::log(dst,dst);
	dst =c*dst;
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX); //�����Ա�������������ֵ��һ����0-255���õ����յ�ͼ��
	dst.convertTo(dst, CV_8U);  //ת���޷���8λͼ��
	return dst;

}


/*�ֶ����Ա任�����Աȶ�����*/
/*****************
�������Ա任
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
	dst.convertTo(dst, CV_8U);  //ת���޷���8λͼ��
}


/*bitƽ��ֲ�*/
/*ʮ������ת��������*/
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
num_bit - ָ��bitƽ��
num_bit = 1~8
num_bit=1���������1��Bitƽ��
****************************/
void Bitplane_stratification(cv::Mat& src,cv::Mat& B , int num_Bit){
	int b[8];//8��������bitλ
	if (src.channels()>1)
		cv::cvtColor(src, src, CV_RGB2GRAY);
	B.create(src.size(), src.type()); 
	for (int i = 0; i < src.rows; i++){
		const uchar* ptr_src = src.ptr<uchar>(i);
		uchar* ptr_B = B.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++){
			num2Binary(ptr_src[j], b);
			ptr_B[j] = b[num_Bit - 1]*255;  //0��1�ҶȲ��̫С����255�����Ӿ��۲�
		}
	}
}



int main(){
	cv::Mat src =cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\(2nd_from_top).tif");
	if (src.empty()){
		return -1;
	}
	cv::Mat dst;
	//Image_inversion(src, dst); //ͼ��ת
	//LogTransform1(src, dst, 10); //�����任1
    //dst = LogTransform2(src,10); //�����任2
	//contrast_stretching(src, dst, 20, 100, 30, 200); //�ֶκ����任
	Bitplane_stratification(src, dst, 8); //Bitƽ��ֲ�

	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::waitKey(0);
}

