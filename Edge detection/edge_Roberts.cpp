#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void getRobert_oper(cv::Mat& getRobert_oper1, cv::Mat& getRobert_oper2){
	//135�㷽��
	getRobert_oper1 = (cv::Mat_<float>(2, 2) << 1, 0, -1, 0);
	//45�㷽��
	getRobert_oper2 = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);

	//��ʱ�뷴ת180��õ�����ˣ����ﷴת֮����ԭ��һ����Ϊ���Ͻ����������������
	cv::flip(getRobert_oper1, getRobert_oper1, -1);
	cv::flip(getRobert_oper2, getRobert_oper2, -1);
}

void edge_Robert(cv::Mat& src, cv::Mat& dst1, cv::Mat& dst2, cv::Mat& dst,int ddepth, double delta = 0, int borderType = cv::BORDER_DEFAULT){
	//��ȡRobert����
	cv::Mat getRobert_oper1;
	cv::Mat getRobert_oper2;
	getRobert_oper(getRobert_oper1, getRobert_oper2);

	//����õ�135�㷽���Ե
	cv::filter2D(src, dst1, ddepth, getRobert_oper1, cv::Point(0, 0), delta, borderType);

	//����õ�45�㷽���Ե
	cv::filter2D(src, dst2, ddepth, getRobert_oper2, cv::Point(1, 0), delta, borderType);

	//��Եǿ�ȣ����ƣ�
	cv::convertScaleAbs(dst1, dst1); //�����ֵ��תΪ�޷���8λͼ
	cv::convertScaleAbs(dst2, dst2);
	dst = dst1 + dst2;
}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig1025(a)(building_original).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1) cv::cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat dst, dst1, dst2;

	//ע�⣺Ҫ����CV_32F����Ϊ��Щ�ط������Ϊ����������8λ�޷��ţ���ᵼ����Щ�ط�Ϊ0
	edge_Robert(src, dst1, dst2, dst, CV_32F); 

	imshow("src", src);
	imshow("135���Ե", dst1);
	imshow("45���Ե", dst2);
	imshow("��Եǿ��", dst);
	cv::waitKey(0);
	return 0;
}