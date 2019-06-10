#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void getPrewitt_oper(cv::Mat& getPrewitt_horizontal, cv::Mat& getPrewitt_vertical, cv::Mat& getPrewitt_Diagonal1,cv::Mat& getPrewitt_Diagonal2){
	//ˮƽ����
	getPrewitt_horizontal = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	//��ֱ����
	getPrewitt_vertical = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	//�Խ�135��
	getPrewitt_Diagonal1 = (cv::Mat_<float>(3, 3) << 0, 1, 1, -1, 0, 1, -1, -1, 0);
	//�Խ�45��
	getPrewitt_Diagonal2 = (cv::Mat_<float>(3, 3) << -1, -1, 0, -1, 0, 1, 0, 1, 1);

	//��ʱ�뷴ת180��õ������
	cv::flip(getPrewitt_horizontal, getPrewitt_horizontal, -1);
	cv::flip(getPrewitt_vertical, getPrewitt_vertical, -1);
	cv::flip(getPrewitt_Diagonal1, getPrewitt_Diagonal1, -1);
	cv::flip(getPrewitt_Diagonal2, getPrewitt_Diagonal2, -1);
}

void edge_Prewitt(cv::Mat& src, cv::Mat& dst1, cv::Mat& dst2, cv::Mat& dst3, cv::Mat& dst4, cv::Mat& dst, int ddepth, double delta = 0, int borderType = cv::BORDER_DEFAULT){
	//��ȡPrewitt����
	cv::Mat getPrewitt_horizontal;
	cv::Mat getPrewitt_vertical;
	cv::Mat getPrewitt_Diagonal1;
	cv::Mat getPrewitt_Diagonal2;
	getPrewitt_oper(getPrewitt_horizontal, getPrewitt_vertical, getPrewitt_Diagonal1, getPrewitt_Diagonal2);

	//����õ�ˮƽ�����Ե
	cv::filter2D(src, dst1, ddepth, getPrewitt_horizontal, cv::Point(-1, -1), delta, borderType);

	//����õ�4��ֱ�����Ե
	cv::filter2D(src, dst2, ddepth, getPrewitt_vertical, cv::Point(-1, -1), delta, borderType);

	//����õ�45�㷽���Ե
	cv::filter2D(src, dst3, ddepth, getPrewitt_Diagonal1, cv::Point(-1, -1), delta, borderType);

	//����õ�135�㷽���Ե
	cv::filter2D(src, dst4, ddepth, getPrewitt_Diagonal2, cv::Point(-1, -1), delta, borderType);

	//��Եǿ�ȣ����ƣ�
	cv::convertScaleAbs(dst1, dst1); //�����ֵ��תΪ�޷���8λͼ
	cv::convertScaleAbs(dst2, dst2);





	cv::convertScaleAbs(dst3, dst3); //�����ֵ��תΪ�޷���8λͼ
	cv::convertScaleAbs(dst4, dst4);
	dst = dst1 + dst2 ;
	//std::cout << dst4 << std::endl;
}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\(embedded_square_noisy_512).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1) cv::cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat dst, dst1, dst2, dst3, dst4;

	//ע�⣺Ҫ����CV_32F����Ϊ��Щ�ط������Ϊ����������8λ�޷��ţ���ᵼ����Щ�ط�Ϊ0
	edge_Prewitt(src, dst1, dst2, dst3, dst4, dst, CV_32F);

	cv::namedWindow("src", CV_WINDOW_NORMAL);
	imshow("src", src);
	cv::namedWindow("ˮƽ��Ե", CV_WINDOW_NORMAL);
	imshow("ˮƽ��Ե", dst1);
	cv::namedWindow("��ֱ��Ե", CV_WINDOW_NORMAL);
	imshow("��ֱ��Ե", dst2);
	cv::namedWindow("45���Ե", CV_WINDOW_NORMAL);
	imshow("45���Ե", dst3);
	cv::namedWindow("135���Ե", CV_WINDOW_NORMAL);
	imshow("135���Ե", dst4);
	cv::namedWindow("��Եǿ��", CV_WINDOW_NORMAL);
	imshow("��Եǿ��", dst);
	cv::waitKey(0);
	return 0;
}