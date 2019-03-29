#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

///////////////////////////////
//x��y��������ʵ�ֻ�ȡ��˹ģ��
//////////////////////////////
void generateGaussMask(cv::Mat& Mask,cv::Size wsize, double sigma){
	Mask.create(wsize,CV_64F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	double sum = 0.0;
	double x, y;
	for (int i = 0; i < h; ++i){
		y = pow(i - center_h, 2);
		for (int j = 0; j < w; ++j){
			x = pow(j - center_w, 2);
			//��Ϊ���Ҫ��һ���ģ��������ֿ��Բ����㣬Ҳ������������
			double g = exp(-(x + y) / (2 * sigma*sigma));
			Mask.at<double>(i, j) = g;
			sum += g;
		}
	}
	Mask = Mask / sum;
}

////////////////////////////
//����ά��˹����ʵ�ָ�˹�˲�
///////////////////////////
void GaussianFilter(cv::Mat& src, cv::Mat& dst, cv::Mat window){
	int hh = (window.rows - 1) / 2;
	int hw = (window.cols - 1) / 2;
	dst = cv::Mat::zeros(src.size(),src.type());
	//�߽����
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REPLICATE);//�߽縴��
	
	//��˹�˲�
	for (int i = hh; i < src.rows + hh;++i){
		for (int j = hw; j < src.cols + hw; ++j){
			double sum[3] = { 0 };

			for (int r = -hh; r <= hh; ++r){
				for (int c = -hw; c <= hw; ++c){
					if (src.channels() == 1){
						sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j + c) * window.at<double>(r + hh, c + hw);
					}
					else if (src.channels() == 3){
						cv::Vec3b rgb = Newsrc.at<cv::Vec3b>(i+r,j + c);
						sum[0] = sum[0] + rgb[0] * window.at<double>(r + hh, c + hw);//B
						sum[1] = sum[1] + rgb[1] * window.at<double>(r + hh, c + hw);//G
						sum[2] = sum[2] + rgb[2] * window.at<double>(r + hh, c + hw);//R
					}
				}
			}

			for (int k = 0; k < src.channels(); ++k){
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k]>255)
					sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(i - hh, j - hw) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3)
			{
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i-hh, j-hw) = rgb;
			}

		}
	}

}


////////////////////////////////////
//�������ʵ�ָ�˹�˲������Ӹ�Ч
///////////////////////////////////
void separateGaussianFilter(cv::Mat& src, cv::Mat& dst, int wsize, double sigma){
	//��ȡһά��˹�˲�ģ��
	cv::Mat window;
	window.create(1 , wsize, CV_64F);
	int center = (wsize - 1) / 2;
	double sum = 0.0;
	for (int i = 0; i < wsize; ++i){
		double g = exp(-(pow(i - center, 2)) / (2 * sigma*sigma));
		window.at<double>(0, i) = g;
		sum += g;
	}
	window = window / sum;
	//std::cout << window << std::endl;

	//�߽����
	int boder = (wsize - 1) / 2;
	dst = cv::Mat::zeros(src.size(), src.type());
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, boder, boder, boder, boder, cv::BORDER_REPLICATE);//�߽縴��

	//��˹�˲�--ˮƽ����
	for (int i = boder; i < src.rows + boder; ++i){
		for (int j = boder; j < src.cols + boder; ++j){
			double sum[3] = { 0 };

			for (int r = -boder; r <= boder; ++r){
					if (src.channels() == 1){
						sum[0] = sum[0] + Newsrc.at<uchar>(i, j + r) * window.at<double>(0, r + boder); //�в����б�
					}
					else if (src.channels() == 3){
						cv::Vec3b rgb = Newsrc.at<cv::Vec3b>(i, j +r);
						sum[0] = sum[0] + rgb[0] * window.at<double>(0, r + boder);//B
						sum[1] = sum[1] + rgb[1] * window.at<double>(0, r + boder);//G
						sum[2] = sum[2] + rgb[2] * window.at<double>(0, r + boder);//R
				}
			}
			for (int k = 0; k < src.channels(); ++k){
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k]>255)
					sum[k] = 255;
			}
			if (src.channels() == 1){
				dst.at<uchar>(i - boder, j - boder) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3){
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i - boder, j - boder) = rgb;
			}
		}
	}

	//��˹�˲�--��ֱ����
	//��ˮƽ��������dst�߽����
	cv::copyMakeBorder(dst, Newsrc, boder, boder, boder, boder, cv::BORDER_REPLICATE);//�߽縴��
	for (int i = boder; i < src.rows + boder; ++i){
		for (int j = boder; j < src.cols + boder; ++j){
			double sum[3] = { 0 };

			for (int r = -boder; r <= boder; ++r){
					if (src.channels() == 1){
						sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j ) * window.at<double>(0, r + boder); //�в����б�
					}
					else if (src.channels() == 3){
						cv::Vec3b rgb = Newsrc.at<cv::Vec3b>(i + r, j);
						sum[0] = sum[0] + rgb[0] * window.at<double>(0, r + boder);//B
						sum[1] = sum[1] + rgb[1] * window.at<double>(0, r + boder);//G
						sum[2] = sum[2] + rgb[2] * window.at<double>(0, r + boder);//R
					}
				}
			for (int k = 0; k < src.channels(); ++k){
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k]>255)
					sum[k] = 255;
			}
			if (src.channels() == 1){
				dst.at<uchar>(i - boder, j - boder) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3){
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i - boder, j - boder) = rgb;
			}
	   	}	
	}
 }



int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\lena.jpg");
	if (src.empty()){
		return -1;
	}
	cv::Mat Mask;
	cv::Mat dst1;
	cv::Mat dst2;

	//����ʵ��
	generateGaussMask(Mask, cv::Size(5, 5), 10);//��ȡ��ά��˹�˲�ģ��
	//std::cout << Mask << std::endl;
	GaussianFilter(src, dst1, Mask);

	//����ʵ��
	separateGaussianFilter( src, dst2, 5, 10);
	
	
	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("����ʵ��");
	cv::imshow("����ʵ��", dst1);
	cv::namedWindow("����ʵ��");
	cv::imshow("����ʵ��", dst2);
//	cv::imwrite("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Image Filtering\\GaussianFilter\\woman.jpg", dst);
	cv::waitKey(0);
	return 0;
}