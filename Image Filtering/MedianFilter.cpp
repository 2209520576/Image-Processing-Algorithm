#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

///////////////////////////
///�����㷨-ð������(�Ľ���)
///////////////////////////
void bublle_sort(std::vector<int> &arr){
	bool flag=true;
	for (int i = 0; i < arr.size() - 1; ++i){
		while (flag){
			flag = false;
			for (int j = 0; j < arr.size() - 1; ++j){
				if (arr[j]>arr[j + 1]){
					int tmp = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = tmp;
					flag = true;
				}
			}
		}
	}
}


////////////////////////
//��ֵ�˲�
///////////////////////
void MedianFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize){
    //ͼ��߽�����
	if (wsize.width % 2 == 0 || wsize.height % 2 == 0){
		fprintf(stderr, "Please enter odd size!");
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//�Ա�ԵΪ�ᣬ�Գ�
	dst = cv::Mat::zeros(src.rows, src.cols, src.type());

	//��ֵ�˲�
	std::vector<int> pix(wsize.area());
	for (int i = hh; i < src.rows + hh; ++i){
		uchar* ptrdst = dst.ptr(i - hh);
		for (int j = hw; j < src.cols + hw; ++j){
			
			for (int r = i - hh; r <= i + hh; ++r){
				const uchar* ptrsrc = Newsrc.ptr(r);
				for (int c = j - hw; c <= j + hw; ++c){
					pix.push_back(ptrsrc[c]);
				}
			}
			bublle_sort(pix);//ð������

			ptrdst[j - hw] = pix[(wsize.area()-1)/2];//��ָӳ�䵽���ͼ��
			pix = {0};
		}
	}
}

int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\salt.tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1)
		cv::cvtColor(src, src, CV_RGB2GRAY);

	cv::Mat dst;
	cv::Mat dst1;
	cv::Size wsize(5 , 5);

	double t2 = (double)cv::getTickCount();
	MedianFilter(src, dst, wsize); //��ֵ�˲�
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "my_process=" << time2 << " ms. " << std::endl << std::endl;

	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	//cv::imwrite("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Image Filtering\\MedianFilter\\salt.jpg",dst);
	cv::waitKey(0);
}