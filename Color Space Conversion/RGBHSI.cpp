#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;


Mat RGB2HSI(Mat src){
	int row = src.rows;
	int col = src.cols;
	Mat dsthsi(row, col, CV_64FC3);
	Mat H = Mat(row, col, CV_64FC1);
	Mat S = Mat(row, col, CV_64FC1);
	Mat I = Mat(row, col, CV_64FC1);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			double h, s, newi, th;
			double B = (double)src.at<Vec3b>(i, j)[0] / 255.0;
			double G = (double)src.at<Vec3b>(i, j)[1] / 255.0;
			double R = (double)src.at<Vec3b>(i, j)[2] / 255.0;
			double mi, mx;
			if (R > G && R > B){
				mx = R;
				mi = min(G, B);
			}
			else{
				if (G > B){
					mx = G;
					mi = min(R, B);
				}
				else{
					mx = B;
					mi = min(R, G);
				}
			}
			newi = (R + G + B) / 3.0;
			if (newi < 0)  newi = 0;
			else if (newi > 1) newi = 1.0;
			if (newi == 0 || mx == mi){
				s = 0;
				h = 0;
			}
			else{
				s = 1 - mi / newi;
				th = (R - G) * (R - G) + (R - B) * (G - B);
				th = sqrt(th) + 1e-5;
				th = acos(((R - G + R - B)*0.5) / th);
				if (G >= B) h = th;
				else h = 2 * CV_PI - th;
			}
			h = h / (2 * CV_PI);
			H.at<double>(i, j) = h;
			S.at<double>(i, j) = s;
			I.at<double>(i, j) = newi;

			dsthsi.at<Vec3d>(i, j)[0] = h;
			dsthsi.at<Vec3d>(i, j)[1] = s;
			dsthsi.at<Vec3d>(i, j)[2] = newi;

		}
	}
	return dsthsi;
}

Mat HSI2RGB(Mat src){
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_64FC3);

	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			double preh = src.at<Vec3d>(i, j)[0] * 2 * CV_PI;//H
			double pres = src.at<Vec3d>(i, j)[1];  //S
			double prei = src.at<Vec3d>(i, j)[2];  //I
			double r = 0, g = 0, b = 0;
			double t1, t2, t3;
			t1 = (1.0 - pres) / 3.0;
			if (preh >= 0 && preh < (CV_PI * 2 / 3)){
				b = t1;
				t2 = pres * cos(preh);
				t3 = cos(CV_PI / 3 - preh);
				r = (1 + t2 / t3) / 3;
				r = 3 * prei * r;
				b = 3 * prei * b;
				g = 3 * prei - (r + b);
			}
			else if (preh >= (CV_PI * 2 / 3) && preh < (CV_PI * 4 / 3)){
				r = t1;
				t2 = pres * cos(preh - 2 * CV_PI / 3);
				t3 = cos(CV_PI - preh);
				g = (1 + t2 / t3) / 3;
				r = 3 * prei * r;
				g = 3 * g * prei;
				b = 3 * prei - (r + g);
			}
			else if (preh >= (CV_PI * 4 / 3) && preh <= (CV_PI * 2)){
				g = t1;
				t2 = pres * cos(preh - 4 * CV_PI / 3);
				t3 = cos(CV_PI * 5 / 3 - preh);
				b = (1 + t2 / t3) / 3;
				g = 3 * g * prei;
				b = 3 * prei * b;
				r = 3 * prei - (g + b);
			}
			dst.at<Vec3d>(i, j)[0] = b;
			dst.at<Vec3d>(i, j)[1] = g;
			dst.at<Vec3d>(i, j)[2] = r;
		}
	}
	return dst;
}


int main(){
	cv::Mat src = cv::imread("I:/Learning-and-Practice/2019Change/Image process algorithm/Img/002.jpg");

	if (src.empty()){
		return -1;
	}
	cv::Mat dst, dst2;

	//////////RGB2HSI//////////
	double t1 = (double)cv::getTickCount(); //≤‚ ±º‰

	dst = RGB2HSI(src); //RGB2HSI
	dst2 = HSI2RGB(dst); //HSI2BGR
	//std::cout << dst << std::endl;

	t1 = (double)cv::getTickCount() - t1;
	double time1 = (t1 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "My_RGB2HSI=" << time1 << " ms. " << std::endl << std::endl;


	cv::namedWindow("src", CV_WINDOW_NORMAL);
	imshow("src", src);
	cv::namedWindow("My_RGB2HSI", CV_WINDOW_NORMAL);
	imshow("My_RGB2HSI", dst);
	cv::namedWindow("My_HSI2RGB", CV_WINDOW_NORMAL);
	imshow("My_HSI2RGB", dst2);
	cv::waitKey(0);
	return 0;

}