#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>


/*图像旋转（以图像中心为旋转中心）*/
void affine_trans_rotate(cv::Mat& src, cv::Mat& dst, double Angle){
	double angle = Angle*CV_PI / 180.0;
	//构造输出图像
	int dst_rows = round(fabs(src.rows * cos(angle)) + fabs(src.cols * sin(angle)));//图像高度
	int dst_cols = round(fabs(src.cols * cos(angle)) + fabs(src.rows * sin(angle)));//图像宽度

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	} 
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}

	cv::Mat T1 = (cv::Mat_<double>(3,3) << 1.0,0.0,0.0 , 0.0,-1.0,0.0, -0.5*src.cols , 0.5*src.rows , 1.0); // 将原图像坐标映射到数学笛卡尔坐标
	cv::Mat T2 = (cv::Mat_<double>(3,3) << cos(angle),-sin(angle),0.0 , sin(angle), cos(angle),0.0, 0.0,0.0,1.0); //数学笛卡尔坐标下顺时针旋转的变换矩阵
	double t3[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 }, { 0.5*dst.cols, 0.5*dst.rows ,1.0} }; // 将数学笛卡尔坐标映射到旋转后的图像坐标
	cv::Mat T3 = cv::Mat(3.0,3.0,CV_64FC1,t3);
	cv::Mat T = T1*T2*T3;
	cv::Mat T_inv = T.inv(); // 求逆矩阵

	for (double i = 0.0; i < dst.rows; i++){
		for (double j = 0.0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1.0);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高

			/*双线性插值*/
			// 判断是否越界
			if (int(Angle) % 90 == 0) {
				if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
				if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; //必须要加上，否则会出现边界问题
			}

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top ; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}

/*平移变换*（以图像左顶点为原点）/
/****************************************
tx: 水平平移距离 正数向右移动 负数向左移动
ty: 垂直平移距离 正数向下移动 负数向上移动
*****************************************/
void affine_trans_translation(cv::Mat& src, cv::Mat& dst, double tx, double ty){
	//构造输出图像
	int dst_rows = src.rows;//图像高度
	int dst_cols = src.cols;//图像宽度

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}

	cv::Mat T = (cv::Mat_<double>(3, 3) << 1,0,0 , 0,1,0 , tx,ty,1); //平移变换矩阵
	cv::Mat T_inv = T.inv(); // 求逆矩阵

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高

			/*双线性插值*/
			// 判断是否越界

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}


/*尺度变换*（以图像左顶点为原点）/
/***************
cx: 水平缩放尺度
cy: 垂直缩放尺度
***************/
void affine_trans_scale(cv::Mat& src, cv::Mat& dst, double cx, double cy){
	//构造输出图像
	int dst_rows = round(cy*src.rows);//图像高度
	int dst_cols = round(cx*src.cols);//图像宽度

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}

	cv::Mat T = (cv::Mat_<double>(3, 3) <<cx,0,0, 0,cy,0 ,0,0,1 ); //尺度变换矩阵
	cv::Mat T_inv = T.inv(); // 求逆矩阵

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高

			/*双线性插值*/
			// 判断是否越界
			if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
			if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; 

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}

/*偏移变换*（以图像中心为偏移中心）/
/***************************************
sx: 水平偏移尺度 正数向左偏移 负数向右偏移
sy: 垂直偏移尺度 正数向上偏移 负数向下偏移
****************************************/
void affine_trans_deviation(cv::Mat& src, cv::Mat& dst, double sx, double sy){
	//构造输出图像
	int dst_rows = fabs(sy)*src.cols + src.rows;//图像高度
	int dst_cols = fabs(sx)*src.rows + src.cols;//图像宽度

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}

	cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, -0.5*src.cols, 0.5*src.rows, 1); // 将原图像坐标映射到数学笛卡尔坐标
	cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1,sy,0, sx,1,0, 0,0,1); //数学笛卡尔坐标偏移变换矩阵
	double t3[3][3] = { { 1, 0, 0 }, { 0, -1, 0 }, { 0.5*dst.cols, 0.5*dst.rows, 1 } }; // 将数学笛卡尔坐标映射到旋转后的图像坐标
	cv::Mat T3 = cv::Mat(3, 3, CV_64FC1, t3);
	cv::Mat T = T1*T2*T3;
	cv::Mat T_inv = T.inv(); // 求逆矩阵

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高

			/*双线性插值*/
			// 判断是否越界

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}

/*组合变换*/
/*变换顺序：缩放->旋转—>偏移*/
void affine_trans_comb(cv::Mat& src, cv::Mat& dst, double cx, double cy, double Angle, double sx, double sy){
	double angle = Angle*CV_PI / 180;
	//构造输出图像
	int dst_s_rows = round(cy*src.rows);//尺度变换后图像高度
	int dst_s_cols = round(cx*src.cols);//尺度变换后图像宽度

	int dst_sr_rows = round(fabs(dst_s_rows * cos(angle)) + fabs(dst_s_cols * sin(angle)));//再经过旋转后图像高度
	int dst_sr_cols = round(fabs(dst_s_cols * cos(angle)) + fabs(dst_s_rows * sin(angle)));//再经过旋转后图像宽度

	int dst_srd_rows = fabs(sy)*dst_sr_cols + dst_sr_rows;//最后经过偏移后图像高度
    int dst_srd_cols = fabs(sx)*dst_sr_rows + dst_sr_cols;//最后经过偏移后图像宽度


	if (src.channels() == 1) {

		dst = cv::Mat::zeros(dst_srd_rows, dst_srd_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_srd_rows, dst_srd_cols, CV_8UC3); //RGB图初始
	}

	cv::Mat T1 = (cv::Mat_<double>(3, 3) << cx, 0, 0, 0, cy, 0, 0, 0, 1); //尺度变换矩阵

	cv::Mat T21 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, -0.5*dst_s_cols, 0.5*dst_s_rows, 1); // 将尺度变换后的图像坐标映射到数学笛卡尔坐标
	cv::Mat T22 = (cv::Mat_<double>(3, 3) << cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1); //数学笛卡尔坐标下顺时针旋转的变换矩阵
	cv::Mat T2 = T21*T22;// 这里不需要转回图像坐标了，因为下面的偏移变换是在笛卡尔坐标下进行的

	cv::Mat T32 = (cv::Mat_<double>(3, 3) << 1, sy, 0, sx, 1, 0, 0, 0, 1); //数学笛卡尔坐标偏移变换矩阵
	double t33[3][3] = { { 1, 0, 0 }, { 0, -1, 0 }, { 0.5*dst.cols, 0.5*dst.rows, 1 } }; // 将数学笛卡尔坐标映射到偏移后的图像坐标
	cv::Mat T33 = cv::Mat(3, 3, CV_64FC1, t33);
	cv::Mat T3 = T32*T33;

	cv::Mat T = T1*T2*T3; //矩阵相乘按照组合变换的顺序
	cv::Mat T_inv = T.inv(); // 求逆矩阵

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高

			/*双线性插值*/
			// 判断是否越界
			if (int(Angle) % 90 == 0) {
				if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
				if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; //必须要加上，否则会出现边界问题
			}

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}


int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\5.bmp");
	cvtColor(src, src, CV_BGR2GRAY);
	if (src.empty()){
		std::cout << "Failure to load image..." << std::endl;
		return -1;
	}
	cv::Mat dst;
	double angle =250;  //旋转角度
	double tx = 50, ty = -50; //平移距离
	double cx =1.5, cy = 1.5; //缩放尺度
	double sx = 0.2, sy =0.2; //偏移尺度

	//affine_trans_rotate(src, dst, angle); //旋转

	//affine_trans_translation(src, dst, tx,ty); //平移

	//affine_trans_scale(src, dst, cx, cy);  //尺度缩放

    //affine_trans_deviation(src, dst, sx, sy); //偏移

	affine_trans_comb(src, dst, cx,  cy, angle, sx,  sy); // 缩放->旋转—>偏移

	//cv::imwrite("result.jpg", dst);
	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst",CV_WINDOW_NORMAL);
	cv::imshow("dst", dst);
	cv::waitKey(0);
}