#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

void Histogram_equalization(cv::Mat& src, cv::Mat& dst){
	CV_Assert(src.depth() == CV_8U);
	src.copyTo(dst);
	int nr = src.rows;
	int nc = src.cols;
	int pixnum = nr*nc;
	if (src.channels() == 1){
		//统计直方图
		int gray[256] = { 0 };
		for (int i = 1; i < nr; ++i){
			const uchar* ptr = src.ptr<uchar>(i);
			for (int j = 0; j < nc; ++j){
				gray[ptr[j]]++;
			}
		}
		//计算分布函数
		int LUT[256];
		int sum = 0;
		for (int k = 0; k < 256; k++){
			sum = sum + gray[k];
			LUT[k] = 255 * sum / pixnum;
		}
		//灰度变换（赋值）
		for (int i = 1; i < nr; ++i){
			const uchar* ptr_src = src.ptr<uchar>(i);
			uchar* ptr_dst = dst.ptr<uchar>(i);
			for (int j = 0; j < nc; ++j){
				ptr_dst[j] = LUT[ptr_src[j]];
			}
		}
	}
	else{
		//统计直方图
		int B[256] = { 0 };
		int G[256] = { 0 };
		int R[256] = { 0 };
		for (int i = 0; i < nr; ++i){
			for (int j = 0; j < nc; ++j){
				B[src.at<cv::Vec3b>(i, j)[0]]++;
				G[src.at<cv::Vec3b>(i, j)[1]]++;
				R[src.at<cv::Vec3b>(i, j)[2]]++;
			}
		}
		//计算分布函数
		int LUT_B[256], LUT_G[256], LUT_R[256];
		int sum_B = 0, sum_G = 0, sum_R = 0;
		for (int k = 0; k < 256; k++){
			sum_B = sum_B + B[k];
			sum_G = sum_G + G[k];
			sum_R = sum_R + R[k];
			LUT_B[k] = 255 * sum_B / pixnum;
			LUT_G[k] = 255 * sum_G / pixnum;
			LUT_R[k] = 255 * sum_R / pixnum;
		}
		//灰度变换（赋值）
		for (int i = 0; i < nr; ++i){
			for (int j = 0; j < nc; ++j){
				dst.at<cv::Vec3b>(i, j)[0] = LUT_B[src.at<cv::Vec3b>(i, j)[0]];
				dst.at<cv::Vec3b>(i, j)[1] = LUT_G[src.at<cv::Vec3b>(i, j)[1]];
				dst.at<cv::Vec3b>(i, j)[2] = LUT_R[src.at<cv::Vec3b>(i, j)[2]];
			}
		}
	}
}

void BERF(cv::Mat& src, cv::Mat& dst, int step_r, int step_c , int step_levels, int large_dir){
	//处理子块的行边界（横向边界）
	for (int i = 0; i < dst.rows;){
		for (int j = 0; j < dst.cols; ++j){
			if (i - 1 >= 0 && i + 1 <= dst.rows - 1){ //边界判断，先排除第一行和最后一行
				//块效应检查
				int dfacter_newsrc = abs(src.at<uchar>(i, j) - src.at<uchar>(i + 1, j)) + abs(src.at<uchar>(i, j) - src.at<uchar>(i - 1, j));
				int dfacter_dst = abs(dst.at<uchar>(i, j) - dst.at<uchar>(i + 1, j)) + abs(dst.at<uchar>(i, j) - dst.at<uchar>(i - 1, j));
				if (dfacter_dst - dfacter_newsrc > step_levels){
					//存在块效应执行BERF
					int b = 0;
					int ave_bound = (int)(dst.at<uchar>(i - 1, j) + dst.at<uchar>(i + 1, j)) / 2;
					dst.at<uchar>(i, j) = ave_bound;

					if (dst.at<uchar>(i - 1, j) > dst.at<uchar>(i + 1, j)){ //垂直于子块边界，向上递增方向（ increasing rule）
						if (i - 2 - b >= 0){ //图像边界判断
							int pixel_present_add = dst.at<uchar>(i - 1 - b, j);
							int pixel_next_add = dst.at<uchar>(i - 2 - b, j);
							pixel_present_add = ave_bound + step_levels;
							while (i - 2 - b >= 0 && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir){
								pixel_next_add = pixel_present_add + step_levels;
								b += 1;
								if (i - 2 - b >= 0){ //图像边界判断
									pixel_present_add = dst.at<uchar>(i - 1 - b, j);
									pixel_next_add = dst.at<uchar>(i - 2 - b, j);
								}
							}
						}
						//垂直于子块边界，向下递减方向（ decreasing rule）
						if (i + 2 + b <dst.rows){ //图像边界判断
							int pixel_present_dec = dst.at<uchar>(i + 1 + b, j);
							int pixel_next_dec = dst.at<uchar>(i + 2 + b, j);
							pixel_present_dec = ave_bound - step_levels;
							while (i + 2 + b >= 0 && pixel_present_dec - pixel_next_dec >= step_levels && pixel_present_dec - pixel_next_dec <= large_dir){
								pixel_next_dec = pixel_present_dec - step_levels;
								b += 1;
								if (i + 2 + b >= 0){ //图像边界判断
									pixel_present_dec = dst.at<uchar>(i + 1 + b, j);
									pixel_next_dec = dst.at<uchar>(i + 2 + b, j);
								}
							}
						}

					}
					else if (dst.at<uchar>(i - 1, j) < dst.at<uchar>(i + 1, j)){ ////垂直于子块边界，向下递增方向（ increasing rule）
						if (i + 2 + b < dst.rows){ //图像边界判断
							int pixel_present_add = dst.at<uchar>(i + 1 + b, j);
							int pixel_next_add = dst.at<uchar>(i + 2 + b, j);
							pixel_present_add = ave_bound + step_levels;
							while (i + 2 + b >= 0 && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir){
								pixel_next_add = pixel_present_add + step_levels;
								b += 1;
								if (i + 2 + b >= 0){ //图像边界判断
									pixel_present_add = dst.at<uchar>(i + 1 + b, j);
									pixel_next_add = dst.at<uchar>(i + 2 + b, j);
								}
							}
						}
						//垂直于子块边界，向上递减方向（ decreasing rule）
						if (i - 2 - b >= 0){ //图像边界判断
							int pixel_present_dec = dst.at<uchar>(i - 1 - b, j);
							int pixel_next_dec = dst.at<uchar>(i - 2 - b, j);
							pixel_present_dec = ave_bound - step_levels;
							while (i - 2 - b >= 0 && pixel_present_dec - pixel_next_dec >= step_levels && pixel_present_dec - pixel_next_dec <= large_dir){
								pixel_next_dec = pixel_present_dec - step_levels;
								b += 1;
								if (i - 2 - b >= 0){ //图像边界判断
									pixel_present_dec = dst.at<uchar>(i - 1 - b, j);
									pixel_next_dec = dst.at<uchar>(i - 2 - b, j);
								}
							}
						}

					}
				}
			}
		}
		i += step_r;
	}

	//处理子块的列边界（纵向向边界）
	for (int j = 0; j <dst.cols;){
		for (int i = 0; i < dst.rows; ++i){
			if (j - 1 >= 0 && j + 1 <= dst.cols - 1){ //图像边界判断，先排除第一列和最后一列
				//块效应检查
				int dfacter_newsrc = abs(src.at<uchar>(i, j) - src.at<uchar>(i, j + 1)) + abs(src.at<uchar>(i, j) - src.at<uchar>(i, j - 1));
				int dfacter_dst = abs(dst.at<uchar>(i, j) - dst.at<uchar>(i, j + 1)) + abs(dst.at<uchar>(i, j) - dst.at<uchar>(i, j + 1));
				if (dfacter_dst - dfacter_newsrc > step_levels){
					//存在块效应执行BERF
					int b = 0;
					int ave_bound = (int)(dst.at<uchar>(i, j - 1) + dst.at<uchar>(i, j + 1)) / 2;
					dst.at<uchar>(i, j) = ave_bound;

					if (dst.at<uchar>(i, j - 1) > dst.at<uchar>(i, j + 1)){ //垂直于子块边界，向左递增方向（ increasing rule）
						if (j - 2 - b >= 0){ //图像边界判断
							int pixel_present_add = dst.at<uchar>(i, j - 1 - b);
							int pixel_next_add = dst.at<uchar>(i, j - 2 - b);
							pixel_present_add = ave_bound + step_levels;
							while (j - 2 - b >= 0 && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir){
								pixel_next_add = pixel_present_add + step_levels;
								b += 1;
								if (j - 2 - b >= 0){ //图像边界判断
									pixel_present_add = dst.at<uchar>(i, j - 1 - b);
									pixel_next_add = dst.at<uchar>(i, j - 2 - b);
								}
							}
						}
						//垂直于子块边界，向右递减方向（ decreasing rule）
						if (j + 2 + b <dst.cols){ //图像边界判断
							int pixel_present_dec = dst.at<uchar>(i, j + 1 + b);
							int pixel_next_dec = dst.at<uchar>(i, j + 2 + b);
							pixel_present_dec = ave_bound - step_levels;
							while (j + 2 + b >= 0 && pixel_present_dec - pixel_next_dec >= step_levels && pixel_present_dec - pixel_next_dec <= large_dir){
								pixel_next_dec = pixel_present_dec - step_levels;
								b += 1;
								if (j + 2 + b >= 0){ //图像边界判断
									pixel_present_dec = dst.at<uchar>(i, j + 1 + b);
									pixel_next_dec = dst.at<uchar>(i, j + 2 + b);
								}
							}
						}

					}
					else if (dst.at<uchar>(i, j - 1) < dst.at<uchar>(i, j + 1)){ ////垂直于子块边界，向右递增方向（ increasing rule）
						if (j + 2 + b < dst.cols){ //图像边界判断
							int pixel_present_add = dst.at<uchar>(i, j + 1 + b);
							int pixel_next_add = dst.at<uchar>(i, j + 1 + b);
							pixel_present_add = ave_bound + step_levels;
							while (j + 2 + b >= 0 && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir){
								pixel_next_add = pixel_present_add + step_levels;
								b += 1;
								if (j + 2 + b >= 0){ //图像边界判断
									pixel_present_add = dst.at<uchar>(i, j + 1 + b);
									pixel_next_add = dst.at<uchar>(i, j + 2 + b);
								}
							}
						}
						//垂直于子块边界，向左递减方向（ decreasing rule）
						if (j - 2 - b >= 0){ //图像边界判断
							int pixel_present_dec = dst.at<uchar>(i, j - 1 - b);
							int pixel_next_dec = dst.at<uchar>(i, j - 2 - b);
							pixel_present_dec = ave_bound - step_levels;
							while (j - 2 - b >= 0 && pixel_present_dec - pixel_next_dec >= step_levels && pixel_present_dec - pixel_next_dec <= large_dir){
								pixel_next_dec = pixel_present_dec - step_levels;
								b += 1;
								if (j - 2 - b >= 0){ //图像边界判断
									pixel_present_dec = dst.at<uchar>(i, j - 1 - b);
									pixel_next_dec = dst.at<uchar>(i, j - 2 - b);
								}
							}
						}
					}
				}
			}
		}
		j += step_c;
	}
}


/***************************************
POSHE
src - 输入图像
dst - 输出图像
s  -  子块尺寸=原图尺寸/s ，2的倍数
k  -  子块移动步长=子块尺寸/k, 2的倍数
****************************************/
void Poshe(cv::Mat& src, cv::Mat& dst, cv::Mat& newsrc_draw,float s, float k){
	int nr = src.rows;
	int nc = src.cols;

	//边界扩充
	int newnr = ceil(nr / s / k)*s*k;
	int newnc = ceil(nc / s / k)*s*k;
	cv::Mat newsrc;
	cv::copyMakeBorder(src, newsrc, 0 , newnr - nr, 0, newnc - nc, cv::BORDER_REFLECT);
	dst=cv::Mat::zeros(newnr, newnc, CV_16U); //因为8位无符号整型的范围是0~255，而在累加过程中会超过这个范围，所以采用16位无符号整型


	//创建子块，确定子块移动步长
	int sub_block_r = newnr / s; //子块高度
	int sub_block_c = newnc / s; //子块宽度
	int step_r = sub_block_r / k; //垂直移动步长
	int step_c = sub_block_c / k; //水平移动步长

	//对当前子块进行直方图均衡
	int sub_block_x ; //子块左顶点坐标(行)
	int sub_block_y; //子块左顶点坐标(列)
	cv::Mat HE_frequency = cv::Mat::zeros(newnr, newnc, CV_8U); //均衡频率计数矩阵
	cv::Mat sub_block_HE;
	newsrc.copyTo(newsrc_draw);
	for (sub_block_x=0; sub_block_x <= (newnr - sub_block_r); ){
		for (sub_block_y = 0; sub_block_y <= (newnc - sub_block_c); ){
			cv::Mat sub_block = newsrc(cv::Rect(sub_block_y, sub_block_x, sub_block_c, sub_block_r));
			Histogram_equalization(sub_block, sub_block_HE);
			cv::rectangle(newsrc_draw, cv::Rect(sub_block_y, sub_block_x, sub_block_c, sub_block_r), cv::Scalar(0, 0, 0), 1.8, 1, 0);
			
			//将直方图均衡后子块的像素值映射至输出图像	
			int sub_block_HE_i = 0;
			for (int i = sub_block_x; i < sub_block_x + sub_block_r; i++){	
				 int sub_block_HE_j = 0;
				for (int j = sub_block_y; j < sub_block_y + sub_block_c; j++){	 
					 dst.at<ushort>(i, j) = dst.at<ushort>(i, j) + sub_block_HE.at<uchar>(sub_block_HE_i, sub_block_HE_j);
					 HE_frequency.at<uchar>(i, j)++;
					 sub_block_HE_j++;
				}
				sub_block_HE_i++;	
			}

			sub_block_y = sub_block_y + step_c;
		} 
		sub_block_x = sub_block_x + step_r;
	}

	for (int i = 0; i < dst.rows; ++i){
		for (int j = 0; j < dst.cols; ++j){
			dst.at<ushort>(i, j) = (dst.at<ushort>(i, j)) / (HE_frequency.at<uchar>(i, j));
		}
	}
	dst.convertTo(dst, CV_8U, 1, 0); //数据类型转换

	//BERF
	int step_levels = 3; 
	int large_dir = 40;
	BERF(newsrc, dst, step_r, step_c, step_levels, large_dir);

	dst = dst(cv::Rect(0, 0, src.cols, src.rows));	
}



int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\ya.png");
	if (src.empty()){
		return -1;
	}
	cvtColor(src, src, CV_RGB2GRAY);
	cv::Mat dst;
	cv::Mat newsrc_draw;
	Poshe(src, dst, newsrc_draw, 3.0, 12.0); //poshe
	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::namedWindow("newsrc_draw");
	cv::imshow("newsrc_draw", newsrc_draw);
	cv::imwrite("I:/Learning-and-Practice/2019Change/Image process algorithm/POSHE/POSHE/ya_ff.jpg", dst);
	cv::waitKey(0);
}