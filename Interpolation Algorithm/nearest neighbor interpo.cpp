//最近邻插值算法
//sx,xy -- 缩放因子
 void nearest(cv::Mat& src, cv::Mat& dst , float sx, float sy){
	 //由缩放因子计算输出图像的尺寸（四舍五入）
	int dst_cols = round(src.cols * sx);
	int dst_rows = round(src.rows * sy);
	//创建输出图像
	dst = cv::Mat(dst_rows,dst_cols,src.type());
	//灰度图像处理
	if (src.channels() == 1){
		for (int i = 0; i < dst.rows; i++){
			for (int j = 0; j < dst.cols; j++){
				//插值计算，输出图像的像素点由原图像对应的最近的像素点得到（四舍五入）
				int i_index = round(i / sy);
				int j_index = round(j / sx);
				if (i_index >src.rows - 1) i_index = src.rows - 1;//防止越界
				if (j_index >src.cols - 1) j_index = src.cols - 1;//防止越界
				dst.at<uchar>(i, j) = src.at<uchar>(i_index, j_index);
			}
		}	
	}
	//彩色图像处理
	else{
		for (int i = 0; i < dst.rows; i++){
			for (int j = 0; j < dst.cols; j++){
				//插值计算，输出图像的像素点由原图像对应的最近的像素点得到（四舍五入）
				int i_index = round(i / sy);
				int j_index = round(j / sx);
				if (i_index >src.rows - 1) i_index = src.rows - 1;//防止越界
				if (j_index >src.cols - 1) j_index = src.cols - 1;//防止越界
				//B
				dst.at<cv::Vec3b>(i, j)[0] = src.at<cv::Vec3b>(i_index, j_index)[0];
				//G
				dst.at<cv::Vec3b>(i, j)[1] = src.at<cv::Vec3b>(i_index, j_index)[1];
				//R
				dst.at<cv::Vec3b>(i, j)[2] = src.at<cv::Vec3b>(i_index, j_index)[2];
			}
		}
	}	
}
