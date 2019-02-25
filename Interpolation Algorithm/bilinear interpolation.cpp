//双线性插值
////sx,xy -- 缩放因子
void Inter_Linear(cv::Mat& src,cv::Mat& dst, double sx,double sy ){
	int dst_rows = round(sx*src.rows);
	int dst_cols = round(sy*src.cols);
	dst = cv::Mat(dst_rows,dst_cols,src.type());
	for (int i = 0; i < dst.rows; i++){
		//几何中心对齐
		double index_i = (i+0.5) / sx -0.5;
		//防止越界
		if (index_i<0) index_i = 0;
		if (index_i>src.rows - 1) index_i = src.rows - 1;
		//相邻4*4像素的行（坐标）
		int i1 = floor(index_i);
		int i2 = ceil(index_i);
		//u为得到浮点型坐标行的小数部分
		double u = index_i - i1;
		for (int j = 0; j < dst.cols; j++){
			//几何中心对齐
			double index_j = (j+0.5) / sy -0.5;
			//防止越界
			if (index_j<0) index_j = 0;
			if (index_j>src.cols - 1) index_j = src.cols - 1;
			//相邻4*4像素的列（坐标）
			int j1 = floor(index_j);
			int j2 = ceil(index_j);
			//v为得到浮点型坐标列的小数部分
			double v = index_j - j1;
			if (src.channels() == 1){
				//灰度图像
				dst.at<uchar>(i, j) = (1 - u)*(1 - v)*src.at<uchar>(i1, j1) + (1 - u)*v*src.at<uchar>(i1, j2) + u*(1 - v)*src.at<uchar>(i2, j1) + u*v*src.at<uchar>(i2, j2);
			}
			else{
				//彩色图像
				dst.at<cv::Vec3b>(i, j)[0] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[0] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[0] + u*(1 - v)*src.at<cv::Vec3b>(i2, j1)[0] + u*v*src.at<cv::Vec3b>(i2, j2)[0];
				dst.at<cv::Vec3b>(i, j)[1] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[1] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[1] + u*(1 - v)*src.at<cv::Vec3b>(i2, j1)[1] + u*v*src.at<cv::Vec3b>(i2, j2)[1];
				dst.at<cv::Vec3b>(i, j)[2] = (1 - u)*(1 - v)*src.at<cv::Vec3b>(i1, j1)[2] + (1 - u)*v*src.at<cv::Vec3b>(i1, j2)[2] + u*(1 - v)*src.at<cv::Vec3b>(i2, j1)[2] + u*v*src.at<cv::Vec3b>(i2, j2)[2];
			}
		}
	}
}
