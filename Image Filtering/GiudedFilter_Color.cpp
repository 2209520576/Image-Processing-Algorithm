
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


///////////////////////////////////////////////////////////////////
//   GUIDEDFILTER_COLOR   O(1) time implementation of guided filter using a color image as the guidance.
//
//   -guidance image : I(should be a color(RGB) image)
//	 -filtering input image : p(should be a gray - scale / single channel image)
//   -local window radius : r
//   -regularization parameter : eps
///////////////////////////////////////////////////////////////////
cv::Mat GuidedFilter_Color(cv::Mat& I, cv::Mat& p, int r, double eps ){
	int wsize = 2 * r + 1;
	//数据类型转换
	I.convertTo(I, CV_64F, 1.0 / 255.0);
	p.convertTo(p, CV_64F, 1.0 / 255.0);
	
	//引导图通道分离
	if (I.channels() == 1){
		std::cout<<"I should be a color(RGB) image "<<std::endl;
	}
	std::vector<cv::Mat> rgb;
	cv::split(I, rgb);

	//meanI=fmean(I)
	cv::Mat mean_I_r, mean_I_g, mean_I_b;
	cv::boxFilter(rgb[0], mean_I_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(rgb[1], mean_I_g, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(rgb[2], mean_I_r, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//meanP=fmean(P)
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//corrI=fmean(I.*I)
	cv::Mat mean_II_rr, mean_II_rg, mean_II_rb, mean_II_gb, mean_II_gg, mean_II_bb;
	cv::boxFilter(rgb[2].mul(rgb[2]), mean_II_rr, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(rgb[2].mul(rgb[1]), mean_II_rg, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(rgb[2].mul(rgb[0]), mean_II_rb, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(rgb[1].mul(rgb[0]), mean_II_gb, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(rgb[1].mul(rgb[1]), mean_II_gg, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(rgb[0].mul(rgb[0]), mean_II_bb, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//corrIp=fmean(I.*p)
	cv::Mat mean_Ip_r, mean_Ip_g, mean_Ip_b;
	mean_Ip_b = rgb[0].mul(p);
	mean_Ip_g = rgb[1].mul(p);
	mean_Ip_r = rgb[2].mul(p);
	cv::boxFilter(mean_Ip_b, mean_Ip_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(mean_Ip_g, mean_Ip_g, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(mean_Ip_r, mean_Ip_r, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//covIp=corrIp-meanI.*meanp
	cv::Mat cov_Ip_r, cov_Ip_g, cov_Ip_b;
	cv::subtract(mean_Ip_r, mean_I_r.mul(mean_p), cov_Ip_r);
	cv::subtract(mean_Ip_g, mean_I_g.mul(mean_p), cov_Ip_g);
	cv::subtract(mean_Ip_b, mean_I_b.mul(mean_p), cov_Ip_b);

	//varI=corrI-meanI.*meanI
	//variance of I in each local patch : the matrix Sigma in Eqn(14).
	//Note the variance in each local patch is a 3x3 symmetric matrix :
	//           rr, rg, rb
	//   Sigma = rg, gg, gb
	//           rb, gb, bb
	cv::Mat var_I_rr, var_I_rg, var_I_rb, var_I_gb, var_I_gg, var_I_bb;
	cv::subtract(mean_II_rr, mean_I_r.mul(mean_I_r), var_I_rr);
	cv::subtract(mean_II_rg, mean_I_r.mul(mean_I_g), var_I_rg);
	cv::subtract(mean_II_rb, mean_I_r.mul(mean_I_b), var_I_rb);
	cv::subtract(mean_II_gb, mean_I_g.mul(mean_I_b), var_I_gb);
	cv::subtract(mean_II_gg, mean_I_g.mul(mean_I_g), var_I_gg);
	cv::subtract(mean_II_bb, mean_I_b.mul(mean_I_b), var_I_bb);

	//a=conIp./(varI+eps)
	int cols = p.cols;
	int rows = p.rows;
	cv::Mat Mat_a = cv::Mat::zeros(rows, cols,CV_64FC3);
	std::vector<cv::Mat> a;
	cv::split(Mat_a, a);
	double rr, rg, rb, gg, gb, bb;
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < cols; ++j){
			rr = var_I_rr.at<double>(i, j); rg = var_I_rg.at<double>(i, j); rb = var_I_rb.at<double>(i, j);
			gg = var_I_gg.at<double>(i, j); gb = var_I_gb.at<double>(i, j);
		    bb = var_I_bb.at<double>(i, j);
			cv::Mat sigma = (cv::Mat_<double>(3, 3) << rr, rg, rb,
													   rg, gg, gb,
				                                       rb, gb, bb);
			cv::Mat cov_Ip = (cv::Mat_<double>(1, 3) << cov_Ip_r.at<double>(i, j), cov_Ip_g.at<double>(i, j), cov_Ip_b.at<double>(i, j));
			cv::Mat eye = cv::Mat::eye(3, 3, CV_64FC1);
			sigma = sigma + eps*eye;
			cv::Mat sigma_inv = sigma.inv();//求逆矩阵
			cv::Mat tmp = cov_Ip*sigma_inv;
			a[2].at<double>(i, j) = tmp.at<double>(0, 0);//r
			a[1].at<double>(i, j) = tmp.at<double>(0, 1);//g
			a[0].at<double>(i, j) = tmp.at<double>(0, 2);//b
		}
	}

	//b=meanp-a.*meanI
	cv::Mat b = mean_p - a[0].mul(mean_I_b) - a[1].mul(mean_I_g) - a[2].mul(mean_I_r);

	//meana=fmean(a)
	//meanb=fmean(b)
	cv::Mat mean_a_r, mean_a_g, mean_a_b, mean_b;
	cv::boxFilter(a[0], mean_a_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(a[1], mean_a_g, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(a[2], mean_a_r, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波
	cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);//盒子滤波

	//q=meana.*I+meanb
	cv::Mat q = mean_a_r.mul(rgb[2]) + mean_a_g.mul(rgb[1]) + mean_a_b.mul(rgb[0]) + mean_b;

	//数据类型转换
	I.convertTo(I, CV_8UC3, 255);
	p.convertTo(p, CV_8U, 255);
	q.convertTo(q, CV_8U, 255);

	return q;
}

int main(){
	cv::Mat I = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\woman.jpg");
	cv::Mat P = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\woman.jpg");
	if (I.empty() || P.empty()){
		return -1;
	}
	if (P.channels() > 1)
		cv::cvtColor(P, P, CV_RGB2GRAY);
	
	//自编GuidedFilter测试
	double t2 = (double)cv::getTickCount(); //测时间
	cv::Mat q;
	q = GuidedFilter_Color(I, P, 9, 0.1*0.1);
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "MyGuidedFilter_process=" << time2 << " ms. " << std::endl << std::endl;

	cv::namedWindow("GuidedImg", CV_WINDOW_NORMAL);
	cv::imshow("GuidedImg", I);
	cv::namedWindow("src", CV_WINDOW_NORMAL);
	cv::imshow("src", P);
	cv::namedWindow("GuidedFilter", CV_WINDOW_NORMAL);
	cv::imshow("GuidedFilter", q);
	cv::waitKey(0);

}
