#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>


/*ͼ����ת����ͼ������Ϊ��ת���ģ�*/
void affine_trans_rotate(cv::Mat& src, cv::Mat& dst, double Angle){
	double angle = Angle*CV_PI / 180.0;
	//�������ͼ��
	int dst_rows = round(fabs(src.rows * cos(angle)) + fabs(src.cols * sin(angle)));//ͼ��߶�
	int dst_cols = round(fabs(src.cols * cos(angle)) + fabs(src.rows * sin(angle)));//ͼ����

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //�Ҷ�ͼ��ʼ
	} 
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGBͼ��ʼ
	}

	cv::Mat T1 = (cv::Mat_<double>(3,3) << 1.0,0.0,0.0 , 0.0,-1.0,0.0, -0.5*src.cols , 0.5*src.rows , 1.0); // ��ԭͼ������ӳ�䵽��ѧ�ѿ�������
	cv::Mat T2 = (cv::Mat_<double>(3,3) << cos(angle),-sin(angle),0.0 , sin(angle), cos(angle),0.0, 0.0,0.0,1.0); //��ѧ�ѿ���������˳ʱ����ת�ı任����
	double t3[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 }, { 0.5*dst.cols, 0.5*dst.rows ,1.0} }; // ����ѧ�ѿ�������ӳ�䵽��ת���ͼ������
	cv::Mat T3 = cv::Mat(3.0,3.0,CV_64FC1,t3);
	cv::Mat T = T1*T2*T3;
	cv::Mat T_inv = T.inv(); // �������

	for (double i = 0.0; i < dst.rows; i++){
		for (double j = 0.0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1.0);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // ԭͼ��ĺ����꣬�У���
			double w = src_coordinate.at<double>(0, 1); // ԭͼ��������꣬�У���

			/*˫���Բ�ֵ*/
			// �ж��Ƿ�Խ��
			if (int(Angle) % 90 == 0) {
				if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
				if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; //����Ҫ���ϣ��������ֱ߽�����
			}

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double pw = w - top ; //pwΪ���� �� ��С������(����ƫ��)
				double pv = v - left; //pvΪ���� �� ��С������(����ƫ��)
				if (src.channels() == 1){
					//�Ҷ�ͼ��
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//��ɫͼ��
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}

/*ƽ�Ʊ任*����ͼ���󶥵�Ϊԭ�㣩/
/****************************************
tx: ˮƽƽ�ƾ��� ���������ƶ� ���������ƶ�
ty: ��ֱƽ�ƾ��� ���������ƶ� ���������ƶ�
*****************************************/
void affine_trans_translation(cv::Mat& src, cv::Mat& dst, double tx, double ty){
	//�������ͼ��
	int dst_rows = src.rows;//ͼ��߶�
	int dst_cols = src.cols;//ͼ����

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //�Ҷ�ͼ��ʼ
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGBͼ��ʼ
	}

	cv::Mat T = (cv::Mat_<double>(3, 3) << 1,0,0 , 0,1,0 , tx,ty,1); //ƽ�Ʊ任����
	cv::Mat T_inv = T.inv(); // �������

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // ԭͼ��ĺ����꣬�У���
			double w = src_coordinate.at<double>(0, 1); // ԭͼ��������꣬�У���

			/*˫���Բ�ֵ*/
			// �ж��Ƿ�Խ��

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double pw = w - top; //pwΪ���� �� ��С������(����ƫ��)
				double pv = v - left; //pvΪ���� �� ��С������(����ƫ��)
				if (src.channels() == 1){
					//�Ҷ�ͼ��
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//��ɫͼ��
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}


/*�߶ȱ任*����ͼ���󶥵�Ϊԭ�㣩/
/***************
cx: ˮƽ���ų߶�
cy: ��ֱ���ų߶�
***************/
void affine_trans_scale(cv::Mat& src, cv::Mat& dst, double cx, double cy){
	//�������ͼ��
	int dst_rows = round(cy*src.rows);//ͼ��߶�
	int dst_cols = round(cx*src.cols);//ͼ����

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //�Ҷ�ͼ��ʼ
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGBͼ��ʼ
	}

	cv::Mat T = (cv::Mat_<double>(3, 3) <<cx,0,0, 0,cy,0 ,0,0,1 ); //�߶ȱ任����
	cv::Mat T_inv = T.inv(); // �������

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // ԭͼ��ĺ����꣬�У���
			double w = src_coordinate.at<double>(0, 1); // ԭͼ��������꣬�У���

			/*˫���Բ�ֵ*/
			// �ж��Ƿ�Խ��
			if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
			if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; 

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double pw = w - top; //pwΪ���� �� ��С������(����ƫ��)
				double pv = v - left; //pvΪ���� �� ��С������(����ƫ��)
				if (src.channels() == 1){
					//�Ҷ�ͼ��
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//��ɫͼ��
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}

/*ƫ�Ʊ任*����ͼ������Ϊƫ�����ģ�/
/***************************************
sx: ˮƽƫ�Ƴ߶� ��������ƫ�� ��������ƫ��
sy: ��ֱƫ�Ƴ߶� ��������ƫ�� ��������ƫ��
****************************************/
void affine_trans_deviation(cv::Mat& src, cv::Mat& dst, double sx, double sy){
	//�������ͼ��
	int dst_rows = fabs(sy)*src.cols + src.rows;//ͼ��߶�
	int dst_cols = fabs(sx)*src.rows + src.cols;//ͼ����

	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //�Ҷ�ͼ��ʼ
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGBͼ��ʼ
	}

	cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, -0.5*src.cols, 0.5*src.rows, 1); // ��ԭͼ������ӳ�䵽��ѧ�ѿ�������
	cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1,sy,0, sx,1,0, 0,0,1); //��ѧ�ѿ�������ƫ�Ʊ任����
	double t3[3][3] = { { 1, 0, 0 }, { 0, -1, 0 }, { 0.5*dst.cols, 0.5*dst.rows, 1 } }; // ����ѧ�ѿ�������ӳ�䵽��ת���ͼ������
	cv::Mat T3 = cv::Mat(3, 3, CV_64FC1, t3);
	cv::Mat T = T1*T2*T3;
	cv::Mat T_inv = T.inv(); // �������

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // ԭͼ��ĺ����꣬�У���
			double w = src_coordinate.at<double>(0, 1); // ԭͼ��������꣬�У���

			/*˫���Բ�ֵ*/
			// �ж��Ƿ�Խ��

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double pw = w - top; //pwΪ���� �� ��С������(����ƫ��)
				double pv = v - left; //pvΪ���� �� ��С������(����ƫ��)
				if (src.channels() == 1){
					//�Ҷ�ͼ��
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//��ɫͼ��
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}

/*��ϱ任*/
/*�任˳������->��ת��>ƫ��*/
void affine_trans_comb(cv::Mat& src, cv::Mat& dst, double cx, double cy, double Angle, double sx, double sy){
	double angle = Angle*CV_PI / 180;
	//�������ͼ��
	int dst_s_rows = round(cy*src.rows);//�߶ȱ任��ͼ��߶�
	int dst_s_cols = round(cx*src.cols);//�߶ȱ任��ͼ����

	int dst_sr_rows = round(fabs(dst_s_rows * cos(angle)) + fabs(dst_s_cols * sin(angle)));//�پ�����ת��ͼ��߶�
	int dst_sr_cols = round(fabs(dst_s_cols * cos(angle)) + fabs(dst_s_rows * sin(angle)));//�پ�����ת��ͼ����

	int dst_srd_rows = fabs(sy)*dst_sr_cols + dst_sr_rows;//��󾭹�ƫ�ƺ�ͼ��߶�
    int dst_srd_cols = fabs(sx)*dst_sr_rows + dst_sr_cols;//��󾭹�ƫ�ƺ�ͼ����


	if (src.channels() == 1) {

		dst = cv::Mat::zeros(dst_srd_rows, dst_srd_cols, CV_8UC1); //�Ҷ�ͼ��ʼ
	}
	else {
		dst = cv::Mat::zeros(dst_srd_rows, dst_srd_cols, CV_8UC3); //RGBͼ��ʼ
	}

	cv::Mat T1 = (cv::Mat_<double>(3, 3) << cx, 0, 0, 0, cy, 0, 0, 0, 1); //�߶ȱ任����

	cv::Mat T21 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, -0.5*dst_s_cols, 0.5*dst_s_rows, 1); // ���߶ȱ任���ͼ������ӳ�䵽��ѧ�ѿ�������
	cv::Mat T22 = (cv::Mat_<double>(3, 3) << cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1); //��ѧ�ѿ���������˳ʱ����ת�ı任����
	cv::Mat T2 = T21*T22;// ���ﲻ��Ҫת��ͼ�������ˣ���Ϊ�����ƫ�Ʊ任���ڵѿ��������½��е�

	cv::Mat T32 = (cv::Mat_<double>(3, 3) << 1, sy, 0, sx, 1, 0, 0, 0, 1); //��ѧ�ѿ�������ƫ�Ʊ任����
	double t33[3][3] = { { 1, 0, 0 }, { 0, -1, 0 }, { 0.5*dst.cols, 0.5*dst.rows, 1 } }; // ����ѧ�ѿ�������ӳ�䵽ƫ�ƺ��ͼ������
	cv::Mat T33 = cv::Mat(3, 3, CV_64FC1, t33);
	cv::Mat T3 = T32*T33;

	cv::Mat T = T1*T2*T3; //������˰�����ϱ任��˳��
	cv::Mat T_inv = T.inv(); // �������

	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // ԭͼ��ĺ����꣬�У���
			double w = src_coordinate.at<double>(0, 1); // ԭͼ��������꣬�У���

			/*˫���Բ�ֵ*/
			// �ж��Ƿ�Խ��
			if (int(Angle) % 90 == 0) {
				if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
				if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; //����Ҫ���ϣ��������ֱ߽�����
			}

			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double pw = w - top; //pwΪ���� �� ��С������(����ƫ��)
				double pv = v - left; //pvΪ���� �� ��С������(����ƫ��)
				if (src.channels() == 1){
					//�Ҷ�ͼ��
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//��ɫͼ��
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
	double angle =250;  //��ת�Ƕ�
	double tx = 50, ty = -50; //ƽ�ƾ���
	double cx =1.5, cy = 1.5; //���ų߶�
	double sx = 0.2, sy =0.2; //ƫ�Ƴ߶�

	//affine_trans_rotate(src, dst, angle); //��ת

	//affine_trans_translation(src, dst, tx,ty); //ƽ��

	//affine_trans_scale(src, dst, cx, cy);  //�߶�����

    //affine_trans_deviation(src, dst, sx, sy); //ƫ��

	affine_trans_comb(src, dst, cx,  cy, angle, sx,  sy); // ����->��ת��>ƫ��

	//cv::imwrite("result.jpg", dst);
	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst",CV_WINDOW_NORMAL);
	cv::imshow("dst", dst);
	cv::waitKey(0);
}