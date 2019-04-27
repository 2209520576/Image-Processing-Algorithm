%% This is an example of POSHE.
%@ date �� 2019/4/27
%@ author �� Xiaowu
%@ CSDN ��https://blog.csdn.net/weixin_40647819/article/details/88416512
%@ github ��https://github.com/2209520576/Image-Processing-Algorithm
%//////////////////////////////////////////
%POSHE_demo
%src - ����ͼ��
%dst - ���ͼ��
%s   -  �ӿ�ߴ�=ԭͼ�ߴ�/s ��һ��ȡ2�ı���
%k   -  �ӿ��ƶ�����=�ӿ�ߴ�/k, һ��ȡ2�ı���
%///////////////////////////////////////////

%%
clc;clear all;close all
src=imread('vessel.bmp');

if ndims(src)==3
   src=rgb2gray(src);
end
figure;imshow(src);title('src');
%%
s=2; %����2~4 
k=16; %����12~16
dst=POSHE(src, s ,k);
figure;imshow(dst);title('dst');