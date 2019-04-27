%% This is an example of POSHE.
%@ date ： 2019/4/27
%@ author ： Xiaowu
%@ CSDN ：https://blog.csdn.net/weixin_40647819/article/details/88416512
%@ github ：https://github.com/2209520576/Image-Processing-Algorithm
%//////////////////////////////////////////
%POSHE_demo
%src - 输入图像
%dst - 输出图像
%s   -  子块尺寸=原图尺寸/s ，一般取2的倍数
%k   -  子块移动步长=子块尺寸/k, 一般取2的倍数
%///////////////////////////////////////////

%%
clc;clear all;close all
src=imread('vessel.bmp');

if ndims(src)==3
   src=rgb2gray(src);
end
figure;imshow(src);title('src');
%%
s=2; %建议2~4 
k=16; %建议12~16
dst=POSHE(src, s ,k);
figure;imshow(dst);title('dst');