%% This is a function of POSHE.
%@ date ： 2019/4/27
%@ author ： Xiaowu
%@ CSDN ：https://blog.csdn.net/weixin_40647819/article/details/88416512
%@ github ：https://github.com/2209520576/Image-Processing-Algorithm
%//////////////////////////////////////
%POSHE
%src - 输入图像
%dst - 输出图像
%s   -  子块尺寸=原图尺寸/s ，取2的倍数
%k   -  子块移动步长=子块尺寸/k, 取2的倍数
%/////////////////////////////////////
%%
function dst=POSHE(src, s ,k)

if ndims(src)==3
   src=rgb2gray(src);
end
[nr,nc]=size(src);

%边界扩充
newnr = ceil(nr / s / k)*s*k;
newnc = ceil(nc / s / k)*s*k;
newsrc=padarray(src,[newnr - nr  newnc - nc],'replicate','post');
dst=zeros(newnr,newnc,'uint16'); %因为8位无符号整型的范围是0~255，而在累加过程中会超过这个范围，所以采用16位无符号整型

%创建子块，确定子块移动步长
sub_block_r = newnr / s;  %子块高度
sub_block_c = newnc / s;  %子块宽度
step_r = sub_block_r / k; %垂直移动步长
step_c = sub_block_c / k; %水平移动步长

%对当前子块进行直方图均衡
HE_frequency=zeros(newnr,newnc,'uint16'); %均衡频率计数矩阵
newsrc_draw=newsrc;
figure;imshow(newsrc_draw); title('子块移动网格'); %画子块产生的网格
for sub_block_x= 1 : step_r : (newnr - sub_block_r + 1)
    for  sub_block_y= 1 : step_c : (newnc - sub_block_c + 1)
         sub_block=newsrc(sub_block_x:sub_block_x+sub_block_r-1 ,sub_block_y:sub_block_y+sub_block_c-1);
         sub_block_HE=histeq(sub_block); %子块直方图均衡
         rectangle('Position',[sub_block_y,sub_block_x,sub_block_c-1,sub_block_r-1],'edgecolor','c'); %画子块产生的网格
         
         %将直方图均衡后子块的像素值映射至输出图像	
         sub_block_HE_i=1;
         for i=sub_block_x:(sub_block_x + sub_block_r-1)
             sub_block_HE_j = 1;
             for j=sub_block_y:(sub_block_y + sub_block_c-1)
                dst(i,j)=dst(i,j) + uint16(sub_block_HE(sub_block_HE_i,sub_block_HE_j)); 
                HE_frequency(i,j)=HE_frequency(i,j) + 1;
                sub_block_HE_j=sub_block_HE_j + 1;
             end
             sub_block_HE_i=sub_block_HE_i + 1;
         end

    end
    
end

 dst=dst./HE_frequency;
 dst=uint8(dst); %数据类型转换
 
 %BERF
 step_levels = 2; 
 large_dir = 15;
 dst=BERF(newsrc, dst, step_r, step_c, step_levels, large_dir);
 dst=imcrop(dst,[1,1,nc-1,nr-1]);
end

