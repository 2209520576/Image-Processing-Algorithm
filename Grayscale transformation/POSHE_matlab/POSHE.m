%% This is a function of POSHE.
%@ date �� 2019/4/27
%@ author �� Xiaowu
%@ CSDN ��https://blog.csdn.net/weixin_40647819/article/details/88416512
%@ github ��https://github.com/2209520576/Image-Processing-Algorithm
%//////////////////////////////////////
%POSHE
%src - ����ͼ��
%dst - ���ͼ��
%s   -  �ӿ�ߴ�=ԭͼ�ߴ�/s ��ȡ2�ı���
%k   -  �ӿ��ƶ�����=�ӿ�ߴ�/k, ȡ2�ı���
%/////////////////////////////////////
%%
function dst=POSHE(src, s ,k)

if ndims(src)==3
   src=rgb2gray(src);
end
[nr,nc]=size(src);

%�߽�����
newnr = ceil(nr / s / k)*s*k;
newnc = ceil(nc / s / k)*s*k;
newsrc=padarray(src,[newnr - nr  newnc - nc],'replicate','post');
dst=zeros(newnr,newnc,'uint16'); %��Ϊ8λ�޷������͵ķ�Χ��0~255�������ۼӹ����лᳬ�������Χ�����Բ���16λ�޷�������

%�����ӿ飬ȷ���ӿ��ƶ�����
sub_block_r = newnr / s;  %�ӿ�߶�
sub_block_c = newnc / s;  %�ӿ���
step_r = sub_block_r / k; %��ֱ�ƶ�����
step_c = sub_block_c / k; %ˮƽ�ƶ�����

%�Ե�ǰ�ӿ����ֱ��ͼ����
HE_frequency=zeros(newnr,newnc,'uint16'); %����Ƶ�ʼ�������
newsrc_draw=newsrc;
figure;imshow(newsrc_draw); title('�ӿ��ƶ�����'); %���ӿ����������
for sub_block_x= 1 : step_r : (newnr - sub_block_r + 1)
    for  sub_block_y= 1 : step_c : (newnc - sub_block_c + 1)
         sub_block=newsrc(sub_block_x:sub_block_x+sub_block_r-1 ,sub_block_y:sub_block_y+sub_block_c-1);
         sub_block_HE=histeq(sub_block); %�ӿ�ֱ��ͼ����
         rectangle('Position',[sub_block_y,sub_block_x,sub_block_c-1,sub_block_r-1],'edgecolor','c'); %���ӿ����������
         
         %��ֱ��ͼ������ӿ������ֵӳ�������ͼ��	
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
 dst=uint8(dst); %��������ת��
 
 %BERF
 step_levels = 2; 
 large_dir = 15;
 dst=BERF(newsrc, dst, step_r, step_c, step_levels, large_dir);
 dst=imcrop(dst,[1,1,nc-1,nr-1]);
end

