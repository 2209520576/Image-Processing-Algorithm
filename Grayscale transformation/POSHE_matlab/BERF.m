%% This is a function of BERF.
%@ date �� 2019/4/27
%@ author �� Xiaowu
%@ CSDN ��https://blog.csdn.net/weixin_40647819/article/details/88416512
%@ github ��https://github.com/2209520576/Image-Processing-Algorithm
%///////////////////////////////////////////////////
%BERF�˲���
%src - ����ͼ��
%dst - Ŀ��ͼ��
%step_r  - �ӿ鴹ֱ�ƶ�����
%step_c  - �ӿ�ˮƽ�ƶ�����
%step_levels - �Ҷȼ������仯�Ĳ���
%large_dir  - ���ҶȲ���
%//////////////////////////////////////////////////
%%
function dst=BERF(src, dst,step_r, step_c, step_levels, large_dir)
[dstrows,dstcols]=size(dst);

%�����ӿ���б߽磨����߽磩
for i=1:step_r:dstrows
    for j=1:dstcols
        if (i>1 && i<dstrows ) %ͼ��߽��жϣ����ų���һ�к����һ��
            %��ЧӦ���
           dfacter_newsrc=abs(src(i,j)-src(i+1,j)) + abs(src(i,j)-src(i-1,j)); %ԭͼ�ӿ�߽��������������صĻҶȲ���
           dfacter_dst=abs(dst(i,j)-dst(i+1,j)) + abs(dst(i,j)-dst(i-1,j));  %POSHEedͼ�ӿ�߽��������������صĻҶȲ���
           if (abs(dfacter_dst - dfacter_newsrc) > step_levels)
               %���ڿ�ЧӦִ��BERF
               b = 0;
               ave_bound=uint8( (uint16(dst(i - 1, j)) + uint16(dst(i + 1, j))) / 2 ); 
               dst(i,j)=ave_bound;
               %��ֱ���ӿ�߽磬���ϵ������� increasing rule��
               if (dst(i-1,j)> dst(i+1,j)) 
                   if (i - 2 - b >= 1)  %ͼ��߽��ж�(��һλ������)
                       pixel_present_add = dst(i - 1 - b, j);
                       pixel_next_add = dst(i - 2 - b, j);
                       pixel_present_add = ave_bound + step_levels;
                       while(i - 2 - b >= 1 && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir)
                           pixel_next_add = pixel_present_add + step_levels;
                           dst(i - 1 - b, j)=pixel_present_add;%%%%%%
                           dst(i - 2 - b, j)=pixel_next_add;%%%%%%
                           b =b+1;
                           if(i - 2 - b >= 1) %ͼ��߽��ж�(��һλ������)
                               pixel_present_add = dst(i - 1 - b, j);
                               pixel_next_add = dst(i - 2 - b, j);
                           end
                       end
                   end
                   
                   %��ֱ���ӿ�߽磬���µݼ����� decreasing rule��
                   b=0;
                   if (i + 2 + b <=dstrows)
                       pixel_present_dec = dst(i + 1 + b, j);
                       pixel_next_dec = dst(i + 2 + b, j);
                       pixel_present_dec = ave_bound - step_levels;
                       while (i + 2 + b <=dstrows && pixel_present_dec - pixel_next_dec >= step_levels && pixel_present_dec - pixel_next_dec <= large_dir)
                           pixel_next_dec = pixel_present_dec - step_levels;
                           dst(i + 1 + b, j)=pixel_present_dec;%%%%%%
                           dst(i + 2 + b, j)=pixel_next_dec;%%%%%%
                           b = b+1;
                           if (i + 2 + b <=dstrows) %ͼ��߽��ж�
                               pixel_present_dec = dst(i + 1 + b, j);
							   pixel_next_dec = dst(i + 2 + b, j);
                           end
                       end
                   end
               end
               
               %��ֱ���ӿ�߽磬���µ������� increasing rule��
               b=0;
                if (dst(i - 1, j) < dst(i + 1, j)) 
                     if (i + 2 + b <= dstrows) %ͼ��߽��ж�
                         pixel_present_add = dst(i + 1 + b, j);
                         pixel_next_add = dst(i + 2 + b, j);
                         pixel_present_add = ave_bound + step_levels;
                         while (i + 2 + b  <= dstrows && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir)
                             pixel_next_add = pixel_present_add + step_levels;
                             dst(i + 1 + b, j)=pixel_present_add;%%%%%%
                             dst(i + 2 + b, j)=pixel_next_add;%%%%%%
                             b =b + 1;
                             if (i + 2 + b  <= dstrows) %ͼ��߽��ж�
                                 pixel_present_add = dst(i + 1 + b, j);
                                 pixel_next_add = dst(i + 2 + b, j);
                             end
                         end
                     end
                     
                     %��ֱ���ӿ�߽磬���ϵݼ����� decreasing rule��
                     b=0;
                     if (i - 2 - b >= 1)
                         pixel_present_dec = dst(i - 1 - b, j);
                         pixel_next_dec = dst(i - 2 - b, j);
                         pixel_present_dec = ave_bound - step_levels;
                         while (i - 2 - b >= 1 && pixel_present_dec - pixel_next_dec >= step_levels && pixel_present_dec - pixel_next_dec <= large_dir)
                             pixel_next_dec = pixel_present_dec - step_levels;
                             dst(i - 1 - b, j)=pixel_present_dec;%%%%%%
                             dst(i - 2 - b, j)=pixel_next_dec;%%%%%%
                             b =b + 1;
                             if (i - 2 - b >= 1) %ͼ��߽��ж�
                                 pixel_present_dec = dst(i - 1 - b, j);
                                 pixel_next_dec = dst(i - 2 - b, j);
                             end
                         end
                     end
                end
               
           end
        end
    end
end


%�����ӿ���б߽磨������߽磩
for j=1:step_c:dstcols
    for i=1:dstrows
        if (j>1 && j<dstcols ) %ͼ��߽��жϣ����ų���һ�к����һ��
            %��ЧӦ���
           dfacter_newsrc=abs(src(i,j)-src(i,j+1)) + abs(src(i,j)-src(i,j-1)); %ԭͼ�ӿ�߽��������������صĻҶȲ���
           dfacter_dst=abs(dst(i,j)-dst(i,j+1)) + abs(dst(i,j)-dst(i,j-1));  %POSHEedͼ�ӿ�߽��������������صĻҶȲ���
           if (abs(dfacter_dst - dfacter_newsrc) > step_levels)
               %���ڿ�ЧӦִ��BERF
               b = 0;
               ave_bound=uint8( (uint16(dst(i, j-1)) + uint16(dst(i, j+1))) / 2 ); 
               dst(i,j)=ave_bound;
               
               %��ֱ���ӿ�߽磬����������� increasing rule��
               if (dst(i,j-1)> dst(i,j+1)) 
                   if (j - 2 - b >= 1)  %ͼ��߽��ж�(��һλ������)
                       pixel_present_add = dst(i,j - 1 - b);
                       pixel_next_add = dst(i,j - 2 - b);
                       pixel_present_add = ave_bound + step_levels;
                       while(j - 2 - b >= 1 && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir)
                           pixel_next_add = pixel_present_add + step_levels;
                           dst(i,j - 1 - b)=pixel_present_add;%%%%%%
                           dst(i,j - 2 - b)=pixel_next_add;%%%%%%
                           b =b+1;
                           if(j - 2 - b >= 1) %ͼ��߽��ж�(��һλ������)
                               pixel_present_add = dst(i,j - 1 - b);
                               pixel_next_add = dst(i,j - 2 - b);
                           end
                       end
                   end
                   
                   %��ֱ���ӿ�߽磬���ҵݼ����� decreasing rule��
                   b=0;
                   if (j + 2 + b <dstcols)
                       pixel_present_dec = dst(i,j + 1 + b);
                       pixel_next_dec = dst(i,j + 2 + b);
                       pixel_present_dec = ave_bound - step_levels;
                       while (i + 2 + b < dstcols && pixel_present_dec - pixel_next_dec >= step_levels && pixel_present_dec - pixel_next_dec <= large_dir)
                           pixel_next_dec = pixel_present_dec - step_levels;
                           dst(i,j + 1 + b)=pixel_present_dec;%%%%%%
                           dst(i,j + 2 + b)=pixel_next_dec;%%%%%%
                           b = b+1;
                           if (j + 2 + b <= dstcols) %ͼ��߽��ж�
                               pixel_present_dec = dst(i, j + 1 + b);
							   pixel_next_dec = dst(i, j + 2 + b);
                           end
                       end
                   end
               end
               
               %��ֱ���ӿ�߽磬���ҵ������� increasing rule��
               b=0;
                if(dst(i , j-1) < dst(i , j+1))  
                     if (j + 2 + b < dstcols) %ͼ��߽��ж�
                         pixel_present_add = dst(i, j + 1 + b);
                         pixel_next_add = dst(i, j + 2 + b);
                         pixel_present_add = ave_bound + step_levels;
                         while (j + 2 + b < dstcols && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir)
                             pixel_next_add = pixel_present_add + step_levels;
                             dst(i,j + 1 + b)=pixel_present_add;%%%%%%
                             dst(i,j + 2 + b)=pixel_next_add;%%%%%%
                             b =b + 1;
                             if (j + 2 + b  < dstcols) %ͼ��߽��ж�
                                 pixel_present_add = dst(i, j + 1 + b);
                                 pixel_next_add = dst(i, j + 2 + b);
                             end
                         end
                     end
                     
                     %��ֱ���ӿ�߽磬����ݼ����� decreasing rule��
                     b=0;
                     if (j - 2 - b >= 1)
                         pixel_present_dec = dst(i, j - 1 - b);
                         pixel_next_dec = dst(i, j - 2 - b);
                         pixel_present_dec = ave_bound - step_levels;
                         while (j - 2 - b >= 1 && pixel_present_dec - pixel_next_dec >= step_levels && pixel_present_dec - pixel_next_dec <= large_dir)
                             pixel_next_dec = pixel_present_dec - step_levels;
                             dst(i,j - 1 - b)=pixel_present_dec;%%%%%%
                             dst(i,j - 2 - b)=pixel_next_dec;%%%%%%
                             b =b + 1;
                             if (j - 2 - b >= 1) %ͼ��߽��ж�
                                 pixel_present_dec = dst(i, j - 1 - b);
                                 pixel_next_dec = dst(i, j - 2 - b);
                             end
                         end
                     end
                 end
               
           end
        end
    end
end

