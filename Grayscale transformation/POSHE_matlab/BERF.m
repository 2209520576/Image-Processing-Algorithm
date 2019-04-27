%% This is a function of BERF.
%@ date ： 2019/4/27
%@ author ： Xiaowu
%@ CSDN ：https://blog.csdn.net/weixin_40647819/article/details/88416512
%@ github ：https://github.com/2209520576/Image-Processing-Algorithm
%///////////////////////////////////////////////////
%BERF滤波器
%src - 输入图像
%dst - 目标图像
%step_r  - 子块垂直移动步长
%step_c  - 子块水平移动步长
%step_levels - 灰度级缓慢变化的步长
%large_dir  - 最大灰度差异
%//////////////////////////////////////////////////
%%
function dst=BERF(src, dst,step_r, step_c, step_levels, large_dir)
[dstrows,dstcols]=size(dst);

%处理子块的行边界（横向边界）
for i=1:step_r:dstrows
    for j=1:dstcols
        if (i>1 && i<dstrows ) %图像边界判断，先排除第一行和最后一行
            %块效应检查
           dfacter_newsrc=abs(src(i,j)-src(i+1,j)) + abs(src(i,j)-src(i-1,j)); %原图子块边界与相邻两侧像素的灰度差异
           dfacter_dst=abs(dst(i,j)-dst(i+1,j)) + abs(dst(i,j)-dst(i-1,j));  %POSHEed图子块边界与相邻两侧像素的灰度差异
           if (abs(dfacter_dst - dfacter_newsrc) > step_levels)
               %存在块效应执行BERF
               b = 0;
               ave_bound=uint8( (uint16(dst(i - 1, j)) + uint16(dst(i + 1, j))) / 2 ); 
               dst(i,j)=ave_bound;
               %垂直于子块边界，向上递增方向（ increasing rule）
               if (dst(i-1,j)> dst(i+1,j)) 
                   if (i - 2 - b >= 1)  %图像边界判断(下一位置像素)
                       pixel_present_add = dst(i - 1 - b, j);
                       pixel_next_add = dst(i - 2 - b, j);
                       pixel_present_add = ave_bound + step_levels;
                       while(i - 2 - b >= 1 && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir)
                           pixel_next_add = pixel_present_add + step_levels;
                           dst(i - 1 - b, j)=pixel_present_add;%%%%%%
                           dst(i - 2 - b, j)=pixel_next_add;%%%%%%
                           b =b+1;
                           if(i - 2 - b >= 1) %图像边界判断(下一位置像素)
                               pixel_present_add = dst(i - 1 - b, j);
                               pixel_next_add = dst(i - 2 - b, j);
                           end
                       end
                   end
                   
                   %垂直于子块边界，向下递减方向（ decreasing rule）
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
                           if (i + 2 + b <=dstrows) %图像边界判断
                               pixel_present_dec = dst(i + 1 + b, j);
							   pixel_next_dec = dst(i + 2 + b, j);
                           end
                       end
                   end
               end
               
               %垂直于子块边界，向下递增方向（ increasing rule）
               b=0;
                if (dst(i - 1, j) < dst(i + 1, j)) 
                     if (i + 2 + b <= dstrows) %图像边界判断
                         pixel_present_add = dst(i + 1 + b, j);
                         pixel_next_add = dst(i + 2 + b, j);
                         pixel_present_add = ave_bound + step_levels;
                         while (i + 2 + b  <= dstrows && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir)
                             pixel_next_add = pixel_present_add + step_levels;
                             dst(i + 1 + b, j)=pixel_present_add;%%%%%%
                             dst(i + 2 + b, j)=pixel_next_add;%%%%%%
                             b =b + 1;
                             if (i + 2 + b  <= dstrows) %图像边界判断
                                 pixel_present_add = dst(i + 1 + b, j);
                                 pixel_next_add = dst(i + 2 + b, j);
                             end
                         end
                     end
                     
                     %垂直于子块边界，向上递减方向（ decreasing rule）
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
                             if (i - 2 - b >= 1) %图像边界判断
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


%处理子块的行边界（纵向向边界）
for j=1:step_c:dstcols
    for i=1:dstrows
        if (j>1 && j<dstcols ) %图像边界判断，先排除第一列和最后一列
            %块效应检查
           dfacter_newsrc=abs(src(i,j)-src(i,j+1)) + abs(src(i,j)-src(i,j-1)); %原图子块边界与相邻两侧像素的灰度差异
           dfacter_dst=abs(dst(i,j)-dst(i,j+1)) + abs(dst(i,j)-dst(i,j-1));  %POSHEed图子块边界与相邻两侧像素的灰度差异
           if (abs(dfacter_dst - dfacter_newsrc) > step_levels)
               %存在块效应执行BERF
               b = 0;
               ave_bound=uint8( (uint16(dst(i, j-1)) + uint16(dst(i, j+1))) / 2 ); 
               dst(i,j)=ave_bound;
               
               %垂直于子块边界，向左递增方向（ increasing rule）
               if (dst(i,j-1)> dst(i,j+1)) 
                   if (j - 2 - b >= 1)  %图像边界判断(下一位置像素)
                       pixel_present_add = dst(i,j - 1 - b);
                       pixel_next_add = dst(i,j - 2 - b);
                       pixel_present_add = ave_bound + step_levels;
                       while(j - 2 - b >= 1 && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir)
                           pixel_next_add = pixel_present_add + step_levels;
                           dst(i,j - 1 - b)=pixel_present_add;%%%%%%
                           dst(i,j - 2 - b)=pixel_next_add;%%%%%%
                           b =b+1;
                           if(j - 2 - b >= 1) %图像边界判断(下一位置像素)
                               pixel_present_add = dst(i,j - 1 - b);
                               pixel_next_add = dst(i,j - 2 - b);
                           end
                       end
                   end
                   
                   %垂直于子块边界，向右递减方向（ decreasing rule）
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
                           if (j + 2 + b <= dstcols) %图像边界判断
                               pixel_present_dec = dst(i, j + 1 + b);
							   pixel_next_dec = dst(i, j + 2 + b);
                           end
                       end
                   end
               end
               
               %垂直于子块边界，向右递增方向（ increasing rule）
               b=0;
                if(dst(i , j-1) < dst(i , j+1))  
                     if (j + 2 + b < dstcols) %图像边界判断
                         pixel_present_add = dst(i, j + 1 + b);
                         pixel_next_add = dst(i, j + 2 + b);
                         pixel_present_add = ave_bound + step_levels;
                         while (j + 2 + b < dstcols && pixel_next_add - pixel_present_add >= step_levels && pixel_next_add - pixel_present_add <= large_dir)
                             pixel_next_add = pixel_present_add + step_levels;
                             dst(i,j + 1 + b)=pixel_present_add;%%%%%%
                             dst(i,j + 2 + b)=pixel_next_add;%%%%%%
                             b =b + 1;
                             if (j + 2 + b  < dstcols) %图像边界判断
                                 pixel_present_add = dst(i, j + 1 + b);
                                 pixel_next_add = dst(i, j + 2 + b);
                             end
                         end
                     end
                     
                     %垂直于子块边界，向左递减方向（ decreasing rule）
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
                             if (j - 2 - b >= 1) %图像边界判断
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

