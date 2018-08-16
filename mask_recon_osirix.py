import cv2
import scipy.ndimage as nd
import scipy
import numpy as np
import os
import scipy.misc
import pydicom
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import scipy.ndimage as nd
import scipy
import numpy as np
import os
import scipy.misc
import pydicom
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import pandas as pd
def int_fn(point_list,img_mask):
    count = 0
    #point_list.append(point_list[-2])
    #point_list.append(point_list[-1])
    rev_point_list = point_list[3:]+point_list[:3]
    rev_point_list = np.array(rev_point_list)
    
    rev_point_list1 = point_list[1:]+point_list[:1]
    rev_point_list1 = np.array(rev_point_list1)
    
    rev_point_list2 = point_list[4:]+point_list[:4]
    rev_point_list2 = np.array(rev_point_list2)
    
    rev_point_list3 = point_list[2:]+point_list[:2]
    rev_point_list3 = np.array(rev_point_list3)
    
    rev_point_list4 = point_list[::-1][3:]+point_list[::-1][:3]
    rev_point_list4 = np.array(rev_point_list4)
    
    #point_list=np.array(point_list).astype(np.int64)
    
    rev_points = [point_list,rev_point_list,rev_point_list1,rev_point_list2,rev_point_list3,rev_point_list4]
    
    #rev_point_list = point_list[2:]+point_list[:2]
    #x = point_list[:,1]
    #y = point_list[:,0]
    
    #z = np.polyfit(x,y,3)
    #f = np.poly1d(z)
    
    #x_new = np.linspace(x[0],x[-1],300)
    #y_new = f(x_new)
    
    #print(y_new)
    #plt.plot(x_new,y_new),plt.show()
    #img_mask[y_new.astype(int),x_new.astype(int)]=255
    #plt.imshow(img_mask),plt.show()
    #return img_mask
    #print('len list)')
    #print(len(point_list))
    #print(point_list)
    
    #print(point_list)
    
    #new_point_list = 
    
    #return
    
    if(len(point_list)<=5):
        #print(point_list)
        #print(rev_point_list)
        
        for point_list_1 in rev_points:
            
            for i in range(0,len(point_list_1[:-1])):
                point1 = point_list_1[i]
                point2 = point_list_1[i+1]
        #        point3 = point_list[i+2]

                y1 = int(point1[0])
                x1 = int(point1[1])
                y2 = int(point2[0])
                x2 = int(point2[1])

    #             img_mask[min(x1,x2):max(x1,x2),\
    #                  min(y1,y2):max(y1,y2)] = 255
                
                img_mask[x1,min(y1,y2):max(y1,y2)]=255
                img_mask[x2,min(y1,y2):max(y1,y2)]=255
                img_mask[min(x1,x2):max(x1,x2),y1]=255
                img_mask[min(x1,x2):max(x1,x2),y2]=255
                
                #scipy.misc.imsave('',img_mask)
                
            point1 = point_list_1[-1]
            point2 = point_list_1[0]
            y1 = int(point1[0])
            x1 = int(point1[1])
            y2 = int(point2[0])
            x2 = int(point2[1])

            #             img_mask[min(x1,x2):max(x1,x2),\
            #                  min(y1,y2):max(y1,y2)] = 255

            img_mask[x1,min(y1,y2):max(y1,y2)]=255
            img_mask[x2,min(y1,y2):max(y1,y2)]=255
            img_mask[min(x1,x2):max(x1,x2),y1]=255
            img_mask[min(x1,x2):max(x1,x2),y2]=255

            
    else:
        count=0
        #print(point_list)
        
        #point_list = sorted(point_list,key=lambda x:x[0])
        
        #x_points =[(i[0]) for i in point_list]
        #y_points =[(i[1]) for i in point_list]
        #print(x_points)
        #print(y_points)
        #tck = scipy.interpolate.splrep(x_points,y_points)
        new_p = np.array([list(i) for i in point_list])
        tck,u = splprep(new_p.T,u=None,s=0.0,per=1)
        u_new = np.linspace(u.min(),u.max(),1000)
        x_new,y_new = splev(u_new,tck,der=0)
        
#         for point_list_1 in rev_points:
#             #print(point_list_1)
#             deg = 4
#             #x_points = [(i[0]) for i in point_list]
#             #y_points = [(i[1]) for i in point_list]
# #             print(x_points)
# #             print(y_points)
# #             print(len(x_points))
# #             print(len(y_points))
#             #tck = scipy.interpolate.splrep(x_points,y_points)
            
#             for i in range(0,len(point_list_1[:-1])):
#                 point1 = point_list_1[i]
#                 #print(point1)
#                 point2 = point_list_1[i+1]
#         #        point3 = point_list[i+2]
#                 #print(point2)
#                 y1 = int(point1[0])
#                 x1 = int(point1[1])
#                 y2 = int(point2[0])
#                 x2 = int(point2[1])
                
#                 if x1==x2:
#                     x2=x2+1
#                 if(y1==y2):
#                     y2=y2+1
                
#                 #x3 =int(point3[0])
#                 #y3 =int(point3[1])

#                 z = np.polyfit(np.array([x1,x2]),np.array([y1,y2]),deg)
#                 f = np.poly1d(z)
                
#                 x_new = np.linspace(min(x1,x2),max(x1,x2),500)
#                 y_new = f(x_new)
#                 #print(x_new)
#                 #print(y_new)
#                 #print(x_new)
#                 #print(y_new)
                
#                 coord_arr = np.zeros((x_new.shape[0],2))
#                 coord_arr[:,0]=y_new.astype(int)
#                 coord_arr[:,1]=x_new.astype(int)
                
#                 #np.savetxt('/data/gabriel/Osirix/dest2_test/'+str(count)+'.txt',coord_arr)
#                 #x_new = 
#                 #y_new = 
#                 #img_mask[point1[0],point1[1]] = 255

#                 #img_mask[point2[0],point2[1]] = 255
#                 img_mask[x_new.astype(np.uint64),y_new.astype(np.uint64)]=255
#                 #scipy.misc.imsave('/data/gabriel/Osirix/dest2_test/'+str(count)+'.png',img_mask)
                
#                 count+=1
                
#             point1 = point_list_1[-1]
#             point2 = point_list_1[0]
#             y1 = int(point1[0])
#             x1 = int(point1[1])
#             y2 = int(point2[0])
#             x2 = int(point2[1])
#             img_mask[x1,min(y1,y2):max(y1,y2)]=255
#             img_mask[x2,min(y1,y2):max(y1,y2)]=255
#             img_mask[min(x1,x2):max(x1,x2),y1]=255
#             img_mask[min(x1,x2):max(x1,x2),y2]=255
            #x3 =int(point3[0])
            #y3 =int(point3[1])

            #z = np.polyfit(np.array([x1,x2]),np.array([y1,y2]),deg)
            #f = np.poly1d(z)
            
            
            #x_new = np.linspace(min(x1,x2),max(x1,x2),500)
            #x_new = np.linspace(min(x_points),max(x_points),500)
            #y_new = scipy.interpolate.splev(x_new,tck)
            #y_new = f(x_new)

            #print(x_new)
#             img_mask[point1[0],point1[1]] = 255

#             img_mask[point2[0],point2[1]] = 255
#             coord_arr = np.zeros((x_new.shape[0],2))
#             coord_arr[:,0]=y_new.astype(int)
#             coord_arr[:,1]=x_new.astype(int)

#             np.savetxt('/data/gabriel/Osirix/dest2_test/'+str(count)+'.txt',coord_arr)

#             #print(y_new)
#             #print(x_new)
#             #print(y_new)
#             img_mask[x_new.astype(np.uint64),y_new.astype(np.uint64)]=255
            #scipy.misc.imsave('/data/gabriel/Osirix/dest2_test/'+str(count)+'.png',img_mask)
            #count+=1
        img_mask[y_new.astype(np.uint32),x_new.astype(np.uint32)] = 255.0
            
    scipy.misc.imsave('temp.png',img_mask)
    im=cv2.imread('temp.png',0)
    im_floodfill=im.copy()
    h,w = im.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(im_floodfill,mask,(0,0),255)
    #plt.imshow(im_floodfill),plt.show()
    im_floodfill_inv= cv2.bitwise_not(im_floodfill)
    im_out = im|im_floodfill_inv
    #return img_mask
    return im_out 
    #return img_mask 
import csv
def xl_to_anno(path_xl,dicom_path,dest_dir,mode=True):
    ### path of png files corresponding to dicom
    ### path of excel file
    

    row_len  = []
    with open(path_xl,'r',newline='') as csvfile:
        spamwriter = csv.reader(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        for row in spamwriter:
            #print(row)
            row_len.append(len(row))
    
    
    df = pd.read_csv(path_xl,header=0,\
                     usecols=np.arange(0,max(row_len)),\
                     names=np.arange(0,max(row_len)))
    
    new_dic = {}
    count=0
    for k in df:
        if(k==0):
            new_dic[count] = df[k]
            count+=1

        if(k==7):
            new_dic[count] = df[k]
            count+=1
        if(k%5==3 and k>20):
            new_dic[count] = df[k]
            count+=1
        if(k%5==4 and k>20):
            new_dic[count] = df[k]
            count+=1
    
    new_dic = pd.DataFrame(new_dic)
    frame_num=0
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    for dic_row in np.array(new_dic):
        #print(type(dic_row[0]))
        if(mode ):
            #mode =
            # mode True == calcium score
            #frame_num = dic_row[0]+1
            frame_num = dic_row[0] + 1
            #frame_num = 4
            coords = [(min((dic_row[i]),511),min((dic_row[i+1]),511)) for i in range(2,dic_row.shape[0]-1,2) if not np.isnan(dic_row[i+1])]
        else:
            # mode False == echo annotations
            frame_num = dic_row[0]
            coords = [(min((dic_row[i]),511),min((dic_row[i+1]),511)) for i in range(2,dic_row.shape[0]-1,2) if not np.isnan(dic_row[i])]
            
        #print(frame_num)
        
        roi_name = str(dic_row[1])
        #print(dic_row)
        #print(dic_row.shape)
        #print(type(dic_row))
        
#         for i in range(1,dic_row.shape[0]-1):
#             if not np.isnan(dic_row[i]):
#                 print(dic_row[i])
        #print(coords)
        #break
        
        #print(coords)
        #print(coords)
        dcm_file = pydicom.dcmread(dicom_path)
        #print(coords)
        
        
        if not os.path.isfile(dest_dir+'/'+roi_name+'/'+'IMG_'+str(frame_num)+'.png'):
            #print(coords)
            output_mask = int_fn(coords,np.zeros((512,512)))
        else:
            img_mask = plt.imread(dest_dir+'/'+roi_name+'/'+'IMG_'+str(frame_num)+'.png')
            thresh = threshold_otsu(img_mask)
            binary = 255*(img_mask>thresh).astype(np.int64)
            
            output_mask = int_fn(coords,plt.imread(dest_dir+'/'+roi_name+'/'+'IMG_'+str(frame_num)+'.png'))
            
            #plt.imshow(output_mask),plt.show()
            #plt.imshow(),plt.show()
        #print(output_mask.shape)
        if not os.path.isdir((dest_dir+'/'+roi_name)):
            os.makedirs(dest_dir+'/'+roi_name)
        
        scipy.misc.imsave(dest_dir+'/'+roi_name+'/'+'IMG_'+str(frame_num)+'.png',output_mask)
        #frame_num+=1
        
        
        #img = plt.imread('/data/Gurpreet/Projects/Echo_Annotation/F1_4/IMG_'+str(slice_num)+'.jpg')
        
    #             except:
    #                 continue