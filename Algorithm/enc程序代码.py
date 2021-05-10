import numpy as np
import copy
import os
#yuv,1280*720,4:4:4
def yuvRd(file_path,height,width,start_frame):
    #read binary
    fp=open(file_path,'rb')
    #get total length
    fp.seek(0,2)
    fp_total=fp.tell()
    frame_size=height*width*3
    total_frame=fp_total/frame_size
    print('the yuv has {} frames.'.format(total_frame))
    fp.seek(frame_size*start_frame,0)
    total_value=min(width,height)+1
    p=0.15
    #compute r
    r=int(p*total_value)
    #u,m(h)*r
    yuv_u=np.zeros(r).reshape(-1,r)
    yuv_sigma=np.zeros(1)
    yuv_vt=np.zeros(width).reshape(-1,width)
    for i in range(int(total_frame)):
        frame_y=np.zeros(height*width)
        frame_u=copy.deepcopy(frame_y)
        frame_v=copy.deepcopy(frame_y)
        print('{} frame start!'.format(i))
        for j in range(height*width):
            frame_y[j]=ord(fp.read(1))
        for j in range(height*width):
            frame_u[j]=ord(fp.read(1))
        for j in range(height*width):
            frame_v[j]=ord(fp.read(1))
        
        y_u,y_sigma,y_vt=svd_reduce(frame_y.reshape(height,width),r)
        u_u,u_sigma,u_vt=svd_reduce(frame_u.reshape(height,width),r)
        v_u,v_sigma,v_vt=svd_reduce(frame_v.reshape(height,width),r)
        print('{} yuv computed.'.format(i))
        #np.savetxt('test01.code',ss,delimiter=',')
        #append by rows
        axis=0
        uu=np_append(y_u,u_u,v_u,axis)
        ss=np_append(y_sigma,u_sigma,v_sigma,axis)
        vv=np_append(y_vt,u_vt,v_vt,axis)
        uu=np.rint(uu).astype('uint8')
        ss=np.rint(ss).astype('uint8')
        vv=np.rint(vv).astype('uint8')
       # print(uu.shape,ss.shape,vv.shape)
        yuv_u=np.append(yuv_u,uu,axis=0)
        yuv_sigma=np.append(yuv_sigma,ss,axis=0)
        yuv_vt=np.append(yuv_vt,vv,axis=0)

    return yuv_u,yuv_sigma,yuv_vt

def np_append(y,u,v,axis):
    matr=np.append(y,u,axis=axis)
    matr=np.append(matr,v,axis=axis)
    return matr

def svd_reduce(fmetrics,r):
    #get 3 matrix,(m,r),(r,r),(r,n)
    #print('fme:',fmetrics.shape)
    u, sigma, vt=np.linalg.svd(fmetrics,full_matrices=False,compute_uv=True)
    #total=min(m-1,n-1)+1
    
    sigma=sigma[:r]
    u=u[:,:r]
    vt=vt[:r,:]
    #print(r,sigma.shape,u.shape)
    return u,sigma,vt

if __name__=='__main__':
    #print(sys.path)
    savepath='D:/ZTE/data/enc/LiuHui_CUHK_0'
    #os.mknod(savepath)
    filepaths=['D:/ZTE/data/excel_01.yuv','D:/ZTE/data/ppt_02.yuv','D:/ZTE/data/web_03.yuv','D:/ZTE/data/word_04.yuv']
    height=720
    width=1280
    i=0
    for filepath in filepaths:
        i+=1
        yuv_u,yuv_s,yuv_v=yuvRd(file_path=filepath,height=height,width=width,start_frame=0)
        #yuv_svd=np.rint(yuv_svd).astype('uint8')
        yuv_u=np.rint(yuv_u).astype('uint8')
        yuv_s=np.rint(yuv_s).astype('uint8')
        yuv_v=np.rint(yuv_v).astype('uint8')
        np.savez_compressed(savepath,u=yuv_u,s=yuv_s,v=yuv_v,delimiter=',')
        os.rename(savepath+'.npz',savepath+'{}_enc.code'.format(i))
        print('con! {} finished.'.format(i))
        

