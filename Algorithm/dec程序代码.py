import numpy as np
import os
import copy

def write(mtr_ori,fp):
    #mtr_ori=np.rint(mtr_ori).astype('uint8')
    for i in range(mtr_ori.shape[0]):
        for j in range(mtr_ori.shape[1]):

            #print(type(mtr_ori[i][j]))
            fp.write(mtr_ori[i][j])
def getori(u,s,v):
    s=np.diag(s)
    #print(u.shape,s.shape,v.shape)
    return u.dot(s).dot(v)

if __name__=='__main__':
    filepath='D:/ZTE/data/enc/LiuHui_CUHK_0'
    encpath='D:/ZTE/data/dec/LiuHui_CUHK_0'
    p=0.15
    height=720
    width=1280
    r=int(p*min(height,width))
    for k in range(1,5):
        os.rename(filepath+'{}_enc.code'.format(k),filepath+'.npz')
        ff=np.load(filepath+'.npz')
        height=720
        width=1280
        #print(ff['arr_0'].shape)
        u=copy.deepcopy(ff['u'])
        s=copy.deepcopy(ff['s'])
        vt=copy.deepcopy(ff['v'])
        #print(u)
        totalframe=(u.shape[0]-1)/(height*3)
        print('totalframe: ',totalframe)
        ff.close()
        with open(encpath+'{}_dec.yuv'.format(k),'wb') as fp:
            #set cur strip 0
            cur_u,cur_s,cur_v=1,1,1
            #dec by frame
            for i in range(int(totalframe)):
                print('{} frame enc start'.format(i))
                for j in range(3):
                    y_ori=getori(u[cur_u:cur_u+height,:],s[cur_s:cur_s+r],vt[cur_v:cur_v+r,:])
                    y_ori=np.rint(y_ori).astype('uint8')
                    write(y_ori,fp)
                    cur_u+=height
                    cur_s+=r
                    cur_v+=r
        fp.close()
        os.rename(filepath+'.npz',filepath+'{}_enc.code'.format(k))
        print('enc {} finished!'.format(k))