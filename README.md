# ZTETuring2021

数据：
原始数据raw.yuv， 格式YUV44P，分辨率1280*720
压缩方式：
SVD奇异值分解
压缩思路：
已知分解出的sigma对角线矩阵前10%对角线元素包含99%以上的矩阵信息，即
假设原数据矩阵为m*n矩阵，分解后得到m*m左奇异向量u，m*n奇异值对角线矩阵sigma，n*n右奇异向量v，若对角线矩阵取前k个对角线元素，即只需保存m*k矩阵u，k*k矩阵sigma，k*n矩阵n，在转码时将其相乘得到包含原矩阵大部分信息的数据矩阵，此时矩阵信息由m*n下降为k*（m+n+k）；进一步，由于sigma为对角线矩阵，只需保存对角线元素即可，需保存信息进一步下降为k*（m+n+1），并使用numpy.savez_compressed将三个矩阵打包压缩

