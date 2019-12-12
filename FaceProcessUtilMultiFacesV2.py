####
#Source codes of the paper 'Facial Expression Recognition via Region-based Convolutional Fusion Network'
# submitted to Neurocomputing for the purpose of review.
# Platform windows 10 with Tensorflow 1.2.1, CUDA 8.0, python 3.5.2 64 bit, MSC v.1900 64bit,
# Untrimmed source file with irrelevant methods without prejudice to the current RCFN setting.
###################
import math
from datetime import datetime
import cv2, os
import dlib
import numpy as np
from PIL import Image as IM
from scipy import ndimage
import time
import pywt

scale_factor=1.0/255.0

rescaleImg = [1.4504, 1.5843, 1.4504, 1.3165]
mpoint = [63.78009, 41.66620]
target_size = 128

inner_width = 1.3
inner_up_height = 1.3
inner_down_height = 3.2
inner_face_size= (72,96)
#inner_face_size= (48,36)

finalsize=(224,224)
#eyelog='eyecenterlogv4.txt'

'''all three patches must be the same size if they are to be concatenated in the patch network'''
eye_patch_size = (64, 26)
eye_height_width_ratio = 0.40625 #alternative 0.5-0.6 including forehead
#middle_height_width_ratio = 1.25
#middle_patch_size = (32, 40)
middle_height_width_ratio = 1.75
middle_patch_size = (28, 49)
mouth_width_height_ratio = 1.8
mouth_patch_size = (54, 30)

##LMP1_keys=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 27]
#LMP1_keys=[17, 19, 21, 26, 24, 22, 37, 41, 44, 46, 27]
LMP1_Dists=[[37,41],[41,19],[37,19],[41,21],[37,21],[39,19],[39,21],
                      [44,46],[44,24],[46,24],[44,22],[46,22],[42,24],[42,22]]
                      #[21,22],[21,27],[22,27]] not helpful
LMP1_triangle=[[17, 19, 21], [17, 19, 27], [27, 37, 41], [41, 19, 27], [41, 17, 21], [41, 21, 27], [17, 41, 27], 
                           [26, 24, 22], [26, 24, 27], [27, 44, 46], [46, 24, 27], [46, 26, 22], [46, 22, 27], [26, 46, 27]]
#LMP1_triangle=[[17, 19, 21], [17, 19, 27], [27, 37, 41], [41, 19, 27], [41, 17, 21], [27, 21, 41], [17, 41, 27], 
#                           [26, 24, 22], [26, 24, 27], [27, 44, 46], [46, 24, 27], [46, 26, 22], [27, 22, 46], [26, 46, 27]]
#LMP1_aug=[[37,38],[37,39],[38,39],
#                    [44,43],[44,42],[33,42]]  #not helpful
#LMP1_aug=[[17,19],[19,21],
#                    [26,24],[24,22]]

#LMP2_keys=[4, 5, 48, 12, 11, 54, 49, 59, 51, 57, 53, 55, 62, 66] 
LMP2_Dists=[[51,57],[62,66],[49,59],[53,55]]
                      #[48,49],[48,51],[49,51], not helpful
                      #[54,53],[54,51],[53,51]] not helpful
'''the order of three points in a triangle should be considered
 by making the outcoming triangle features share bigger variance'''
#LMP2_triangle=[[4, 5, 48], [12, 11, 54], [62, 66, 48], [62, 66, 54],
#                           [4, 5, 66], [12, 11, 66], [4, 5, 51], [12, 11, 51],[4, 5, 62], [12, 11, 62]
#                           , [51, 57, 48], [51, 57, 54], [4, 5, 57], [12, 11, 57]]
LMP2_triangle=[[4, 5, 48], [12, 11, 54], [62, 66, 48], [62, 66, 54],
                           [4, 5, 66], [12, 11, 66], [4, 5, 51], [12, 11, 51],[4, 5, 62], [12, 11, 62]
                           , [51, 57, 48], [51, 57, 54], [4, 5, 57], [12, 11, 57]
                           ,[62,66,12],[62,66,4],[62,66,11],[62,66,5]]
#LMP2_triangle=[[48, 5, 4], [54, 11, 12], [48, 57, 51], 
#                           [54, 57, 51], [48, 66, 62], [54, 66, 62],
#                           [66, 5, 4], [66, 11, 12], [51, 4, 5], [51, 11, 12],
#                           [62, 5, 4], [62, 11, 12], [57, 5, 4], [57, 11, 12]]
#LMP2_aug=[[48,51], #not helpful
#                    [54,51]]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./dlibmodel/shape_predictor_68_face_landmarks.dat")
#
#
######### formalization the image
def __formalizeImg(in_img):
    return (in_img-np.mean(in_img))/np.std(in_img)

#
#
######### get Patches images
def __getEyePatch(img, w, h, X=None, Y=None):
    '''Return the eye patch image from the rescaled images'''
    if X is None or Y is None:###use default configs for unexpected cases. this could be strongly unstable.
        w_patch = int(round(w*0.734375))
        h_patch = int(round(w_patch*eye_height_width_ratio))
        bry = int(mpoint[1]+round(h_patch*0.5))
        tlx = int(round(w-w_patch)*0.5)
        brx = tlx+w_patch
        tly = bry-h_patch
    else:
        half_width = int(max((X[27]-(X[0]+X[17])/2), ((X[16]+X[26])/2-X[27])))
        w_patch = 2*half_width
        h_patch = int(round(w_patch*eye_height_width_ratio))
        cx = np.mean(np.concatenate([X[17:27],X[36:48]]))
        cy = np.mean(np.concatenate([Y[17:27],Y[36:48]]))
        tly = int(round(cy-h_patch/2))
        bry = tly+h_patch
        tlx = int(round(cx-half_width))
        brx = tlx + w_patch
    
    #patch = np.zeros((h_patch, w_patch, 3), dtype = "uint8")
    patch = np.zeros((h_patch, w_patch), dtype = "uint8")
    blxstart = 0
    if tlx < 0:
        blxstart = -tlx
        tlx = 0
    brxend = w_patch
    if brx > w:
        brxend = w + w_patch - brx#brxend=w_patch-(brx-w)
        brx = w
    btystart = 0
    if tly < 0:
        btystart = -tly
        tly = 0
    bbyend = h_patch
    if bry > h:
        bbyend = h + h_patch - bry#bbyend=h_patch-(bry-h)
        bry = h
    
    #patch[btystart:bbyend,blxstart:brxend,:] = img[tly:bry,tlx:brx,:]
    patch[btystart:bbyend,blxstart:brxend] = img[tly:bry,tlx:brx]
    patch = cv2.resize(patch, eye_patch_size)
    patch = np.multiply(patch, scale_factor)
    patch = np.reshape(patch, [patch.shape[0], patch.shape[1], 1])
    return patch

def __getMiddlePatch(img, w, h, X=None, Y=None):
    '''Return the middle patch image from the rescaled images'''
    if X is None or Y is None:###use default configs for unexpected cases. this could be strongly unstable.
        w_patch = int(round(w*0.25))
        h_patch = int(round(w_patch*middle_height_width_ratio))
        bry = int(round(h*0.40625))# starts at 52th row in 128x128
        tlx = int(round(w-w_patch)*0.5)
        brx = tlx+w_patch
        tly = bry-h_patch
    else:
        tlx = int(X[39])
        brx = int(X[42])
        w_patch = brx - tlx
        bry = int((Y[28]+Y[29])/2)
        h_patch = int(round(w_patch*middle_height_width_ratio))
        tly = bry - h_patch
    #patch = np.zeros((h_patch, w_patch, 3), dtype = "uint8")
    patch = np.zeros((h_patch, w_patch), dtype = "uint8")
    
    blxstart = 0
    if tlx < 0:
        blxstart = -tlx
        tlx = 0
    brxend = w_patch
    if brx > w:
        brxend = w + w_patch - brx#brxend=w_patch-(brx-w)
        brx = w
    btystart = 0
    if tly < 0:
        btystart = -tly
        tly = 0
    bbyend = h_patch
    if bry > h:
        bbyend = h + h_patch - bry#bbyend=h_patch-(bry-h)
        bry = h
    #patch[btystart:bbyend,blxstart:brxend,:] = img[tly:bry,tlx:brx,:]
    patch[btystart:bbyend,blxstart:brxend] = img[tly:bry,tlx:brx]
    patch = cv2.resize(patch, middle_patch_size)
    patch = np.multiply(patch, scale_factor)
    patch = np.reshape(patch, [patch.shape[0], patch.shape[1], 1])
    return patch

def __getMouthPatch(img, w, h, X=None, Y=None):
    '''Return the mouth patch image from the rescaled images'''
    if X is None or Y is None:###use default configs for unexpected cases. this could be strongly unstable.
        w_patch = int(round(w*0.421875))
        h_patch = int(round(w_patch/mouth_width_height_ratio))
        tly = int(round(h*0.59375))#starts at 76th row in 128x128
        tlx = int(round(w-w_patch)*0.5)
        brx = tlx+w_patch
        bry = tly+h_patch
    else:
        up_h_patch = int(round(Y[8]-Y[33])/3)#upheight, two of the five
        w_patch = int(up_h_patch*2.25)#half width=up_h_patch*2.5*mouth_width_height_ratio/2=up_h_patch*
        cx=int(np.mean(X[48:68]))
        cy=int(np.mean(Y[48:68]))
        tly = cy-up_h_patch
        tlx = cx-w_patch
        h_patch = int(round(up_h_patch*2.5))#
        w_patch = w_patch*2
        bry=tly+h_patch
        brx=tlx+w_patch

    #patch = np.zeros((h_patch, w_patch, 3), dtype = "uint8")
    patch = np.zeros((h_patch, w_patch), dtype = "uint8")

    blxstart = 0
    if tlx < 0:
        blxstart = -tlx
        tlx = 0
    brxend = w_patch
    if brx > w:
        brxend = w + w_patch - brx#brxend=w_patch-(brx-w)
        brx = w
    btystart = 0
    if tly < 0:
        btystart = -tly
        tly = 0
    bbyend = h_patch
    if bry > h:
        bbyend = h + h_patch - bry#bbyend=h_patch-(bry-h)
        bry = h
    #patch[btystart:bbyend,blxstart:brxend,:] = img[tly:bry,tlx:brx,:]
    patch[btystart:bbyend,blxstart:brxend] = img[tly:bry,tlx:brx]
    patch = cv2.resize(patch, mouth_patch_size)
    patch = np.multiply(patch, scale_factor)
    patch = np.reshape(patch, [patch.shape[0], patch.shape[1], 1])
    return patch

######
#
#return the croppedface with the inner_face_size defined at the start of this file
def __getCroppedFace(img, w, h):
    '''Return the croppedface with the inner_face_size defined at the start of this page.
    you need to change the operation before resize() to get the different types of croppedface'''
    tly = 16
    tlx = 28
    h_patch = 96
    w_patch =72
    brx = tlx+w_patch
    bry = tly+h_patch
    patch = np.zeros((h_patch, w_patch), dtype = "uint8")
    
    blxstart = 0
    if tlx < 0:
        blxstart = -tlx
        tlx = 0
    brxend = w_patch
    if brx > w:
        brxend = w + w_patch - brx#brxend=w_patch-(brx-w)
        brx = w
    btystart = 0
    if tly < 0:
        btystart = -tly
        tly = 0
    bbyend = h_patch
    if bry > h:
        bbyend = h + h_patch - bry#bbyend=h_patch-(bry-h)
        bry = h
    patch[btystart:bbyend,blxstart:brxend] = img[tly:bry,tlx:brx]

    '''change the method down here'''
    #patch=__weberface(patch,revers=True) #reverse version
    #patch=None
    #patch=img[:,:]
    #patch=__weberface(patch)
    #patch=patch*1.25#25% up
    #patch=np.clip(patch,1,255)#clip the values into [1,255]

    #patch=__ELTFS(patch)
    '''change the method up here to get the croppedface that you want'''

    #patch=cv2.resize(patch,inner_face_size)
    patch = np.multiply(patch, scale_factor)
    patch = np.reshape(patch, [patch.shape[0], patch.shape[1], 1])
    return patch
############### croppedface operation ends here

######
#
#return the wavelet complex shift data
def getWaveletComplexShiftData(img, wf='rbio3.1', mo='periodization', log=False, Type=0):
    '''input:
    img: image data should in range [0,1] with shape [2x,2y]
    wf: wavelet family, default is rbio3.1
    mo: mode of wavelet family, default is periodization
    log: whether return the complex data for logging
    Type: 0 performs waveletPacket2D first, then performs fft, output with shape [x, y, 2]; 
            1 only performs fft, output with shape [2x, 2y, 2].
            3 perform the logarithm of absolute fft, output with shape [2x, 2y]

    return:
    return wavelet complex shift data with m x n x 2 shape, where the wavelet data is the sum of 'v' 'h' 'd' components.
    '''
    #wc=np.fft.fft2(img) without wavelet decomposing
    if Type==3:
        wc=np.fft.fft2(img)# without wavelet decomposing
        wcs=np.fft.fftshift(wc)
        wcs=np.log(np.abs(wcs))
        twcs=wcs.real
        twcs=np.reshape(twcs, [twcs.shape[0], twcs.shape[1], 1])
    if Type==0:
        wave=pywt.WaveletPacket2D(data=img, wavelet=wf, mode=mo)
        wst=np.add(np.add(wave['h'].data, wave['v'].data), wave['d'].data)
        wc=np.fft.fft2(wst)
        wcs=np.fft.fftshift(wc)
        twcs=np.stack((wcs.real, wcs.imag), axis=2)
    elif Type==1:
        wc=np.fft.fft2(img)# without wavelet decomposing
        wcs=np.fft.fftshift(wc)
        twcs=np.stack((wcs.real, wcs.imag), axis=2)
    if log:
        return (twcs, np.hstack((wcs.real, wcs.imag)))
    return (twcs, None)
######
#
#return the wavelet data
def getWaveletData(img, wf='rbio3.1', mo='periodization'):
    '''input:
    img: image data should in range [0,1] with shape [2x,2y]
    wf: wavelet family, default is rbio3.1
    mo: mode of wavelet family, default is periodization
    log: whether return the complex data for logging
    
    return:
    return wavelet data with m x n shape, where the wavelet data is the sum of 'v' 'h' 'd' components.
    '''
    #wc=np.fft.fft2(img) without wavelet decomposing
    wave=pywt.WaveletPacket2D(data=img, wavelet=wf, mode=mo)
    wst=np.add(np.add(wave['h'].data, wave['v'].data), wave['d'].data)
    wst=np.reshape(wst, [wst.shape[0], wst.shape[1], 1])
    return wst
#return the wavelet data
def get_WaveletDataC3(img, wf='rbio3.1', mo='periodization'):
    '''input:
    img: image data should in range [0,1] with shape [2x,2y]
    wf: wavelet family, default is rbio3.1
    mo: mode of wavelet family, default is periodization
    
    return:
    return wavelet data with m x n x 3 shape, where the wavelet data is the stacks of 'v' 'h' 'd' components.
    '''
    wave=pywt.WaveletPacket2D(data=img, wavelet=wf, mode=mo)
    wst=np.stack([wave['h'].data,wave['v'].data,wave['d'].data], axis=-1)
    return wst
######

##
# from LXH
#weber face operation for normalizing the images' illuminations
weber_sigma=0.85#0.6
def __weberface(image, weberface_sigma=weber_sigma, revers=False):
    ''' when revers is true, the return image is set to the reverse version.
    the revers defaut value is False'''
    imgr=cv2.copyMakeBorder(image,1,1,1,1, cv2.BORDER_REPLICATE)
    imgr=imgr/255.0
    imf1 = ndimage.filters.gaussian_filter(imgr, weberface_sigma)
    lx, ly = imgr.shape
    imfc = imf1[1:(lx - 1), 1:(ly - 1)]
    imflu = imf1[0:(lx - 2), 0:(ly - 2)]
    imflb = imf1[0:(lx - 2), 2:ly]
    imfru = imf1[2:lx, 0:(ly - 2)]
    imfrb = imf1[2:lx, 2:ly]
    constc = 0.01
    weber_face_in = (4 * imfc - imflu - imflb - imfru - imfrb) / (imfc + constc)
    out = np.arctan(weber_face_in)
    if revers:
        out = 1-out
    else:
        out = out+1
    maxv=np.max(out)
    minv=np.min(out)
    out=255*(out-minv)/(maxv-minv)
    #out=out*1.25
    #np.clip(out, 1, 255,out)
    return out
############weber face operation ends here

##
# from RCL
#The followings are operations for Enhance Local Texture Feature Set 
def __normalizeImage(x):
    max1 = np.max(x)
    min1 = np.min(x)
    x=np.asarray(x)
    x=(x-min1)/(max1-min1)*255
    return x
def __Gamma(x):  #第一步:伽马校正
    max1 = np.max(x)
    x2 = np.power(x/max1, 0.4)
    x2 = x2*max1
    return x2
def __DOG(x): #第二步：高斯差分
    blur1 = cv2.GaussianBlur(x, (0, 0), 1.0);
    blur2 = cv2.GaussianBlur(x, (0, 0), 2.0);
    dog = blur1 - blur2
    return dog
def __constrast_equlization(x2): #第三步: 对比均衡化
    tt = 10
    a = 0.1
    x_temp = np.power(abs(x2), a)
    mm = np.mean(x_temp)
    x2 = x2 / (mm ** (1 / a))
    # 第三步第一个公式完成

    for i in range(x2.shape[0]):
        for j in range(x2.shape[1]):
            x_temp[i, j] = min(tt, (abs(x2[i, j])))**a

    mm = np.mean(x_temp)
    x2 = x2 / (mm ** (1 / a))
    x2 = tt*np.tanh(x2/tt)

    return x2
    # 第三步第二个公式完成
##Enhance local texture feature set
def __ELTFS(img, blur=0):  #整合前面三步,对图像x进行处理
    x = __Gamma(img)  # Gramma
    x = __DOG(x)  # __DOG
    x = __constrast_equlization(x)  # __constrast_equlization

    x = __normalizeImage(x)
    if blur>0:
        x = cv2.medianBlur(np.uint8(x), blur)
        x = __normalizeImage(x)

    return x
########The Enhance Local Texture Feature Set operations end here

######
#
#the following functions are for geometry feature extractions
def __getDistFrom3PTS(x1, y1, x2, y2, x3, y3):
    '''get the Euclidean distance of p1 and the center of p2 and p3'''
    return math.sqrt((x1-0.5*(x2+x3))**2+(y1-0.5*(y2+y3))**2)

def __getD(x1,y1,x2,y2):
    '''get the Euclidean distance of p1 and p2'''
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def __getTriangleFeatures(X, Y, triangles):
    '''Return the triangleFeatures of ratios among lengths of middle lines'''
    tri_feat=[]
    for i in range(len(triangles)):
        m1=__getDistFrom3PTS(X[triangles[i][0]], Y[triangles[i][0]], 
                                                X[triangles[i][1]], Y[triangles[i][1]], 
                                                X[triangles[i][2]], Y[triangles[i][2]])
        m2=__getDistFrom3PTS(X[triangles[i][1]], Y[triangles[i][1]], 
                                                X[triangles[i][0]], Y[triangles[i][0]], 
                                                X[triangles[i][2]], Y[triangles[i][2]])
        m3=__getDistFrom3PTS(X[triangles[i][2]], Y[triangles[i][2]], 
                                                X[triangles[i][1]], Y[triangles[i][1]], 
                                                X[triangles[i][0]], Y[triangles[i][0]])
        if m1>0:
            tri_feat.append(m2/m1)
            tri_feat.append(m3/m1)
        else:
            tri_feat.append(m2*1.41421356)#minimun distance in integer pixels is sqrt(0.5)
            tri_feat.append(m3*1.41421356)#minimun distance in integer pixels is sqrt(0.5)
    return tri_feat

def __getEDList(X, Y, keyPs):
    '''Return the Euclidean distances list of the keypoints keyPs'''
    ed=[]
    for i in keyPs:
        ed.append(__getD(X[i[0]],Y[i[0]],
                         X[i[1]],Y[i[1]]))
    return ed

def __getDirectionalXY(X, Y, keyPs):
    dxy=[]
    for v in keyPs:
        dxy.append(X[v[1]]-X[v[0]])
        dxy.append(Y[v[1]]-Y[v[0]])
    return dxy

def __getXYcor(X, Y):
    xc=X[17:]-X[27]
    yc=Y[17:]-Y[27]
    xc=xc/np.std(xc)
    yc=yc/np.std(yc)
    #xc=list(xc[0:10])+list(xc[19:])#85.88
    #yc=list(yc[0:10])+list(yc[19:])#85.88
    #xc=list(xc[19:])#85.725
    #yc=list(yc[19:])#85.725
    #xc=list(xc[17:27])+list(xc[36:])#87.0875
    #yc=list(yc[17:27])+list(yc[36:])#87.0875
    return list(xc),list(yc)

def __getLocalCordSetAnalysis(X, Y):
    assert len(X)==len(Y), 'Unexpected lengths between X and Y'
    cx = np.mean(X)
    cy = np.mean(Y)
    nX = X-cx
    nY = Y-cy
    nD = nX[:]#initialized
    #cos = nX[:]#initialized
    #sin = nY[:]#initialized
    for i in range(len(nX)):
        nD[i]=__getD(nX[i], nY[i], cx, cy)
        #if nD[i]>0:
        #    cos[i]=cos[i]/nD[i]
        #    sin[i]=sin[i]/nD[i]
        #else:
        #    cos[i]=0
        #    sin[i]=0
    #met=np.std(nD)
    #nX=nX/met
    #nY=nY/met
    nD=nD/np.max(nD)
    #return (list(nD)+list(cos)+list(sin))
    #return list(nD)
    return (list(nD)+list(nX)+list(nY))


def __landmarkPart1(X, Y, stdxy):
    '''Return Part1(eyes part) features'''
    part1=[]
    #deprecated by __LandmarkFeaturesV2
    #tx= (X[39]+X[42])*0.25+X[27]*0.5
    #ty= (Y[39]+Y[42])*0.25+Y[27]*0.5
    #centx=(tx+X[28])*0.5
    #centy=(ty+Y[28])*0.5
    ##sme=math.sqrt((centx-X[27])**2+(centy-Y[27])**2)
    #sme=(centx-X[27])**2+(centy-Y[27])**2#137
    #part1.append((centx-tx)/sme)
    #part1.append((centy-ty)/sme)
    #for i in range(len(LMP1_keys)):#deprecated by __LandmarkFeaturesV2 with features=tri_f1+tri_f2+ed1+ed2+x+y+l_eye+r_eye+m
    #    part1.append((centx-X[LMP1_keys[i]])/sme)
    #    part1.append((centy-Y[LMP1_keys[i]])/sme)

    #tri_features1=np.asarray(__getTriangleFeatures(X, Y, LMP1_triangle))#testing
    #tri_features1=list(tri_features1/np.std(tri_features1))#not helpful
    tri_features1=list(np.asarray(__getTriangleFeatures(X, Y, LMP1_triangle))/stdxy)
    eud1=list(np.asarray(__getEDList(X, Y,LMP1_Dists))/stdxy)
    return part1, tri_features1, eud1#Dimension: 24, 28, 10

def __landmarkPart2(X, Y, stdxy):
    '''Return Part2(mouth part) features'''
    part2=[]
    ##deprecated by __LandmarkFeaturesV2
    #centx= (X[33]+X[51])*0.5
    #centy= (Y[33]+Y[51])*0.5
    ##sme=math.sqrt((centx-X[33])**2+(centy-Y[33])**2)
    #sme=(centx-X[33])**2+(centy-Y[33])**2#137
    #for i in range(len(LMP2_keys)):#deprecated by __LandmarkFeaturesV2 with features=tri_f1+tri_f2+ed1+ed2+x+y+l_eye+r_eye+m
    #    part2.append((centx-X[LMP2_keys[i]])/sme)
    #    part2.append((centy-Y[LMP2_keys[i]])/sme)

    #tri_features2=np.asarray(__getTriangleFeatures(X, Y, LMP2_triangle))#testing
    #tri_features2=list(tri_features2/np.std(tri_features2))# not helpful
    tri_features2=list(np.asarray(__getTriangleFeatures(X, Y, LMP2_triangle))/stdxy)#testing
    eud2=list(np.asarray(__getEDList(X, Y,LMP2_Dists))/stdxy)
    return part2, tri_features2, eud2#Dimension: 28, 28, 4

def __LandmarkFeaturesV2(X, Y):
    stdxy=np.std(X)+np.std(Y)
    part1, tri_f1, ed1=__landmarkPart1(X, Y, stdxy)
    part2, tri_f2, ed2=__landmarkPart2(X, Y, stdxy)
    x, y = __getXYcor(X, Y)
    #features=part1+part2+tri_f1+tri_f2+ed1+ed2+x+y###Dx55  236
    
    #getlocalCordSetAna
    l_eye=__getLocalCordSetAnalysis(np.concatenate([X[17:22],X[36:42]]),
                                                  np.concatenate([Y[17:22],Y[36:42]]))
    r_eye=__getLocalCordSetAnalysis(np.concatenate([X[22:27],X[42:48]]),
                                                   np.concatenate([Y[22:27],Y[42:48]]))
    m=__getLocalCordSetAnalysis(X[48:68],Y[48:68])
    #features=part1+part2+tri_f1+tri_f2+ed1+ed2+x+y+l_eye+r_eye+m
    features=tri_f1+tri_f2+ed1+ed2+x+y+l_eye+r_eye+m#Dx63  310
    #print('LocalTriangleFeat: %d\tLocalDistFeat: %d\tLocalCordAnaFeat: %d\tAUXFeat%d'%(len(tri_f1+tri_f2),len(ed1+ed2),len(l_eye+r_eye+m),len(x+y)))
    #exit()
    #features=part1+part2+tri_f1+tri_f2+ed1+ed2+l_eye+r_eye+m#Dx64
    print('Geometry feature length: %d'%len(features))
    return features
##########The above functions are for geometry feature extractions

def __UnifiedOutputs(rescaleImg, Geof, Geo_features, Patchf, eyepatch, foreheadpatch, mouthpatch, croppedface):
    ''''add additional operations'''
    ##rescale to 224x224 from 128x128
    #rescaleImg=cv2.resize(rescaleImg, finalsize)

    #if Patchf:
    #    #rescale to 224x224 from 128x128
    #    croppedface=cv2.resize(croppedface, finalsize)

    return rescaleImg, Geof, Geo_features, Patchf, eyepatch, foreheadpatch, mouthpatch, croppedface
    #return (__formalizeImg(rescaleImg), Geof, Geo_features, Patchf, __formalizeImg(eyepatch), 
    #        __formalizeImg(foreheadpatch), __formalizeImg(mouthpatch), __formalizeImg(croppedface))

def __genLMFandIP(img, w, h, LM, Patches, Cropped, regular=False, X=None, Y=None, XYNoneFlag=True):
    """Return the Geometry features from landmarks and Images patches.
    If regularize is set to False, it will always return cosine True and three images for the patches operation.
    Otherwise, it could return cosine False and four None values
    X and Y are ndarray"""
    landmark_features, eye_patch, forehead_patch, mouth_patch, crop_face=None, None, None, None, None
    if LM:
        if XYNoneFlag:
            LM=False
        else:
            landmark_features= __LandmarkFeaturesV2(X, Y)
    if Patches:
        if XYNoneFlag:
           if regular:
                eye_patch = __getEyePatch(img, w, h)
                forehead_patch = __getMiddlePatch(img, w, h)
                mouth_patch = __getMouthPatch(img, w, h)
           else:
                Patches=False
        else:
            eye_patch = __getEyePatch(img, w, h, X, Y)
            forehead_patch = __getMiddlePatch(img, w, h, X, Y)
            mouth_patch = __getMouthPatch(img, w, h, X, Y)
    if Cropped:
        if XYNoneFlag:
            if regular:
                crop_face = __getCroppedFace(img, w, h)
            else:
                Cropped=False
        else:
            crop_face = __getCroppedFace(img, w, h, X, Y)
    #return __UnifiedOutputs(img, LM, landmark_features, Patches, eye_patch, forehead_patch, mouth_patch, crop_face)
    return rescaleImg, LM, landmark_features, Patches, eye_patch, forehead_patch, mouth_patch, crop_face, Cropped
def __cropImg(img, shape=None, LM=False, Patches=False, Cropped=False, regularize=False, trg_size=target_size, rescale=rescaleImg):
    """Rescale, adjust, and crop the images.
    If shape is None, it will recale the img without croping and return __genLMFandIP"""
    if not shape==None:
        nLM = shape.num_parts
        lms_x = np.asarray([shape.part(i).x for i in range(0,nLM)])
        lms_y = np.asarray([shape.part(i).y for i in range(0,nLM)])

        tlx = float(min(lms_x))#top left x
        tly = float (min(lms_y))#top left y
        ww = float (max(lms_x) - tlx)
        hh = float(max(lms_y) - tly)
        # Approximate LM tight BB
        h = img.shape[0]
        w = img.shape[1]
        cx = tlx + ww/2
        cy = tly + hh/2
        #tsize = max(ww,hh)/2
        tsize = ww/2

        # Approximate expanded bounding box
        btlx = int(round(cx - rescale[0]*tsize))
        btly = int(round(cy - rescale[1]*tsize))
        bbrx = int(round(cx + rescale[2]*tsize))
        bbry = int(round(cy + rescale[3]*tsize))
        #since recale[0]+recale[2]=recale[1]+recale[3], nw=nh
        nw = int(bbrx-btlx)
        nh = int(bbry-btly)
        #nh = nw

        #adjust relative location
        x0=(np.mean(lms_x[36:42])+np.mean(lms_x[42:48]))/2
        y0=(np.mean(lms_y[36:42])+np.mean(lms_y[42:48]))/2
        Mpx=int(round((mpoint[0]*nw/float(target_size))-x0+btlx))
        Mpy=int(round((mpoint[1]*nh/float(target_size))-y0+btly))
        btlx=btlx-Mpx
        bbrx=bbrx-Mpx
        bbry=bbry-Mpy
        btly=btly-Mpy
        
        print('coordinate adjustment: %d\t%d'%(Mpx, Mpy))
        Xa=np.round((lms_x-btlx)*trg_size/nw)
        Ya=np.round((lms_y-btly)*trg_size/nh)
        ##simplified cropping logic edite on 2018.03.07
        #h = img.shape[0]
        #w = img.shape[1]
        #nsize = float (max(lms_x) - min(lms_x))*1.4504# (rescale[0]+rescle[2])/2
        #x0=(np.mean(lms_x[36:42])+np.mean(lms_x[42:48]))/2
        #y0=(np.mean(lms_y[36:42])+np.mean(lms_y[42:48]))/2
        #btlx=x0-mpoint[0]*nsize/target_size
        #btly=y0-mpoint[1]*nsize/target_size
        #bbrx=btlx+nsize
        #bbry=btly+nsize
        #Xa=np.round((lms_x-btlx)*trg_size/nsize)
        #Ya=np.round((lms_y-btly)*trg_size/nsize)
        #few=open(eyelog,'a')
        #few.write('%lf %lf\n'%((np.mean(Xa[36:42])+np.mean(Xa[42:48]))/2,(np.mean(Ya[36:42])+np.mean(Ya[42:48]))/2))
        #few.close()

        imcrop = np.zeros((nh,nw), dtype = "uint8")

        blxstart = 0
        if btlx < 0:
            blxstart = -btlx
            btlx = 0
        brxend = nw
        if bbrx > w:
            brxend = w+nw - bbrx#brxend=nw-(bbrx-w)
            bbrx = w
        btystart = 0
        if btly < 0:
            btystart = -btly
            btly = 0
        bbyend = nh
        if bbry > h:
            bbyend = h+nh - bbry#bbyend=nh-(bbry-h)
            bbry = h
        imcrop[btystart:bbyend, blxstart:brxend] = img[btly:bbry, btlx:bbrx]
        im_rescale=cv2.resize(imcrop,(trg_size, trg_size))
        return __genLMFandIP(im_rescale, trg_size, trg_size, LM, Patches, Cropped, regular=regularize, X=Xa, Y=Ya, XYNoneFlag=False)
    else:
        im_rescale=cv2.resize(img, (trg_size, trg_size))
        return __genLMFandIP(im_rescale, trg_size, trg_size, LM, Patches, Cropped, False)

def getLandMarkFeatures_and_ImgPatches(img, withLM=True, withPatches=True, CropFace=True, fromfacedataset=True):
    """Input:
    img: image to be processed.
    withLM: flag indicates whether to process the landmark operations or not.
    withPatches: flag indicates whether to process the patches operations or not.
    CropFace: flag indicates whether to process the cropping operations or not
    fromfacedataset: flag whether the image is from cosine regular face dataset.
        If fromfacedataset is set to True, always return an img.
    
Outputs: 
    rescaleimg, lmf, lmfeat, pf, eyeP, middleP, mouthP
    
    rescaleimg: the rescale image of the input img
    
    lmf: landmark flag. If True means landmarks were detected in the images. If False means no landmarks were detected.
    lmfeat: landmark features. If lmf is True, it is cosine ndarray, otherwise it is cosine None value.
    
    pf: patches flag, signal whether valid patches are return or not.
    eyeP: if pf is True, it's cosine patch image of img, otherwise it's cosine None value.
    middleP: if pf is True, it's cosine patch image of img, otherwise it's cosine None value.
    mouthP: if pf is True, it's cosine patch image of img, otherwise it's cosine None value.
    """
    if len(img.shape) == 3 and img.shape[2]==3:
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g_img = img
    td1= time.time()
    #f_ds=detector(g_img, 1)#1 represents upsample the image 1 times for detection
    f_ds=detector(g_img, 0)
    td2 = time.time()
    print('Time in detecting face: %fs'%(td2-td1))
    if len(f_ds) == 0:
        f_ds=detector(g_img, 1)
        if len(f_ds)==0:
            #pl.write('0')
            print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: No face was detected from the image")
            if not fromfacedataset:
                print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: No face was detected, and return False and None values")

                return None, False, None, False, None, None, None, None, False
            else:
                print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: Processing the default config on the image")
                return __cropImg(g_img, LM=withLM, Patches=withPatches, Cropped=CropFace, regularize=fromfacedataset)
    elif len(f_ds) > 1:
        print(">>>***%%%Warning [getLandMarkFeatures_and_ImgPatches()]: Only process the first face detected.")
    f_shape = predictor(g_img, f_ds[0])
    #pl.write('1')
    return __cropImg(g_img, shape=f_shape, LM=withLM, Patches=withPatches, Cropped=CropFace, regularize=fromfacedataset)

 
######
#
#The followings are for calibrate the image
def __RotateTranslate(image, angle, center =None, new_center =None, resample=IM.BICUBIC):
    '''Rotate the image according to the angle'''
    if center is None:  
        return image.rotate(angle=angle, resample=resample)  
    nx,ny = x,y = center  
    if new_center:  
        (nx,ny) = new_center  
    cosine = math.cos(angle)  
    sine = math.sin(angle)  
    c = x-nx*cosine-ny*sine  
    d =-sine  
    e = cosine
    f = y-nx*d-ny*e  
    return image.transform(image.size, IM.AFFINE, (cosine,sine,c,d,e,f), resample=resample)
def __RotaFace(image, eye_left=(0,0), eye_right=(0,0)):
    '''Rotate the face according to the eyes'''
    # get the direction from two eyes
    eye_direction = (eye_right[0]- eye_left[0], eye_right[1]- eye_left[1])
    # calc rotation angle in radians
    rotation =-math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    # rotate original around the left eye  
    image = __RotateTranslate(image, center=eye_left, angle=rotation)
    return image
def __shape_to_np(shape):
    '''Transform the shape points into numpy array of 68*2'''
    nLM = shape.num_parts
    x = np.asarray([shape.part(i).x for i in range(0,nLM)])
    y = np.asarray([shape.part(i).y for i in range(0,nLM)])
    return x,y
def calibrateImge(imgpath):
    '''Calibrate the image of the face'''
    if os.path.exists(imgpath):
        imgcv_gray=cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    else:
        if imgpath.find('.jpg')>0 and os.path.exists(imgpath.replace('.jpg','.png')):
            imgcv_gray=cv2.imread(imgpath.replace('.jpg','.png'), cv2.IMREAD_GRAYSCALE)
        else:
            raise RuntimeError('The value read from the imagepath is None. No image was loaded.\nimagepath: %s'%(imgpath))
    dets = detector(imgcv_gray,0)
    if len(dets)==0:
        dets=detector(imgcv_gray, 1)
        if len(dets)==0:
            print("No face was detected^^^^^^^^^^^^^^")
            return False, imgcv_gray
    lmarks=[]
    for id, det in enumerate(dets):
        if id > 0:
            print("ONLY process the first face>>>>>>>>>")
            break
        shape = predictor(imgcv_gray, det)
        x, y = __shape_to_np(shape)
    lmarks = np.asarray(lmarks, dtype='float32')
    pilimg=IM.fromarray(imgcv_gray)
    rtimg=__RotaFace(pilimg, eye_left=(np.mean(x[36:42]),np.mean(y[36:42])),
                                           eye_right=(np.mean(x[42:48]),np.mean(y[42:48])))
    imgcv_gray=np.array(rtimg)
    return True, imgcv_gray



### system module
crop_size=0.7
def __getLandMarkFeatures_and_ImgPatches_for_Facelist(img_list, points,withLM=True, withPatches=True, cropFace=True):
    """Input:
    img_list: face image list to be processed.
    withLM: flag indicates whether to process the landmark operations or not.
    withPatches: flag indicates whether to process the patches operations or not.
    fromfacedataset: flag whether the image is from cosine regular face dataset.
        If fromfacedataset is set to True, always return an img.
    
Outputs: 
    rescaleimg, lmf, lmfeat, pf, eyeP, middleP, mouthP
    
    rescaleimg: the rescale image of the input img
    
    lmf: landmark flag. If True means landmarks were detected in the images. If False means no landmarks were detected.
    lmfeat: landmark features. If lmf is True, it is cosine ndarray, otherwise it is cosine None value.
    
    pf: patches flag, signal whether valid patches are return or not.
    eyeP: if pf is True, it's cosine patch image of img, otherwise it's cosine None value.
    middleP: if pf is True, it's cosine patch image of img, otherwise it's cosine None value.
    mouthP: if pf is True, it's cosine patch image of img, otherwise it's cosine None value.
    """
    featlist={}
    featlist['img']=[]
    featlist['geo']=[]
    featlist['eye']=[]
    featlist['fore']=[]
    featlist['mou']=[]
    featlist['crop']=[]
    featlist['imgRec']=[]
    featlist['geoRec']=[]
    featlist['patchRec']=[]
    featlist['cropRec']=[]
    PTS=[]
    for ind, img in enumerate(img_list):
        if len(img.shape) == 3 and img.shape[2]==3:
            g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g_img = img

        f_ds=detector(g_img, 1)
        if len(f_ds) == 0:
            #pl.write('0')
            raise RuntimeWarning(">>>***%%%Warning [__getLandMarkFeatures_and_ImgPatches_for_Facelist()]: No face was detected, and return None values")
        elif len(f_ds) == 1:
            f_shape = predictor(g_img, f_ds[0])
            rescaleimg, gf, geo_features, pf, eyepatch, foreheadpatch, mouthpatch, croppedface, cf=__cropImg(g_img, shape=f_shape, LM=withLM, Patches=withPatches, Cropped=cropFace, regularize=False)
            featlist['img'].append(rescaleimg)
            featlist['imgRec'].append(points[ind])
            if gf:
                featlist['geo'].append(geo_features)
                featlist['geoRec'].append(points[ind])
            if pf:
                featlist['eye'].append(eyepatch)
                featlist['fore'].append(foreheadpatch)
                featlist['mou'].append(mouthpatch)
                featlist['patchRec'].append(points[ind])
            if cf:
                featlist['crop'].append(croppedface)
                featlist['cropRec'].append(points[ind])
        else:
            max_area=0
            for i in range(len(f_ds)):
                f_shape = predictor(g_img, f_ds[i])
                curr_area = (f_ds[i].right()-f_ds[i].left()) * (f_ds[i].bottom()-f_ds[i].top())#
                if curr_area > max_area:
                    max_area = curr_area
                    rescaleimg, gf, geo_features, pf, eyepatch, foreheadpatch, mouthpatch, croppedface, cf=__cropImg(g_img, shape=f_shape, LM=withLM, Patches=withPatches, Cropped=cropFace, regularize=False)
            featlist['img'].append(rescaleimg)
            featlist['imgRec'].append(points[ind])
            if gf:
                featlist['geo'].append(geo_features)
                featlist['geoRec'].append(points[ind])
            if pf:
                featlist['eye'].append(eyepatch)
                featlist['fore'].append(foreheadpatch)
                featlist['mou'].append(mouthpatch)
                featlist['patchRec'].append(points[ind])
            if cf:
                featlist['crop'].append(croppedface)
                featlist['cropRec'].append(points[ind])

    return featlist

def __calibrateImageWithArrayInput(img):
    '''Calibrate the image of the face'''
    if img is None:
        print('Unexpected ERROR: The value input is None. No image was loaded')
        return False, None, None
    if len(img.shape)==3:
        imgcv_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape)==2:
        imgcv_gray=img[:]
    else:
        print('ERROR: Unexpected data format.')
        return False, None
    dets = detector(imgcv_gray,1)
    img_face_list=[]
    rectPoint=[]
    if len(dets)==0:
        print("No face was detected^^^^^^^^^^^^^^")
        return False, img_face_list, rectPoint
    h=imgcv_gray.shape[0]
    w=imgcv_gray.shape[1]
    for id, det in enumerate(dets):
        shape = predictor(imgcv_gray, det)
        x, y = __shape_to_np(shape)
        top=[]
        top.append((det.left(),det.top()))
        top.append((det.right(),det.bottom()))
        rectPoint.append(top)

        #crop face
        tlx=float(min(x))
        tly=float(min(y))
        ww=float(max(x)-tlx)
        hh=float(max(y)-tly)
        cx=tlx+ww/2
        cy=tly+hh/2
        tsize=ww*crop_size
        # Approximate expanded bounding box
        btlx = int(round(cx - rescaleImg[0]*tsize))
        btly = int(round(cy - rescaleImg[1]*tsize))
        bbrx = int(round(cx + rescaleImg[2]*tsize))
        bbry = int(round(cy + rescaleImg[3]*tsize))
        nw = int(bbrx-btlx)
        nh = int(bbry-btly)
        imcrop = np.zeros((nh,nw), dtype = "uint8")
        blxstart = 0
        if btlx < 0:
            blxstart = -btlx
            btlx = 0
        brxend = nw
        if bbrx > w:
            brxend = w+nw - bbrx#brxend=nw-(bbrx-w)
            bbrx = w
        btystart = 0
        if btly < 0:
            btystart = -btly
            btly = 0
        bbyend = nh
        if bbry > h:
            bbyend = h+nh - bbry#bbyend=nh-(bbry-h)
            bbry = h
        imcrop[btystart:bbyend, blxstart:brxend] = imgcv_gray[btly:bbry, btlx:bbrx]
        pilimg=IM.fromarray(imcrop)
        rtimg=__RotaFace(pilimg, eye_left=(np.mean(x[36:42]),np.mean(y[36:42])),
                                           eye_right=(np.mean(x[42:48]),np.mean(y[42:48])))
        img_face_list.append(np.array(rtimg))

    return True, img_face_list, rectPoint

def preprocessImage(img, LM=False, Patches=False, Crop=False):#
    """process image as input for model, extract all human faces in the image and their corresponding coordinate points
        
    Args:
        img (ndarray): input image represent in numpy.ndarray
    
    Returns: a dictionnary contains the following information
        featlist['img']: a list of rescaled and cropped image of the detected face
        featlist['geo']: a list of geometry features
        featlist['eye']: a list of eye patch
        featlist['fore']: a list of forehead patch
        featlist['mou']: a list of mouth patch
        featlist['crop']: a list of cropped face
        featlist['imgRec']: a list of rectangle for img
        featlist['geoRec']: a list of rectangle for geo
        featlist['patchRec']: a list of rectangle for eye, fore, and mou
        featlist['cropRec']: a list of rectangle for crop
    """
    
    # pack the features and return   
    features = {}
    detected, face_list, originalPoints = __calibrateImageWithArrayInput(img)
    if detected: # detect human face
        features= __getLandMarkFeatures_and_ImgPatches_for_Facelist(face_list, LM, Patches, Crop)
        
        #print('detect {0} human faces'.format(len(originalPoints)))
    return features
    
########image calibration ends here
#------------------------------------------------------------------------#
