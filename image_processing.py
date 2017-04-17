# -------------------------------------------------------------------------------------- #
# Cervical Cancer Screening
# -------------------------------------------------------------------------------------- #

def process_all_images():
    import time
    import os
    import multiprocessing    

    
    startTime = time.time()
    
    #rootDir = "C:/Users/rarez/Documents/Data Science/cervical_cancer/data_work"
    rootDir = "C:/Users/rarez/Documents/Data Science/cervical_cancer/data_all"
    
    imgSet1 = {"Source": "/full_resolution/train/Type_1/", "Dest": "/128x128/train/Type_1/"}
    imgSet2 = {"Source": "/full_resolution/train/Type_2/", "Dest": "/128x128/train/Type_2/"}
    imgSet3 = {"Source": "/full_resolution/train/Type_3/", "Dest": "/128x128/train/Type_3/"}
    #imgSet4 = {"Source": "/full_resolution/test/", "Dest": "/Cropped_128x128/test/"}
    
    #imgSets = [imgSet1, imgSet2, imgSet3, imgSet4]
    imgSets = [imgSet1, imgSet2, imgSet3]
    
    for imgSet in imgSets:
        print("Processing folder: {}".format(imgSet["Source"]), flush=True)
        
        sourcePath = rootDir + imgSet["Source"]    
        destPath = rootDir + imgSet["Dest"]
        
        fileNames = os.listdir(sourcePath)
        
        argList = []
        for fileName in fileNames:
            inFile = sourcePath + fileName
            outFile = destPath + fileName
            argList.append([inFile, outFile])
    
        pool = multiprocessing.Pool(processes = 8)
        result = pool.starmap_async(process_image, argList)
        result.get()
        pool.close()
        pool.join()
    
    print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))
    
    return


def process_image(inFile, outFile):
    import warnings
    import skimage.io as io
    import skimage.transform as transf

    new_size = (128, 128)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            inImage = io.imread(inFile)
            outImage = transf.resize(inImage, new_size, mode='reflect')
            #outImage = cropCervix(inImage)
            #outImage = transf.resize(outImage, new_size, mode='reflect')
            io.imsave(outFile, outImage)    
        
    except:
        print("Could not process image: " + inFile, flush=True)
        
    return

# --------------------------------------------------------------------------- #
# Identify the Cervix and crop the image around it

def cropCervix(img):
    import cv2
    import numpy as np
    from sklearn import mixture
    from sklearn.utils import shuffle
    from skimage import measure

    
    img = cropFrame(img)
    w = img.shape[0]
    h = img.shape[1]
                    
    # Saturating the a-channel at 150 helps avoiding wrong segmentation
    # in the case of close-up cervix pictures where the bloody os is falsly segemented as the cervix.
    Ra = Ra_space(img, 1.0, 150) 
    a_channel = np.reshape(Ra[:,1], (w,h))

    g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
    image_array_sample = shuffle(Ra, random_state=0)[:1000]
    g.fit(image_array_sample)
    labels = g.predict(Ra)
    labels += 1 # Add 1 to avoid labeling as 0 since regionprops ignores the 0-label.

    # The cluster that has the highest a-mean is selected.
    labels_2D = np.reshape(labels, (w,h))
    gg_labels_regions = measure.regionprops(labels_2D, intensity_image = a_channel)
    gg_intensity = [prop.mean_intensity for prop in gg_labels_regions]
    cervix_cluster = gg_intensity.index(max(gg_intensity)) + 1

    mask = np.zeros((w * h,1),'uint8')
    mask[labels==cervix_cluster] = 255
    mask_2D = np.reshape(mask, (w,h))

    cc_labels = measure.label(mask_2D, background=0)
    regions = measure.regionprops(cc_labels)
    areas = [prop.area for prop in regions]

    regions_label = [prop.label for prop in regions]
    largestCC_label = regions_label[areas.index(max(areas))]
    mask_largestCC = np.zeros((w,h),'uint8')
    mask_largestCC[cc_labels==largestCC_label] = 255

    img_masked = img.copy()
    img_masked[mask_largestCC==0] = (0,0,0)
    img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY);
        
    _,thresh_mask = cv2.threshold(img_masked_gray,0,255,0)
        
    kernel = np.ones((11,11), np.uint8)
    thresh_mask = cv2.dilate(thresh_mask, kernel, iterations = 1)
    thresh_mask = cv2.erode(thresh_mask, kernel, iterations = 2)
    _, contours_mask, _ = cv2.findContours(thresh_mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours_mask, key = cv2.contourArea, reverse = True)[0]
                
    x,y,w,h = cv2.boundingRect(main_contour)
    
    imgCervix = img[y:y+h, x:x+w]
    
    return imgCervix

def cropFrame(img):
    import cv2
    import numpy as np

    out_size = 2048    
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*out_size/img.shape[0]), out_size)
    else:
        tile_size = (out_size, int(img.shape[0]*out_size/img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)
            
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            
    ff = np.zeros((gray.shape[0],gray.shape[1]), 'uint8') 
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0]+2,gray.shape[1]+2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 1)
    
    rect = maxRect(ff)
    img_crop = img[min(rect[0],rect[2]):max(rect[0],rect[2]), min(rect[1],rect[3]):max(rect[1],rect[3])]

    return img_crop


def Ra_space(img, Ra_ratio, a_threshold):
    import math
    import cv2
    import numpy as np

    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            Ra[i*h+j, 1] = min(imgLab[i][j][1], a_threshold)
            
    Ra[:,0] /= max(Ra[:,0])
    Ra[:,0] *= Ra_ratio
    Ra[:,1] /= max(Ra[:,1])

    return Ra


def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else: 
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i-position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif(height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()    
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area =  maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea
            

def maxRect(img):
    import numpy as np

    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r-1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (int(maxArea[3]+1-maxArea[0]/abs(maxArea[1]-maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])

    
    
# -------------------------------------------------------------------------------------- #
# Main module function
# -------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    process_all_images()

