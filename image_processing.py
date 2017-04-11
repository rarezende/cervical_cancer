# -------------------------------------------------------------------------------------- #
# Cervical Cancer Screening
# -------------------------------------------------------------------------------------- #

def process_all_images():
    import time
    import os
    import multiprocessing    

    
    startTime = time.time()
    
    rootDir = "C:/Users/rarez/Documents/Data Science/cervical_cancer/data_work"
    #rootDir = "C:/Users/rarez/Documents/Data Science/cervical_cancer/data_all"
    
    imgSet1 = {"Source": "/full_resolution/train/Type_1/", "Dest": "/200x200/train/Type_1/"}
    imgSet2 = {"Source": "/full_resolution/train/Type_2/", "Dest": "/200x200/train/Type_2/"}
    imgSet3 = {"Source": "/full_resolution/train/Type_3/", "Dest": "/200x200/train/Type_3/"}
    imgSet4 = {"Source": "/full_resolution/test/", "Dest": "/200x200/test/"}
    
    imgSets = [imgSet1, imgSet2, imgSet3, imgSet4]
    #imgSets = [imgSet1, imgSet2, imgSet3]
    
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
    
        pool = multiprocessing.Pool(processes = 6)
        result = pool.starmap_async(resize_image, argList)
        result.get()
        pool.close()
        pool.join()
    
    print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))
    
    return


def resize_image(inFile, outFile):
    import warnings
    import skimage.io as io
    import skimage.transform as transf

    new_size = (200, 200)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            inImage = io.imread(inFile)
            outImage = transf.resize(inImage, new_size, mode='reflect')
            io.imsave(outFile, outImage)    
        
    except:
        print("Could not process image: " + inFile, flush=True)
        
    return
    
# -------------------------------------------------------------------------------------- #
# Main module function
# -------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    process_all_images()

