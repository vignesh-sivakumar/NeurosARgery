import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
import pydicom
import matplotlib.pyplot as plt
import sys
import getopt
import os.path

def CropIt(dicom_array):
    img = np.uint8(ds.pixel_array)
    img = np.stack((img,)*3, axis=-1)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((4,3), np.uint8) 
    gray = cv2.erode(gray, kernel, iterations=3)
    gray = cv2.dilate(gray, kernel, iterations=3)
    
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(gray,-1,kernel)
    Z = gray.reshape((-1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    km = res.reshape((dst.shape))
    
    ret, thresh = cv2.threshold(km,0,100,cv2.THRESH_OTSU)
    
    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[thresh!=0] = np.array((200,150,50))
    blended = cv2.addWeighted(img,0.7,colormask,0.1,0)
    
    ret, markers = cv2.connectedComponents(thresh)

    #Get the area taken by each component. Ignore label 0 since this is the background.
    if ret!=2:
        marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    else:
        marker_area = [np.sum(markers==m) for m in range(np.max(markers))] 
    
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
    
    #Get pixels which correspond to the brain
    brain_mask = markers==largest_component

    brain_out = img.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_out[brain_mask==False] = (0,0,0)
    brain_out = brain_out[:,:,0]
    
    return(brain_out)

def CropTumor(dicom_array):
    img = np.uint8(ds.pixel_array)
    img = np.stack((img,)*3, axis=-1)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((4,3), np.uint8) 
    gray = cv2.erode(gray, kernel, iterations=3)
    gray = cv2.dilate(gray, kernel, iterations=3)
    
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(gray,-1,kernel)
    Z = gray.reshape((-1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    km = res.reshape((dst.shape))
    
    trial = img.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    trial[km<120] = (0,0,0)
    trial = trial[:,:,0]
    
    return(trial)

def CropTumorless(brain,tumor):
    tumorless=np.zeros((brain.shape))
    #tumorless[tumor!=0]=0
    #tumorless[tumor==0]=brain
    
    for i in range(brain.shape[0]):
        for j in range(brain.shape[1]):
            if(tumor[i][j]==0):
                tumorless[i][j]=brain[i][j]
            else:
                tumorless[i][j]=0
    
    return(tumorless)

if __name__ == '__main__':

    fullcmd = sys.argv
    argumentList = fullcmd[1:]

    unixOptions = "ho:i:v"
    gnuOptions  = ["help","output=","input=","verbose"]
    
    try:
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.err as err:
        print(str(err))
        sys.exit(2)
    
    for currArg, currValue in arguments:
        if currArg in ("-v","--verbose"):
            print("enabling verbose")
        elif currArg in ("-h","--help"):
            print("Help")
        elif currArg in ("-o","--output"):
            #print(("enabling output (%s)")%(currValue))
            opdir = currValue
        elif currArg in ("-i","--input"):
            ipdir = currValue

    InputPath = 'original'
    TumorPath = 'tumor'
    TumorlessPath = 'tumorless'
    num_files = len([f for f in os.listdir(InputPath)
                    if os.path.isfile(os.path.join(InputPath, f))])

    start = int(num_files*0.45)
    end = int(num_files*0.85)

    if start<10 and end<10:
        flag=1
    elif start<10 and end>10 and end<100:
        flag=2
    elif start>10 and end<100:
        flag=3
    else:
        flag=4

    startTumor = 1
    startTumorless = 1
    
    if flag==1:
        for i in range(start,end):
            ds = pydicom.dcmread(InputPath+'\IMG000'+str(i)+'.dcm')
            dicomImage = ds.pixel_array
            
            brain = np.zeros(dicomImage.shape)
            brain = CropIt(dicomImage)
            ds.pixel_array.flat=brain.flat
            ds.PixelData = ds.pixel_array.tobytes()
            
            tumor = np.zeros(dicomImage.shape)
            tumor = CropTumor(dicomImage)
            ds.pixel_array.flat=tumor.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorPath+"\IMG000"+str(startTumor)+".dcm")
            startTumor = startTumor+1
            
            tumorless = np.zeros(dicomImage.shape)
            tumorless = CropTumorless(brain,tumor)
            ds.pixel_array.flat=tumorless.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorlessPath+"\IMG000"+str(startTumorless)+".dcm")
            startTumorless = startTumorless+1
            
    elif flag==2:
        for i in range(start,9):
            ds = pydicom.dcmread(InputPath+'\IMG000'+str(i)+'.dcm')
            dicomImage = ds.pixel_array
            
            brain = np.zeros(dicomImage.shape)
            brain = CropIt(dicomImage)
            ds.pixel_array.flat=brain.flat
            ds.PixelData = ds.pixel_array.tobytes()
            
            tumor = np.zeros(dicomImage.shape)
            tumor = CropTumor(dicomImage)
            ds.pixel_array.flat=tumor.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorPath+"\IMG000"+str(startTumor)+".dcm")
            startTumor = startTumor+1
            
            tumorless = np.zeros(dicomImage.shape)
            tumorless = CropTumorless(brain,tumor)
            ds.pixel_array.flat=tumorless.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorlessPath+"\IMG000"+str(startTumorless)+".dcm")
            startTumorless = startTumorless+1
            
        for i in range(10,end):
            ds = pydicom.dcmread(InputPath+'\IMG00'+str(i)+'.dcm')
            dicomImage = ds.pixel_array
            
            
            brain = np.zeros(dicomImage.shape)
            brain = CropIt(dicomImage)
            ds.pixel_array.flat=brain.flat
            ds.PixelData = ds.pixel_array.tobytes()
            
            tumor = np.zeros(dicomImage.shape)
            tumor = CropTumor(dicomImage)
            ds.pixel_array.flat=tumor.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorPath+"\IMG000"+str(startTumor)+".dcm")
            startTumor = startTumor+1
            
            tumorless = np.zeros(dicomImage.shape)
            tumorless = CropTumorless(brain,tumor)
            ds.pixel_array.flat=tumorless.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorlessPath+"\IMG000"+str(startTumorless)+".dcm")
            startTumorless = startTumorless+1
            
    elif flag==3:
        for i in range(start,end):
            ds = pydicom.dcmread('original\IMG00'+str(i)+'.dcm')
            dicomImage = ds.pixel_array
            
            
            brain = np.zeros(dicomImage.shape)
            brain = CropIt(dicomImage)
            ds.pixel_array.flat=brain.flat
            ds.PixelData = ds.pixel_array.tobytes()
            
            tumor = np.zeros(dicomImage.shape)
            tumor = CropTumor(dicomImage)
            ds.pixel_array.flat=tumor.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorPath+"\IMG000"+str(startTumor)+".dcm")
            startTumor = startTumor+1
            
            tumorless = np.zeros(dicomImage.shape)
            tumorless = CropTumorless(brain,tumor)
            ds.pixel_array.flat=tumorless.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorlessPath+"\IMG000"+str(startTumorless)+".dcm")
            startTumorless = startTumorless+1
            
    else:
        for i in range(start,9):   
            ds = pydicom.dcmread(InputPath+'\IMG000'+str(i)+'.dcm')
            dicomImage = ds.pixel_array
            
            
            brain = np.zeros(dicomImage.shape)
            brain = CropIt(dicomImage)
            ds.pixel_array.flat=brain.flat
            ds.PixelData = ds.pixel_array.tobytes()
            
            tumor = np.zeros(dicomImage.shape)
            tumor = CropTumor(dicomImage)
            ds.pixel_array.flat=tumor.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorPath+"\IMG000"+str(startTumor)+".dcm")
            startTumor = startTumor+1
            
            tumorless = np.zeros(dicomImage.shape)
            tumorless = CropTumorless(brain,tumor)
            ds.pixel_array.flat=tumorless.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorlessPath+"\IMG000"+str(startTumorless)+".dcm")
            startTumorless = startTumorless+1
            
        for i in range(10,99):   
            ds = pydicom.dcmread(InputPath+'\IMG00'+str(i)+'.dcm')
            dicomImage = ds.pixel_array
            
            
            brain = np.zeros(dicomImage.shape)
            brain = CropIt(dicomImage)
            ds.pixel_array.flat=brain.flat
            ds.PixelData = ds.pixel_array.tobytes()
            
            tumor = np.zeros(dicomImage.shape)
            tumor = CropTumor(dicomImage)
            ds.pixel_array.flat=tumor.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorPath+"\IMG000"+str(startTumor)+".dcm")
            startTumor = startTumor+1
            
            tumorless = np.zeros(dicomImage.shape)
            tumorless = CropTumorless(brain,tumor)
            ds.pixel_array.flat=tumorless.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorlessPath+"\IMG000"+str(startTumorless)+".dcm")
            startTumorless = startTumorless+1
            
            
        for i in range(100,end):   
            ds = pydicom.dcmread(InputPath+'\IMG0'+str(i)+'.dcm')
            dicomImage = ds.pixel_array
            
            
            brain = np.zeros(dicomImage.shape)
            brain = CropIt(dicomImage)
            ds.pixel_array.flat=brain.flat
            ds.PixelData = ds.pixel_array.tobytes()
            
            tumor = np.zeros(dicomImage.shape)
            tumor = CropTumor(dicomImage)
            ds.pixel_array.flat=tumor.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorPath+"\IMG000"+str(startTumor)+".dcm")
            startTumor = startTumor+1
            
            tumorless = np.zeros(dicomImage.shape)
            tumorless = CropTumorless(brain,tumor)
            ds.pixel_array.flat=tumorless.flat
            ds.PixelData = ds.pixel_array.tobytes()
            ds.save_as(TumorlessPath+"\IMG000"+str(startTumorless)+".dcm")
            startTumorless = startTumorless+1

print("Done cropping !!")
sys.exit()
