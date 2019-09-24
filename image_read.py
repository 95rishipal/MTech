import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
  
img = cv2.imread('data\\train\\cats\\cat.0.jpg', cv2.IMREAD_GRAYSCALE) 

def showIm(path):
    cv2.imshow('image', path) 
    cv2.waitKey(0) 
    cv2.destroyWindow('image') 
 
