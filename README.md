# 1. DeepLearning Course work
## 1.1 Deep Neural Network
 
## 1.2 OpenCV - Canny Edge Detection, Contours

/homeworks/HW4_DigitsRecognition

![image](https://github.com/YoonjungChoi/CMPE258DL_ObjectDetectionSegmentation_study/assets/20979517/f260b9fd-90b0-43ab-8210-480c2bda9e4b)
  
## 1.3 YouLookOnlyOnce (YoLo) : Object Detection
  ### 1.3.1 Architecture
  <img src="https://user-images.githubusercontent.com/20979517/236288382-3f41f498-e1a7-42fd-84a3-d51d0b5aa680.png" width="350" height="350">
  
  * IoU: Intersection Over Union
    There are two Bounding Boxes; predicted bounding box and ground truth box.
    
    
    
  * Probability Map by K-means
    K-Means Steps:
      1. Determine 'K' value to represent groups.
      2. Randomly select 'K' centroids.
      3. Measure Euclidean Distance between all feature vectors and all centroids.
      4. Assgin feature vectors to nearest a centroid, which have a shortest and smaller distance.
      5. Calculate and Update new centroids (mean of groups, among assigned feature vectors)
      6. repeat 3-5 steps.

## 1.4 Semantic Segmentation
  
 * Deep Convolutional Nneural Network - Encoder: feature extraction (Convolutions, kernels), classification(FeedForward NN)
  
 * Upsampling - Decoder: To get pixel-by-pixel Recognition of objects e.g. Segmentation.

  ### 1.4.1 Upsampling
  The process of noving from lower resolution feature maps to higher resolution feature map, eventually to the resolution of original
  image is what we called "upsampling".
  
 * Nearest Neighbor
 * Bi-Linear Interpolationn
 * Bed of Nails nailing/annchor point based upsampling
 * MaxPooling, remember with element was max (location)
 * Transposed Convolution

# 2. Project: Semantic Segementation with breast cancer datasdet : Attention U-Net

 - https://github.com/AI-Medical-Robotics/Breast-Cancer-Segmentation

# 3. Project 2: YoLo, Object Detection with PawPatrol

