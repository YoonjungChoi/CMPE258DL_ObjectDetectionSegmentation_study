# 1. DeepLearning Course work
## 1.1 Deep Neural Network

[Lecture: Harry Li 's github](https://github.com/hualili/opencv/tree/master/deep-learning-2022s)

- [Lecture: Loss Function](https://github.com/hualili/opencv/blob/master/deep-learning-2022s/2022S-103a-notation-neuro-loss-function-2022-2-8.pdf)
- [Lecture: Gradient Descent](https://github.com/hualili/opencv/blob/master/deep-learning-2022s/2022S-105c-%2320-2021S-4gradient-descent-v2-final-2021-2-8.pdf)

Simple implementation Code:

```
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y= np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

def simple_train(x, y, weight, bias, learning_rate, iters):
        
    for i in range(iters):
       
        pred = weight*x + bias
        weight_deriv = -2*x * (y - pred)
        bias_deriv = -2*(y - pred)

        #update weight and bias
        weight -= (np.sum(weight_deriv) / len(x)) * learning_rate
        bias -= (np.sum(bias_deriv) / len(x)) * learning_rate
        
        
        if i % 10 == 0:
            cost = np.sum((y-pred)**2)/len(x)
            print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))
    return weight, bias


w, b = simple_train(x, y, 1, 1, 0.01, 100)
```
 
## 1.2 OpenCV - Canny Edge Detection, Contours

/homeworks/HW4_DigitsRecognition

![image](https://github.com/YoonjungChoi/CMPE258DL_ObjectDetectionSegmentation_study/assets/20979517/f260b9fd-90b0-43ab-8210-480c2bda9e4b)


1. each frame of video can be processed
2. crops bounding boxes by using canny edge detection and contrours. (crop them without destroy resolutions)
3. give the cropped images to the classifier to detect an number
4. display the number 

  
## 1.3 YouLookOnlyOnce (YoLo) : Object Detection

  ### 1.3.1 Architecture
  
  https://github.com/hualili/opencv/blob/master/deep-learning-2022s/2022F-108a-Yolo-architecture-loss-function-2022-10-10.pdf
  
  
  * IoU: Intersection Over Union
    There are two Bounding Boxes; predicted bounding box and ground truth box.
    
   <img src="https://user-images.githubusercontent.com/20979517/236288382-3f41f498-e1a7-42fd-84a3-d51d0b5aa680.png" width="250" height="250">

    
  * Probability Map by K-means
    K-Means Steps:
      1. Determine 'K' value to represent groups.
      2. Randomly select 'K' centroids.
      3. Measure Euclidean Distance between all feature vectors and all centroids.
      4. Assgin feature vectors to nearest a centroid, which have a shortest and smaller distance.
      5. Calculate and Update new centroids (mean of groups, among assigned feature vectors)
      6. repeat 3-5 steps.

## 1.4 Semantic Segmentation
  
 * Deep Convolutional Neural Network

![image](https://github.com/YoonjungChoi/CMPE258DL_ObjectDetectionSegmentation_study/assets/20979517/cac14e6a-66fb-401c-8cb6-bbf64c19d37a)

 
   - Encoder: feature extraction (Convolutions, kernels), classification(FeedForward NN)
   
   - Decoder: To get pixel-by-pixel Recognition of objects e.g. Segmentation. (Up Sampling)

  ### 1.4.1 Upsampling
  
  The process of noving from lower resolution feature maps to higher resolution feature map, eventually to the resolution of original
  image is what we called "upsampling".

  ![image](https://github.com/YoonjungChoi/CMPE258DL_ObjectDetectionSegmentation_study/assets/20979517/6718cce1-4e77-4a6f-832a-f1b5f5a0567c)
  
 * Nearest Neighbor
 * Bi-Linear Interpolationn
 * Bed of Nails nailing/anchor point based upsampling
 * MaxPooling, remember with element was max (location)
 * Transposed Convolution

# 2. Project: Semantic Segementation with breast cancer datasdet : Attention U-Net

 - Team Project Repository: https://github.com/AI-Medical-Robotics/Breast-Cancer-Segmentation

more explanations - [U-Net](https://jinglescode.github.io/2019/11/07/biomedical-image-segmentation-u-net/) ,  [Attention U-Net](https://jinglescode.github.io/2019/12/08/biomedical-image-segmentation-u-net-attention/) 

# 3. Project 2: YoLo, Object Detection with PawPatrol



