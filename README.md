# face_recognition_using_SNN

## Summary  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This project is a real-time face verification system that uses a Siamese Neural Network (SNN) to decide whether two face images belong to the same person. It's built using TensorFlow/Keras for deep learning, OpenCV for capturing images from the webcam, and Kivy to provide a simple and interactive desktop application.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The system works by comparing a live image from the webcam (called the input image) with a set of stored images (called verification images). Behind the scenes, the Siamese network takes two images at a time and extracts meaningful features (called embeddings) using a shared convolutional neural network. It then calculates how similar the two embeddings are using the L1 distance, a simple difference calculation. If the similarity score is high enough, the model predicts that the images are of the same person.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; During training, the model learns by looking at both similar pairs (anchor and positive) and different pairs (anchor and negative), helping it understand what makes two faces match or not. Once trained, the model is saved and used inside the Kivy app. When a user clicks the “Verify” button, the app captures an image from the webcam and compares it to all the stored verification images. If enough comparisons pass a confidence threshold, the user is marked as “Verified” in real time.

**Algorithm:** Siamese Neural Network (SNN)  
**Framework:** TensorFlow/Keras, OpenCV, Kivy  
**Support Libraries:** NumPy, uuid, Matplotlib, Random	   
**File handling & paths:** OS module   
**Evaluation:** Precision, Recall, Detection Threshold, Verification Therehold   

## Directory Structure
***project_root/***   
&nbsp;&nbsp;&nbsp; - **data/**    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - *anchor/*        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Anchor images (collected via webcam)      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - *positive/*      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Positive images (same person)     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - *negative/*      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Negative images (from LFW dataset)     

&nbsp;&nbsp;&nbsp; - **application_data/**     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - *input_image/*          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Real-time input image captured for verification      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - *verification_images/*  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Stored reference images to compare with        
