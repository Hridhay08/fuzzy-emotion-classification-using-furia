# fuzzy-emotion-classification-using-furia

Purpose:
This project aims to classify facial images into one of the following seven emotions - anger, contempt, disgust, fear, happiness, neutral, sadness, surprise.

Working Summary:
The dataset used is CK+ (Extended Cohn Kanade), which is useful for emotion classification applications. It comprises of images labelled with their corresponding facial emotions. Facial landmark detection (detection of 68 coordinates on the face) is applied on all images present in the dataset. For each image, 18 triangles are constructed by taking groups of 3 coordinates, and the cosines of the 3 angles of each triangle are computed. A modified dataset is derived from the CK+ dataset, which only comprises of the 54 cosine values, and the corresponding emotion label. This modified dataset is used to train the classifier. The classifier used is based on the Fuzzy Unordered Rule Induction Algorithm (FURIA), which generates fuzzy rules to perform rule-based classification. Live image feed is captured from webcam, and facial landmarks are detected in each image. The triangles and their corresponding cosines are computed for each image, and passed to the classifier. The classifier outputs its prediction, which is then displayed along with the webcam feed to the user.

Software Stack:
1. dlib - detection of facial landmarks, and their corresponding coordinates
2. opencv - image manipulation, webcam capture, display, etc.
3. weka - generating the dataset and the classifier, and classifying new instances
4. python3.8 - the development environment used
5. default-jdk - javabridge for python

Python Dependencies:
1. numpy
2. opencv-contrib-python
3. imutils
4. dlib
5. python-javabridge
6. python-weka-wrapper3
7. pillow
8. matplotlib
9. pygraphviz

Additional Package Requirements (for Ubuntu):
1. build-essential
2. python3-dev
3. python3-pip
4. python3-venv
5. default-jdk

File-by-File Explanation:
(sorted by importance and/or order of occurence in workflow)
1. cosine_dataset_generator.py - responsible for generating the cosine dataset in .arff format for weka
2. emotion_cosine_dataset.arff - the generated dataset comprising of 54 cosine attributes, one emotion attribute, and around 920 data points
3. fuzzyUnorderedRuleInduction1.0.2.zip - FURIA package for weka
4. furia_package_installer.py - installs FURIA package in weka
5. fuzzy_model_generator.py - uses the .arff file generated to train the FURIA classifier, outputs the .model file, and performs cross-validation
6. emotion_recog_fuzzy_model.model - the outputted .model file
7. fuzzy_training_logs.txt - 
8. demo.py - a demonstration of the emotion classification, by taking a live image feed from the webcam and displaying its corresponding emotion
9. face_landmark_indices.txt - comprises of the indices of facial landmarks that comprise each triangle
10. shape_predictor_68_face_landmarks.dat - the model file used by dlib to predict facial landmarks 
11. cosine_attribute_string_generator.py - a convenience program to print out the list of cosine attributes
12. requirements.txt - list of python libraries to be installed
