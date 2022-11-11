import cv2
import dlib
from imutils import face_utils
import weka.core.jvm as jvm
from weka.core.dataset import Attribute, Instance, Instances
import os

# facial landmark detection variables
SHAPE_PREDICT_FILE = "shape_predictor_68_face_landmarks.dat"
LANDMARK_INDICES_FILE = 'face_landmark_indices.txt'

# arff, classifier and dataset variables
ARFF_FILE = 'emotion_cosine_dataset.arff'
CK_DATASET_PATH = 'ck+'
DATASET_RELATION_NAME = "emotion"
DATA_COSINE_ATTRIBUTES = \
["COSINE0", "COSINE1", "COSINE2", "COSINE3", "COSINE4", "COSINE5", "COSINE6", "COSINE7", "COSINE8",
"COSINE9", "COSINE10", "COSINE11", "COSINE12", "COSINE13", "COSINE14", "COSINE15", "COSINE16", "COSINE17",
"COSINE18", "COSINE19", "COSINE20", "COSINE21", "COSINE22", "COSINE23", "COSINE24", "COSINE25", "COSINE26",
"COSINE27", "COSINE28", "COSINE29", "COSINE30", "COSINE31", "COSINE32", "COSINE33", "COSINE34", "COSINE35",
"COSINE36", "COSINE37", "COSINE38", "COSINE39", "COSINE40", "COSINE41", "COSINE42", "COSINE43", "COSINE44",
"COSINE45", "COSINE46", "COSINE47", "COSINE48", "COSINE49", "COSINE50", "COSINE51", "COSINE52", "COSINE53"]
DATA_EMOTION_ATTRIBUTE = "emotion_class"
DATA_EMOTION_LABELS = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# function to output the 3 cosines of a triangle, given the 3 input cartesian coordinates of the triangle
def find_cosines_of_triangle(point1, point2, point3):
    a = ((point3[0]-point2[0])**2+(point3[1]-point2[1])**2)**0.5 #2 and 3
    b = ((point1[0]-point3[0])**2+(point1[1]-point3[1])**2)**0.5 #3 and 1
    c = ((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)**0.5 #1 and 2
    cosine1 = (b**2+c**2-a**2)/(2*b*c)
    cosine2 = (a**2+c**2-b**2)/(2*a*c)
    cosine3 = (a**2+b**2-c**2)/(2*a*b)
    return (cosine1, cosine2, cosine3)

# start JVM, create attributes and dataset
jvm.start(packages=True)
attributes = []
for i in range(54):
    attributes.append(Attribute.create_numeric(DATA_COSINE_ATTRIBUTES[i]))
attributes.append(Attribute.create_nominal(DATA_EMOTION_ATTRIBUTE, DATA_EMOTION_LABELS))
dataset = Instances.create_instances(DATASET_RELATION_NAME, attributes, 0)
dataset.class_is_last()

# open the file which contains the triangle vertex indices
triangle_indices_file = open(LANDMARK_INDICES_FILE, 'r')
# read all lines in the file
all_triangle_indices = triangle_indices_file.readlines()

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICT_FILE)

# iterate through CK+ emotion directories
for emotion_name in DATA_EMOTION_LABELS:
    directory=CK_DATASET_PATH+'/'+emotion_name

    # inside each directory, iterate through all image files
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        print(file_path)
        if os.path.isfile(file_path):

            # load the input image and convert it to grayscale
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = detector(gray, 0)
            # assume only one face is there in the image
            rect = rects[0]
            # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # find the 54 cosines from the image
            attribute_values = []
            for triangle_indices in all_triangle_indices:
                triangle_indices = triangle_indices.strip()
                index_list = triangle_indices.split(' ')
                # get the 3 coordinates of the triangle from the list of indices
                point1 = shape[int(index_list[0])]
                point2 = shape[int(index_list[1])]
                point3 = shape[int(index_list[2])]
                # find the 3 cosines of the triangle
                (cosine1, cosine2, cosine3) = find_cosines_of_triangle(point1, point2, point3)
                attribute_values.append(cosine1)
                attribute_values.append(cosine2)
                attribute_values.append(cosine3)
            emotion_index = DATA_EMOTION_LABELS.index(emotion_name)
            attribute_values.append(emotion_index)
            inst = Instance.create_instance(attribute_values)
            dataset.add_instance(inst)

# write dataset to arff file
arff_file = open(ARFF_FILE,'w')
arff_file.write(str(dataset))

# perform close operations
arff_file.close()
triangle_indices_file.close()
cv2.destroyAllWindows()
jvm.stop()