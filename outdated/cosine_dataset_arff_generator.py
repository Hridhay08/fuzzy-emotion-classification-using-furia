# import the necessary packages
import cv2
import dlib
from imutils import face_utils
import os

# general global variables
SHAPE_PREDICT_FILE = "shape_predictor_68_face_landmarks.dat"
LANDMARK_INDICES_FILE = 'face_landmark_indices.txt'
CK_DATASET_PATH = 'CK+'
ARFF_FILE = 'emotion_cosine_dataset.arff'
EMOTION_LABELS = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# arff file global variables
RELATION_STRING = "@relation emotion"
COSINE_ATTRIBUTES = ["@attribute COSINE0 numeric",
"@attribute COSINE1 numeric",
"@attribute COSINE2 numeric",
"@attribute COSINE3 numeric",
"@attribute COSINE4 numeric",
"@attribute COSINE5 numeric",
"@attribute COSINE6 numeric",
"@attribute COSINE7 numeric",
"@attribute COSINE8 numeric",
"@attribute COSINE9 numeric",
"@attribute COSINE10 numeric",
"@attribute COSINE11 numeric",
"@attribute COSINE12 numeric",
"@attribute COSINE13 numeric",
"@attribute COSINE14 numeric",
"@attribute COSINE15 numeric",
"@attribute COSINE16 numeric",
"@attribute COSINE17 numeric",
"@attribute COSINE18 numeric",
"@attribute COSINE19 numeric",
"@attribute COSINE20 numeric",
"@attribute COSINE21 numeric",
"@attribute COSINE22 numeric",
"@attribute COSINE23 numeric",
"@attribute COSINE24 numeric",
"@attribute COSINE25 numeric",
"@attribute COSINE26 numeric",
"@attribute COSINE27 numeric",
"@attribute COSINE28 numeric",
"@attribute COSINE29 numeric",
"@attribute COSINE30 numeric",
"@attribute COSINE31 numeric",
"@attribute COSINE32 numeric",
"@attribute COSINE33 numeric",
"@attribute COSINE34 numeric",
"@attribute COSINE35 numeric",
"@attribute COSINE36 numeric",
"@attribute COSINE37 numeric",
"@attribute COSINE38 numeric",
"@attribute COSINE39 numeric",
"@attribute COSINE40 numeric",
"@attribute COSINE41 numeric",
"@attribute COSINE42 numeric",
"@attribute COSINE43 numeric",
"@attribute COSINE44 numeric",
"@attribute COSINE45 numeric",
"@attribute COSINE46 numeric",
"@attribute COSINE47 numeric",
"@attribute COSINE48 numeric",
"@attribute COSINE49 numeric",
"@attribute COSINE50 numeric",
"@attribute COSINE51 numeric",
"@attribute COSINE52 numeric",
"@attribute COSINE53 numeric"]
EMOTION_ATTRIBUTE = "@attribute emotion_class {anger,contempt,disgust,fear,happiness,neutral,sadness,surprise}"
DATA_STRING = "@data"

# function to output the 3 cosines of a triangle, given the 3 input cartesian coordinates of the triangle
def find_cosines_of_triangle(point1, point2, point3):
    a = ((point3[0]-point2[0])**2+(point3[1]-point2[1])**2)**0.5 #2 and 3
    b = ((point1[0]-point3[0])**2+(point1[1]-point3[1])**2)**0.5 #3 and 1
    c = ((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)**0.5 #1 and 2
    cosine1 = (b**2+c**2-a**2)/(2*b*c)
    cosine2 = (a**2+c**2-b**2)/(2*a*c)
    cosine3 = (a**2+b**2-c**2)/(2*a*b)
    return (cosine1, cosine2, cosine3)

# open arff file in write mode and initialize some things to be written to the file
arff_file = open(ARFF_FILE, 'w')
write_string_for_arff_file = RELATION_STRING+'\n\n'
for attribute in COSINE_ATTRIBUTES:
    write_string_for_arff_file += attribute+'\n'
write_string_for_arff_file += EMOTION_ATTRIBUTE+'\n'
write_string_for_arff_file += '\n'
write_string_for_arff_file += DATA_STRING+'\n'

# open the file which contains the triangle vertex indices
triangle_indices_file = open(LANDMARK_INDICES_FILE, 'r')
# read all lines in the file
all_triangle_indices = triangle_indices_file.readlines()

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICT_FILE)

# iterate through CK+ emotion directories
for emotion_name in EMOTION_LABELS:
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

            data_string = ''
            # find the 54 cosines from the image
            for triangle_indices in all_triangle_indices:
                triangle_indices = triangle_indices.strip()
                index_list = triangle_indices.split(' ')
                # get the 3 coordinates of the triangle from the list of indices
                point1 = shape[int(index_list[0])]
                point2 = shape[int(index_list[1])]
                point3 = shape[int(index_list[2])]
                # find the 3 cosines of the triangle
                (cosine1, cosine2, cosine3) = find_cosines_of_triangle(point1, point2, point3)
                # store the cosine values in a comma separated string
                data_string+=str(cosine1).ljust(25, " ")+','+str(cosine2).ljust(25, " ")+','+str(cosine3).ljust(25, " ")+','
            # append the emotion name to the data
            data_string += emotion_name
            # append the data to arff write-string
            write_string_for_arff_file+=data_string+'\n'

arff_file.write(write_string_for_arff_file)

cv2.destroyAllWindows()
arff_file.close()
triangle_indices_file.close()