import cv2
import dlib
from imutils import face_utils
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.dataset import Attribute, Instance, Instances
import copy

# general global variables
SHAPE_PREDICT_FILE = "shape_predictor_68_face_landmarks.dat"
LANDMARK_INDICES_FILE = 'face_landmark_indices.txt'
OUTPUT_IMAGE = 'facial_detection_output.png'

# classifier and dataset variables
CLASSIFIER_MODEL_PATH = "emotion_recog_fuzzy_model.model"
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

# settings for drawing
FONT = cv2.FONT_HERSHEY_SIMPLEX
LNDMRK_IDX_FONT_SCALE = 0.3
EMTN_LBL_FONT_SCALE = 0.5
LNDMRK_IDX_TEXT_COLOR = (0, 255, 255)
EMTN_LBL_TEXT_COLOR = (255, 255, 255)
CIRCLE_COLOR = (0, 0, 255)
TRIANGLE_COLOR = (255, 0, 0)
FACE_RECTANGLE_COLOR = (0, 255, 0)
CIRCLE_RADIUS = 2
CIRCLE_THICKNESS = -1
THICKNESS = 1

# function to draw triangle on image
def cv2triangle(image, point1, point2, point3, color, thickness):
    cv2.line(image, point1, point2, color, thickness, cv2.LINE_AA)
    cv2.line(image, point1, point3, color, thickness, cv2.LINE_AA)
    cv2.line(image, point2, point3, color, thickness, cv2.LINE_AA)
    return image

# function to output the 3 cosines of a triangle, given the 3 input cartesian coordinates of the triangle
def find_cosines_of_triangle(point1, point2, point3):
    a = ((point3[0]-point2[0])**2+(point3[1]-point2[1])**2)**0.5 #2 and 3
    b = ((point1[0]-point3[0])**2+(point1[1]-point3[1])**2)**0.5 #3 and 1
    c = ((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)**0.5 #1 and 2
    cosine1 = (b**2+c**2-a**2)/(2*b*c)
    cosine2 = (a**2+c**2-b**2)/(2*a*c)
    cosine3 = (a**2+b**2-c**2)/(2*a*b)
    return (cosine1, cosine2, cosine3)

# open triangle indices file, and read all its lines
triangle_indices_file = open(LANDMARK_INDICES_FILE, 'r')
all_triangle_indices = triangle_indices_file.readlines()

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICT_FILE)

# start JVM, create classifier, attributes, dataset
jvm.start(packages=True)
classifier,_ = Classifier.deserialize(CLASSIFIER_MODEL_PATH)
attributes = []
for i in range(54):
    attributes.append(Attribute.create_numeric(DATA_COSINE_ATTRIBUTES[i]))
attributes.append(Attribute.create_nominal(DATA_EMOTION_ATTRIBUTE, DATA_EMOTION_LABELS))
dataset = Instances.create_instances(DATASET_RELATION_NAME, attributes, 0)
dataset.class_is_last()
dataset_copy = copy.deepcopy(dataset)

# initialize webcam video capture
cap = cv2.VideoCapture(0)

while True:
    # load webcam image, convert to b&w, detect all faces
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # loop over all the detected faces
    for (i, rect) in enumerate(rects):
        # determine the face landmark coordinates, convert to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # detect emotion of the subject
        attribute_values = []
        for triangle_indices in all_triangle_indices:
            triangle_indices = triangle_indices.strip()
            index_list = triangle_indices.split(' ')
            point1 = shape[int(index_list[0])]
            point2 = shape[int(index_list[1])]
            point3 = shape[int(index_list[2])]
            (cosine1, cosine2, cosine3) = find_cosines_of_triangle(point1, point2, point3)
            attribute_values.append(cosine1)
            attribute_values.append(cosine2)
            attribute_values.append(cosine3)
        attribute_values.append(Instance.missing_value())
        inst = Instance.create_instance(attribute_values)
        dataset.add_instance(inst)
        pred = classifier.classify_instance(list(dataset)[-1])
        # delete dataset and recreate it to avoid memory leaks due to boundless addition of data-points to dataset
        del dataset
        dataset = copy.deepcopy(dataset_copy)
        detected_emotion = DATA_EMOTION_LABELS[int(pred)]

        # draw coordinates, coordinate index, triangles, face rectangle, and emotions of subject on image
        for i, (x, y) in enumerate(shape):
            org = (x, y)
            cv2.circle(image, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
            cv2.putText(image, str(i), org, FONT, LNDMRK_IDX_FONT_SCALE, LNDMRK_IDX_TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        for triangle_indices in all_triangle_indices:
            triangle_indices = triangle_indices.strip()
            index_list = triangle_indices.split(' ')
            point1 = shape[int(index_list[0])]
            point2 = shape[int(index_list[1])]
            point3 = shape[int(index_list[2])]
            cv2triangle(image, point1, point2, point3, TRIANGLE_COLOR, THICKNESS)
        tl_x=rect.tl_corner().x
        tl_y=rect.tl_corner().y
        br_x=rect.br_corner().x
        br_y=rect.br_corner().y
        cv2.rectangle(image, (tl_x,tl_y), (br_x,br_y), FACE_RECTANGLE_COLOR, THICKNESS)
        cv2.putText(image, detected_emotion, (tl_x,tl_y), FONT, EMTN_LBL_FONT_SCALE, EMTN_LBL_TEXT_COLOR, THICKNESS, cv2.LINE_AA)

    # show the output image with the face detections, facial landmarks, detected emotions, etc.
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # cv2.imwrite(OUTPUT_IMAGE, image)
        break

# perform close operations
cv2.destroyAllWindows()
cap.release()
triangle_indices_file.close()
jvm.stop()
