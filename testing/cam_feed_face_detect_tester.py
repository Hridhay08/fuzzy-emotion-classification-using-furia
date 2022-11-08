# import the necessary packages
import cv2
import dlib
from imutils import face_utils

# settings for drawing
FONT = cv2.FONT_HERSHEY_SIMPLEX
LNDMRK_IDX_FONT_SCALE = 0.3
LNDMRK_IDX_TEXT_COLOR = (0, 255, 255)
CIRCLE_COLOR = (0, 0, 255)
TRIANGLE_COLOR = (255, 0, 0)
FACE_RECTANGLE_COLOR = (0, 255, 0)
CIRCLE_RADIUS = 2
CIRCLE_THICKNESS = -1
THICKNESS = 1

# global variables
SHAPE_PREDICT_FILE = "shape_predictor_68_face_landmarks.dat"
LANDMARK_INDICES_FILE = 'face_landmark_indices.txt'
OUTPUT_FILE = 'sample_detected_facial_landmarks.png'

# function to draw triangle on image
def cv2triangle(image, point1, point2, point3, color, thickness):
    cv2.line(image, point1, point2, color, thickness, cv2.LINE_AA)
    cv2.line(image, point1, point3, color, thickness, cv2.LINE_AA)
    cv2.line(image, point2, point3, color, thickness, cv2.LINE_AA)
    return image

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICT_FILE)

cap = cv2.VideoCapture(0)

while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over all the detected faces
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # draw coordinates and coordinate index on image
        for i, (x, y) in enumerate(shape):
            org = (x, y)
            cv2.circle(image, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
            cv2.putText(image, str(i), org, FONT, LNDMRK_IDX_FONT_SCALE, LNDMRK_IDX_TEXT_COLOR, THICKNESS, cv2.LINE_AA)

        # draw triangles on image
        file1 = open(LANDMARK_INDICES_FILE, 'r')
        lines = file1.readlines()
        for line in lines:
            line = line.strip()
            index_list = line.split(' ')
            point1 = shape[int(index_list[0])]
            point2 = shape[int(index_list[1])]
            point3 = shape[int(index_list[2])]
            cv2triangle(image, point1, point2, point3, TRIANGLE_COLOR, THICKNESS)
        file1.close()

        # draw face rectangle on image
        tl_x=rect.tl_corner().x
        tl_y=rect.tl_corner().y
        br_x=rect.br_corner().x
        br_y=rect.br_corner().y
        cv2.rectangle(image, (tl_x,tl_y), (br_x,br_y), FACE_RECTANGLE_COLOR, THICKNESS)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # cv2.imwrite(OUTPUT_FILE, image)
        break

cv2.destroyAllWindows()
cap.release()
