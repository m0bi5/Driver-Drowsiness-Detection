import dlib
import cv2
import numpy as np
import math

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	dim = (h,width)

	# resize the image
	resized = cv2.resize(image, dim, interpolation=inter)

	return resized

def rect_to_bb(rect):
	# Converts the bounding box predicted by dlib to the OpenCv's (x, y, w, h) format
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# Converts points of interest from (x,y) to [x y] format 
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

def distance(a,b):
	x1,y1=a
	x2,y2=b
	return math.sqrt((abs(x1-x2)**2)+(abs(y1-y2)**2))

def calculate_ear(eye):
	#Formula given in https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
	ear = ( distance(eye[1],eye[5]) + distance(eye[2],eye[4]) ) / (2*distance(eye[0],eye[3]))
	return ear

detector = dlib.get_frontal_face_detector()  				#dlib's face detector (uses HOG)
predictor = dlib.shape_predictor('facial_landmarks.dat')	#dlib's pretrained model to recognise facial features (eyes,jawline,mouth etc)

cam=cv2.VideoCapture(0)

while True:
	ret,image = cam.read()
	image = resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)			#HOG takes grayscale input

	faces = detector(gray, 1)


	for (i, face) in enumerate(faces):
		facial_features = predictor(gray, face)
		facial_features = shape_to_np(facial_features)


		left_eye=facial_features[36:42]
		right_eye=facial_features[42:48]

		ear=(calculate_ear(left_eye)+calculate_ear(right_eye))/2

		#Drawing circles around the eyes
		for (x, y) in left_eye:
			cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
		for (x, y) in right_eye:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	cv2.imshow("Eye detection", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
