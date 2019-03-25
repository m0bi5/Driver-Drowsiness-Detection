from helper import *

def ear_formula(eye):
	#Formula given in https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
	ear = ( distance(eye[1],eye[5]) + distance(eye[2],eye[4]) ) / (2*distance(eye[0],eye[3]))
	return ear

def get_eyes(features):
	left_eye=features[36:42]
	right_eye=features[42:48]
	return left_eye,right_eye

def calculate_ear(features):
	left_eye,right_eye=get_eyes(features)
	ear=(ear_formula(left_eye)+ear_formula(right_eye))/2
	return ear

