import dlib
from yawn_detector import *
from helper import *
from ear_calculate import *
from cnn import *

detector = dlib.get_frontal_face_detector()  				#dlib's face detector (uses HOG)
predictor = dlib.shape_predictor('facial_landmarks.dat')	#dlib's pretrained model to recognise facial features (uses regression trees)
model = load_model('drowsyv3.hd5')							#Trained model for predicting state of the eyes

cam=cv2.VideoCapture(0)

while True:
	ret,image = cam.read()
	image = resize(image, width=500)
	cnn_image=image.copy()
	gray = cv2.cvtColor(cnn_image, cv2.COLOR_BGR2GRAY)				#Gray input for CNN
	brighter_image = increase_brightness(image)
	equalized_image = histogram_equalization(brighter_image)
	faces = detector(equalized_image, 1)

	for (i, face) in enumerate(faces):
		facial_features = predictor(equalized_image, face)
		facial_features = shape_to_np(facial_features)
		ear=calculate_ear(facial_features)

		left_eye_ear,right_eye_ear = get_eyes(facial_features) #Just for drawing

		mouth_width=mouth_open(facial_features)

		right_eye_cnn = reshape_eye(gray, eye_points=right_eye_ear)
		left_eye_cnn = reshape_eye(gray, eye_points=left_eye_ear)

		eye_state_cnn = predict(model,left_eye_cnn,right_eye_cnn)

		cv2.putText(cnn_image, eye_state_cnn, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
		cv2.putText(image, str(ear), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
		
		#Drawing circles around the eyes
		for (x, y) in left_eye_ear:
			cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
		for (x, y) in right_eye_ear:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

		top_lip,bottom_lip=get_mouth(facial_features)
		for (x, y) in top_lip:
			cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
		for (x, y) in bottom_lip:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	
	cv2.imshow("EAR", image)
	
	cv2.imshow("CNN", cnn_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()