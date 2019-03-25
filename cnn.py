
from keras.models import load_model

def get_prediction_value(model,eye_input):
    prediction=model.predict(eye_input)
    return prediction

def predict(model,left_eye,right_eye):
    left_eye_prediction=get_prediction_value(model,left_eye)
    right_eye_prediction=get_prediction_value(model,right_eye)

    prediction=(left_eye_prediction+right_eye_prediction)/2.0

    if prediction>0.5:
        prediction="open"
    else:
        prediction="close"
    
    return prediction
