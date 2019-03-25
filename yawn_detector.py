import numpy as np

def get_mouth(features):
    top_lip=np.vstack([features[50:53],features[61:64]])
    bottom_lip= np.vstack([features[65:68],features[56:60]])
    return top_lip,bottom_lip


def mouth_open(features):
    top_lip,bottom_lip=get_mouth(features)
    bottom_lip_center = np.mean(bottom_lip, axis=0)
    top_lip_center = np.mean(top_lip, axis=0)
    #print(top_lip_center,bottom_lip_center)
    
   # lip_distance = abs(top_lip_center - bottom_lip_center)
    #return lip_distance