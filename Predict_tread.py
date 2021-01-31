import os
import time
from numpy import asarray
import cv2
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import threading
import queue


class predict_worker(threading.Thread):
    def __init__(self,model, face_obj, frame ):
        threading.Thread.__init__(self)
        self.model=model
        self.face_obj=face_obj
        self.frame=frame
        self.extracts=self.extract_face(self.frame)
        print("Initialized thread" + str(self))

    def extract_face(self, pixels):
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        resized_image = cv2.resize(face, (224, 224))
        return resized_image

    def get_embeddings(self, pixls):
        samples = asarray(pixls, 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = self.model.predict(samples)
        return yhat

    def is_match(self, known_embedding, candidate_embedding, k_name, thresh=0.5):
        score = cosine(known_embedding, candidate_embedding)
        if score <= thresh:
            print('>face is a Match (%.3f <= %.3f)' % (score, thresh), k_name)
            return True
        else:
            # print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh),k_name)
            return False

    def run(self):
        pr_lst = [self.face_obj['data'], self.extracts]
        embeds=self.get_embeddings(pr_lst)
        self.is_match(embeds[0], embeds[1], self.face_obj['name'])
