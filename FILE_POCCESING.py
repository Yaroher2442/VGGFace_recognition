import os
from numpy import asarray
import cv2
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


class Face_recognition():
    def __init__(self):
        self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.detector = MTCNN()
        self.data_path = os.path.join(os.getcwd(), 'faces')
        self.faces = [self.extract_face(f) for f in os.listdir(self.data_path)]
        print("Initialized class" + str(self))

    def extract_face(self, filename):
        pixels = cv2.imread(os.path.join(self.data_path, filename))
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        resized_image = cv2.resize(face, (224, 224))
        return resized_image

    def get_embeddings(self):
        faces = self.faces
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = self.model.predict(samples)
        return yhat

    def is_match(self, known_embedding, candidate_embedding, k_name
                 , cand_name, thresh=0.5):
        score = cosine(known_embedding, candidate_embedding)
        if score <= thresh:
            print('>face is a Match (%.3f <= %.3f)' % (score, thresh), k_name, cand_name)
            return True
        else:
            print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh), k_name, cand_name)
            return False

    def processing(self):
        faces_list = os.listdir(self.data_path)
        embeddings = self.get_embeddings()
        print(embeddings)
        for face_name in faces_list:
            known = embeddings[faces_list.index(face_name)]
            for emb in embeddings:
                cand_name = faces_list[embeddings.tolist().index(emb.tolist())]
                self.is_match(known, emb, face_name, cand_name)