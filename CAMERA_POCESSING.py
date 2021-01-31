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
from Predict_tread import predict_worker
class Camera_recognition():
    def __init__(self):
        print("Initialize class" + str(self))
        self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.detector = MTCNN()
        self.data_path = path_faces = os.path.join(os.getcwd(), 'faces')
        self.faces = self.load_data()
        print("Initialize DONE")

    def load_data(self):
        ls = []
        for file in os.listdir(self.data_path):
            buf = {}
            buf['name'] = file
            buf['data'] = self.extract_face(cv2.imread(os.path.join(self.data_path, file)))
            ls.append(buf)
        print(ls)
        return ls

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

    def proccessing_camera(self):
        cap = cv2.VideoCapture(0)
        while (True):
            ret, frame = cap.read()
            for item in self.faces:
                try:
                    ls = [item['data'], self.extract_face(frame)]
                    embeddings = self.get_embeddings(ls)
                    if self.is_match(embeddings[0], embeddings[1], item['name']):
                        cv2.imshow('frame', item['data'])
                except:pass
            cv2.imshow('frame', frame)
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


class Camera_treads():
    def __init__(self):
        print("Initialize class" + str(self))
        self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.detector = MTCNN()
        self.data_path = path_faces = os.path.join(os.getcwd(), 'faces')
        self.faces = self.load_data()
        print("Initialize DONE")

    def load_data(self):
        ls = []
        for file in os.listdir(self.data_path):
            buf = {}
            buf['name'] = file
            buf['data'] = self.extract_face(cv2.imread(os.path.join(self.data_path, file)))
            ls.append(buf)
        return ls

    def extract_face(self, pixels):
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        resized_image = cv2.resize(face, (224, 224))
        return resized_image

    def proccessing_camera(self):
        cap = cv2.VideoCapture(0)
        while (True):
            ret, frame = cap.read()
            threads = []
            for item in self.faces:
                worker = predict_worker(self.model,item,frame)
                worker.setDaemon(True)
                worker.start()
                threads.append(worker)
            for th in threads:
                th.join()
            cv2.imshow('frame', frame)
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    # CLC = Camera_recognition()
    # CLC.proccessing_camera()
    #
    CMS=Camera_treads()
    CMS.proccessing_camera()

if __name__ == '__main__':
    main()
#
# # define filenames
# filenames = ['bina.jpg', 'bina_3.jpg',
#              'yar.jpg', 'bina_2.jpg']
# embeddings = get_embeddings(filenames)
# # define sharon stone
# buna_id = embeddings[0]
# # verify known photos of sharon
# print('Positive Tests')
# is_match(embeddings[0], embeddings[1])
# is_match(embeddings[0], embeddings[3])
# # verify known photos of other people
# print('Negative Tests')
# is_match(embeddings[0], embeddings[2])
