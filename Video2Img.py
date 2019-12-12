# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-08 09:20:58
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-25 09:56:37


#######################################################################
# 1. fetch original frame from kafka 
# 2. detect whether there are human faces in the frame
# 3. if detected, predict the emotions of  human faces, label them on the image
# 4. send the processed image to kafka   
# 5. send the emotion distribution to kafka

# note: in the new version, all the producer and consumer must specify the api_version or it will raise NoBrokersAvailable error.
#######################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # decide to use CPU or GPU. If there is graphic card with CUDA compatibility and string number is within the GPU count, the GPU is used. Otherwise, CPU is used.
import time
import json
from datetime import datetime
from multiprocessing import Pool
from kafka.version import __version__

import cv2
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

from FaceProcessUtilMultiFaces import preprocessImage

from Recorder import FileRecorder, RedisRecorder

#FERM='vgg'# 'vgg' represents the VGG model; 'alex' represents the Alexnet model
FERM='rcfn'

Consumer_SERVER = 'xxx.xxx.xxx.xxx:9092'
Producer_SERVER = 'xxx.xxx.xxx.xxx:9092'
VIDEO_TOPIC = 'video'
IMAGE_TOPIC = 'image'
PROBABILITY_TOPIC = 'emotion_probability'
EMOTION = ('neutral', 'angry', 'surprise', 'disgust', 'fear', 'happy', 'sad')
COLOR_RGB = ((0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0))
COLOR_HEX = ('#00FF00', '#0000FF', '#FF0000', '#FFFF00', '#FF00FF', '#00FFFF')
FONT = cv2.FONT_HERSHEY_SIMPLEX

class VideoConsumer():
    def __init__(self):
        self.consumer = KafkaConsumer(bootstrap_servers = [Consumer_SERVER], api_version=__version__, auto_offset_reset='latest') # earliest

    def get_img(self):
        self.consumer.subscribe([VIDEO_TOPIC])
        for message in self.consumer:
            if message.value != None:
                yield message.value
            """
            # convert bytes to image in memory
            nparr = np.fromstring(message.value, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
            """


class ImageProducer():
    '''ImageProducer in Sever backend for sending back the detect face to client'''
    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers = [Producer_SERVER], api_version=__version__)

    def send_img(self, img):
        self.producer.send(IMAGE_TOPIC, value = img)

class ProbabilityProducer():
    '''ProbabilityProducer in Sever backend for sending back the predict pro to client'''
    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers = [Producer_SERVER], api_version=__version__)

    def send_probability_distribution(self, msg):
        self.producer.send(PROBABILITY_TOPIC, value = msg)


def predict(face_img):
    if FERM=='vgg':
        from FaceProcessUtilMultiFaces import preprocessImage
        from FERMODEL import VGGModel
        global_model = VGGModel()
    elif FERM=='alex':
        from FaceProcessUtilMultiFaces import preprocessImage
        from FERMODEL import AlexNet
        global_model = AlexNet()###do not have the corresponding pre-saved model
    elif FERM=='rcfn':
        from FaceProcessUtilMultiFacesV2 import preprocessImage
        from FERMODEL import RCFN
    return global_model.predict(face_img) # (emotion, probability_distribution)


def predict_and_label_frame(video_consumer , img_producer, probability_producer, recorder, pool, maximum_detect_face = 6):
    """fetch original frame from kafka with video_consumer
       detect whether there are human faces in the frame
       predict the emotions of  human faces, label them on the image
       send the processed image to kafka with img_producer 
       send the emotion distribution to kafka with probability_producer
    """
    consume_count = 0
    produce_count = 0
    if FERM=='vgg' or FERM=='alex':
        while True:
            for img in video_consumer.get_img():
                start_time = time.time()
                consume_count += 1
                
                print('========Consume {0} from video stream'.format(consume_count))
                # write original image to disk
                """
                raw_dest_img = './rev_img/original{0}.png'.format(consume_count)
                with open(raw_dest_img, 'wb') as f:
                    f.write(img)
                """
                # transform image from bytes to ndarray
                np_arr = np.fromstring(img, dtype = np.uint8) # one dimension array
                np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                result = preprocessImage(np_img)
                print('**********time consumed by face detection: {0}s'.format(time.time() - start_time))
            
                start_time = time.time()
                if result['detected']: # detect human face
                    produce_count += 1
                    # deal with multiple face in an image
                    num_faces = min(maximum_detect_face, len(result['rescaleimg']))
                    face_imgs, face_points = result['rescaleimg'], result['originalPoints']
                    emotion_distributions = {}
                    # use multiple processes to predict
                    # predicted_results = pool.map(predict, face_imgs)
                    predicted_results = [pool.apply(predict, args = (face_imgs[i], )) for i in range(num_faces)]
                    for i in range(num_faces):
                        emotion, probability_distribution = predicted_results[i]
                        distribution = dict(zip(EMOTION, probability_distribution.tolist()[0]))
                        emotion_distributions[COLOR_HEX[i]] = distribution
                        print('*****************probability_distribution: {0}'.format(probability_distribution))
                    
                        # write the record to redis     
                        recorder.write_record(face_imgs[i].tostring(), emotion)

                        # add square and text to the human face in the image
                        left_top, right_bottom = face_points[i]
                        cv2.rectangle(np_img, left_top, right_bottom, COLOR_RGB[i], 2)
                        text_left_bottom = (left_top[0], left_top[1] - 20)
                        cv2.putText(np_img, emotion, text_left_bottom, FONT, 1, COLOR_RGB[i], 2)

                    print('**********time consumed by predicting, storing and texting image: {0}s'.format(time.time() - start_time))
                    
                    # cv2.imwrite('./test_imgs/img_{0}.jpg'.format(datetime.now().strftime("%Y%m%d%H%M%S")), np_img)              
                
                    start_time = time.time()
                    # send image to kafka
                    img_producer.send_img(cv2.imencode('.jpeg', np_img)[1].tostring())
                    print('#########produce {0} to image stream'.format(produce_count))
                    # send emotion probability distribution to kafka
                    probability_producer.send_probability_distribution(json.dumps(emotion_distributions).encode('utf8'))
                    print('#########produce {0} to probability stream'.format(emotion_distributions))
                    print('**********time consumed by sending image and distribution: {0}s'.format(time.time() - start_time))

                else:
                    # message = {'img': img, 'distribution': None}
                    img_producer.send_img(img)
                    print('#########produce raw image to image stream')
                    empty_distribution = {COLOR_HEX[0] : dict(zip(EMOTION, [0] * 7))}
                    probability_producer.send_probability_distribution(json.dumps(empty_distribution).encode('utf8'))
    elif FERM=='rcfn':
        while True:
            for img in video_consumer.get_img():
                start_time = time.time()
                consume_count += 1
                #print('========Consume {0} from video stream'.format(consume_count))
                
                # transform image from bytes to ndarray
                np_arr = np.fromstring(img, dtype = np.uint8) # one dimension array
                np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                result = preprocessImage(np_img)
                print('**********time consumed by face detection: {0}s'.format(time.time() - start_time))
            
                start_time = time.time()
                if result['detected']: # detect human face
                    produce_count += 1
                    # deal with multiple face in an image
                    num_faces = min(maximum_detect_face, len(result['rescaleimg']))
                    face_imgs, face_points = result['rescaleimg'], result['originalPoints']
                    emotion_distributions = {}
                    # use multiple processes to predict
                    # predicted_results = pool.map(predict, face_imgs)
                    predicted_results = [pool.apply(predict, args = (face_imgs[i], )) for i in range(num_faces)]
                    for i in range(num_faces):
                        emotion, probability_distribution = predicted_results[i]
                        distribution = dict(zip(EMOTION, probability_distribution.tolist()[0]))
                        emotion_distributions[COLOR_HEX[i]] = distribution
                        print('*****************probability_distribution: {0}'.format(probability_distribution))
                    
                        # write the record to redis     
                        recorder.write_record(face_imgs[i].tostring(), emotion)

                        # add square and text to the human face in the image
                        left_top, right_bottom = face_points[i]
                        cv2.rectangle(np_img, left_top, right_bottom, COLOR_RGB[i], 2)
                        text_left_bottom = (left_top[0], left_top[1] - 20)
                        cv2.putText(np_img, emotion, text_left_bottom, FONT, 1, COLOR_RGB[i], 2)

                    print('**********time consumed by predicting, storing and texting image: {0}s'.format(time.time() - start_time))
                    
                    # cv2.imwrite('./test_imgs/img_{0}.jpg'.format(datetime.now().strftime("%Y%m%d%H%M%S")), np_img)              
                
                    start_time = time.time()
                    # send image to kafka
                    img_producer.send_img(cv2.imencode('.jpeg', np_img)[1].tostring())
                    print('#########produce {0} to image stream'.format(produce_count))
                    # send emotion probability distribution to kafka
                    probability_producer.send_probability_distribution(json.dumps(emotion_distributions).encode('utf8'))
                    print('#########produce {0} to probability stream'.format(emotion_distributions))
                    print('**********time consumed by sending image and distribution: {0}s'.format(time.time() - start_time))


if __name__ == '__main__':
    video_consumer = VideoConsumer()
    img_producer = ImageProducer()
    probability_producer = ProbabilityProducer()
    recorder = RedisRecorder()
    pool = Pool(1)
    # record_dir = './detected_records/'
    # file_recorder = FileRecorder(record_dir)
    # model = AlexNet()

    predict_and_label_frame(video_consumer = video_consumer, 
                            img_producer = img_producer, 
                            probability_producer = probability_producer, 
                            recorder = recorder,
                            pool = pool)