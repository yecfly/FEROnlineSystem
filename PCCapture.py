import cv2
import numpy as np
from kafka import KafkaProducer
from kafka.version import __version__

SHOWIMG=False

if __name__=="__main__":
    #producer = KafkaProducer(bootstrap_servers="127.0.0.1:9092", api_version=__version__)
    producer = KafkaProducer(bootstrap_servers="xxx.xxx.xxx.xxx:9092", api_version=__version__)#added by yys 2019.11.21
    if SHOWIMG:
        cv2.namedWindow('video')
        capture = cv2.VideoCapture(0)
        _, frame = capture.read()
        
        count = 0
        while frame is not None:
            count += 1
            if count % 10 == 0:
                producer.send('video', cv2.imencode('.jpeg', frame)[1].tostring())
                print('send {0} frames, shape {1}'.format(count//10, frame.shape))
                cv2.imshow('frame', frame)
                cv2.waitKey(10)
            _, frame = capture.read()
    else:
        capture = cv2.VideoCapture(0)
        _, frame = capture.read()
        #size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #vw = cv2.VideoWriter(vfile,cv2.VideoWriter_fourcc('M','J','P','G'), 30, size, 1)

        count = 0
        while frame is not None:
            """
            key = cv2.waitKey(10)
            if key == ord('s'):     # 当按下"s"键时，将保存当前画面
                cv2.imwrite('screenshot.jpg', frame)
            elif key == ord('q'):   # 当按下"q"键时，将退出循环
                break
            """
            count += 1
            if count % 10 == 0:
                print('send {0} frames, shape {1}'.format(count//10, frame.shape))
                producer.send('video', cv2.imencode('.jpeg', frame)[1].tostring())
            _, frame = capture.read()