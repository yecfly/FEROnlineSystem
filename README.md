# FEROnlineSystem
A online system for FER.
Author WLC and YYS. WLC had done most of the system framework, including PCCapture.py, Recorder, Video2Img.py, Webserver，and Consumer.py.
YYS wrote the image preprocessing and FER part, including FERMODEL.py(The AlexNet part is from WLC. On 2019.12.12, the rcfn part is nearly finished.), FaceProcessUtilMultiFaces.py, FaceProcessUtilMultiFacesV2.py, and the rcfn part in Video2Img.py.
To be compatible with newly version of kafka in python, extra parameters have been added.

The instructions for the FEROnlineSystem are in "module required and start command.txt", but you still need the two dlib models, which can be downloaded from Dlib offical website, and a pre-trained FER model to run this system.
