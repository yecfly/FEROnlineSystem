# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-09-21 11:16:46
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-24 20:51:03

########################################################
# save the detected face and emotion on disk or database
########################################################

import os
from datetime import datetime

import redis


class FileRecorder():
    """write records in a file on disk
    Attributes:
        curr_id (int): id of the current record to write on disk 
        latest_file (str): path of the record file
        max_record (int): maximum records allowed to store in a file
        record_dir (str): directory containing the record files
    """
    def __init__(self, record_dir, max_record_pre_file = 1000):
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
            print('Successfully creating directory {0}'.format(record_dir))
        self.record_dir = record_dir
        self.max_record = max_record_pre_file
        self.curr_id = 0
        self.latest_file = self._get_latest_file()


    def _get_latest_file(self):
        files = sorted(os.listdir(self.record_dir))
        if len(files) > 0:
            curr_latest = self.record_dir + files[-1]
            with open(curr_latest, 'r') as rf:
                lines_count = sum(1 for line in rf)
            if lines_count < self.max_record:
                self.curr_id = lines_count
                return curr_latest
        new_csv = self.record_dir + '{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
        with open(new_csv, 'w') as wf: # add csv header
            wf.write('id,pixels,label\n')
        return new_csv


    def write_record(self, img, label):
        self.curr_id += 1
        params = {'id' : self.curr_id,  'pixels' : img,  'label' : label}
        with open(self.latest_file, 'a') as wf:
            wf.write('{id:05d},{pixels},{label}\n'.format(**params))
        # create new csv file when reaching max records
        if self.curr_id == self.max_record:
            new_csv = self.record_dir + '{0}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
            with open(new_csv, 'w') as wf: # add csv header
                wf.write('id,pixels,label\n')
            self.latest_file = new_csv
            self.curr_id = 0


class RedisRecorder():
    def __init__(self,
                 HOST = 'xxx.xxx.xxx.xxx',
                 PORT = 6379,
                 PASSWORD = 'xxxxxxxxxxxxxx',
                 DB = 0):
        try:
            self.conn = redis.Redis(host = HOST, port = PORT, password = PASSWORD, db= DB)
        except Exception:
            print('Exception while connecting to redis')
            exit()

        try:
            self.count = int(self.conn.get('count'))
        except Exception:
            print('Exception while getting the count variable, set it to 0\nExit')
            self.count = 0
            self.conn.set('count', 0)


    def write_record(self, img, emotion):
        # update count in the db
        self.count += 1
        self.conn.set('count', str(self.count))
        # store image and label in the db
        record_name = 'face{0:06d}'.format(self.count)
        self.conn.hset(name = record_name, key = 'str_img', value = img)
        self.conn.hset(name = record_name, key = 'emotion', value = emotion)



if __name__ == '__main__':
    """# test file recorder
    record_dir = './records_csv/'
    file_recorder = FileRecorder(record_dir)
    for i in range(52000):
        file_recorder.write_record(i, i*10)
    """
    # test redis recorder
    redis_recorder = RedisRecorder()
    for i in range(20):
        redis_recorder.write_record(str(i), str(i*20))
        print('%s %s Record has been written.'%(str(i), str(i*20)))