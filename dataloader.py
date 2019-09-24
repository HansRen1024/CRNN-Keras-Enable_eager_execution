import json, cv2
import numpy as np

class sequence_order_num:
    def __init__(self, total, batchsize=64):
        self.total = total
        self.range = [i for i in range(total)]
        self.index = 0
        max_index = int(total / batchsize)
        self.index_list = [i for i in range(max_index)]
        np.random.shuffle(self.index_list)

    def get(self, batchsize):
        s_o = []
        if self.index + batchsize > self.total:
            s_o_1 = self.range[self.index:self.total]
            self.index = (self.index + batchsize) - self.total
            s_o_2 = self.range[0:self.index]
            s_o.extend(s_o_1)
            s_o.extend(s_o_2)
        else:
            s_o = self.range[self.index:self.index + batchsize]
            self.index = self.index + batchsize
        return s_o

    def shuffle_batch(self, batchsize):
        if self.index== len(self.index_list): self.index=0
        start_index = self.index_list[self.index]*batchsize
        end_index = start_index + batchsize
        s_o = self.range[start_index:end_index]
        self.index += 1
        return s_o

def read_from_json(images, batchsize):
    idlist = sequence_order_num(len(images))
    while True:
        imgs = []
        labels = []
        inputLengthes = []
        shufimagefile = [images[ind] for ind in idlist.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            temp = json.loads(j.strip('\r\n'))
            IdNumber = temp['label']
            img = temp['img'].encode('utf-8')
            inputL = temp['input_length']
            BluredImg = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
            if len(BluredImg.shape) < 3 or BluredImg.shape[2] == 1:
                BluredImg = cv2.merge([BluredImg, BluredImg, BluredImg])
            img1 = (np.array(BluredImg, 'f')-127.5)/127.5
            imgs.append(img1)
            inputLengthes.append(inputL)
            labels.append(IdNumber)
        inputLengthes = np.array(inputLengthes).astype(np.int32)
        imgs = np.array(imgs)
        imgs = imgs.reshape((-1, BluredImg.shape[1], BluredImg.shape[0], BluredImg.shape[2]))
        yield imgs, labels, inputLengthes
