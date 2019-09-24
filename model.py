import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Activation, LSTM, Bidirectional

import os, editdistance, time, datetime
import numpy as np

from dataloader import read_from_json

class Constants:
    rootPath = "/home/hans/WorkSpace/Data/_@Models/CRNN/"
    mode = 0

    if mode ==0:
        TRAIN_TFRECORD = rootPath+"/SynthText90K/preprocessed_SynthText90K_train_20190911.json"
        VAL_TFRECORDS = rootPath+"/SynthText90K/preprocessed_SynthText90K_test_20190911.json"
        savedPath = "/SynthText90K/"
    elif mode==1:
        TRAIN_TFRECORD = rootPath + "/ICDAR2015-Client1/ICDAR2015_train_20190829.json"
        VAL_TFRECORDS = rootPath + "/ICDAR2015-Client1/ICDAR2015_test_20190829.json"
        savedPath = "/Client1-local/"
    elif mode==2:
        TRAIN_TFRECORD = rootPath + "/IIIT.5K-Client2/IIIT5K_train_20190829.json"
        VAL_TFRECORDS = rootPath + "/IIIT.5K-Client2/IIIT5K_test_20190829.json"
        savedPath = "/Client2-cspc1/"
    else:
        TRAIN_TFRECORD = rootPath + "/SCUT-Client3/SCUT_Eng_word_train_20190829_cropped.json"
        VAL_TFRECORDS = rootPath + "/SCUT-Client3/SCUT_Eng_word_test_20190829_cropped.json"
        savedPath = "/Client3-cspc2/"

    CHARLIST_FILE = rootPath + "/character.txt"
    modelID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ACCURACY_FILE = rootPath + savedPath + modelID + "/accuracy.txt"
    MODEL_DIR = rootPath + savedPath + modelID + "/MODEL/"
    TENSORBOARD_DIR = rootPath+savedPath + modelID + "/TF_BOARD/"
    AUTOTUNE = tf.contrib.data.AUTOTUNE
    BATCH_SIZE = 448
    MAX_TEXT_LENGTH = 32
    REQUIRED_HEIGHT = 32
    REQUIRED_WIDTH = 200
    CHANNEL = 3

class crnn(tf.keras.Model):
    def __init__(self, len_charList):
        super(crnn, self).__init__()
        self.len_charList = len_charList
        # CNN Layer 1
        self.conv1 = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')
        # CNN Layer 2
        self.conv2 = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')
        # CNN Layer 3
        self.conv3 = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False)
        # CNN Layer 4
        self.conv4 = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False)
        self.pool4 = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid')
        # CNN Layer 5
        self.conv5 = Convolution2D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.norm5 = BatchNormalization(trainable=True, scale=True)
        self.relu5 = Activation(activation='relu')
        # CNN Layer 6
        self.conv6 = Convolution2D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.norm6 = BatchNormalization(trainable=True, scale=True)
        self.relu6 = Activation(activation='relu')
        self.pool6 = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid')
        # CNN Layer 7
        self.conv7 = Convolution2D(filters=512, kernel_size=2, strides=(1,2), padding='same', activation='relu', use_bias=False)
        # RNN Layer 1
        self.rnn1 = Bidirectional(layer=LSTM(units=256, unit_forget_bias=True, dropout=0.5, return_sequences=True), merge_mode='concat')
        # RNN Layer 2
        self.rnn2 = Bidirectional(layer=LSTM(units=256, unit_forget_bias=True, dropout=0.5, return_sequences=True), merge_mode='concat')
        # Atrous Layer
        self.atrous_conv = Convolution2D(filters=self.len_charList+1, kernel_size=3, dilation_rate=(1,1), padding='same', activation='softmax')
        
    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        
        conv1 = self.conv1(inputs)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)

        conv4 = self.conv4(conv3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)
        norm5 = self.norm5(conv5)
        relu5 = self.relu5(norm5)

        conv6 = self.conv6(relu5)
        norm6 = self.norm6(conv6)
        relu6 = self.relu6(norm6)
        pool6 = self.pool6(relu6)

        conv7 = self.conv7(pool6)
        x = tf.squeeze(conv7, axis=[2])

        rnn1 = self.rnn1(x)
        rnn2 = self.rnn2(rnn1)
        x = tf.expand_dims(rnn2, axis=2)

        atrous_conv = self.atrous_conv(x)
        x = tf.squeeze(atrous_conv, axis=[2])

        return x

def fast_ctc_decode(char_num,ind):
    # 复现CTC_DECODE
    ResultList = []  # 格式： [数字，无用的对应概率]
    for row in range(0, char_num[ind, :, :].shape[0]):
        MaxIndex = char_num[ind, row, :].tolist().index(max(char_num[ind, row, :].tolist()))
        if row == char_num[ind, :, :].shape[0] - 1:
            if MaxIndex != (char_num[ind, :, :].shape[1]-1):
                ResultList.append([MaxIndex, max(char_num[ind, row, :].tolist())])
            continue
        NeMaxIndex = char_num[ind, row + 1, :].tolist().index(max(char_num[ind, row + 1, :].tolist()))
        if NeMaxIndex != MaxIndex and MaxIndex != (char_num[ind, :, :].shape[1]-1):
            ResultList.append([MaxIndex, max(char_num[ind, row, :].tolist())])
        if MaxIndex != (char_num[ind, :, :].shape[1]-1) and NeMaxIndex == MaxIndex:
            continue
    return ResultList

def ctc_cost(y_pred, y_true, input_length):
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-7)
    loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(labels=y_true, inputs=y_pred, sequence_length=input_length))
    return loss

def train_on_batch(model, charList, optimizer, epoch, images):
    try:
        print("Training")
        processBar = OutputBar()
        batch_count = int(np.floor(len(images) / Constants.BATCH_SIZE))
        generator = read_from_json(images, Constants.BATCH_SIZE)
        for i in range(batch_count):
            startTime = time.time()
            with tf.device("CPU:0"):
                imgs, labels, inputLengthes = next(generator)
            labels = toSparse(labels, charList)

            with tf.GradientTape() as tape:
                y_pred = model(imgs)
                loss = ctc_cost(y_pred, labels, inputLengthes)
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            tf.contrib.summary.scalar("loss", loss, step=(epoch - 1) * batch_count + i + 1)
            print(processBar(i + 1, batch_count, time.time() - startTime, loss.numpy()), end='')
        return loss.numpy()

    except:
        raise

def validate_on_batch(model, charList, epoch, images):
    try:
        numCharErr = 0
        numCharTotal = 0
        numWordOK = 0
        numWordTotal = 0
        print("\nValidation")
        processBar = OutputBar()
        batch_count = int(np.floor(len(images) / Constants.BATCH_SIZE))
        generator = read_from_json(images, Constants.BATCH_SIZE)
        for i in range(batch_count):
            startTime = time.time()
            with tf.device("CPU:0"):
                imgs, labels, inputLengthes = next(generator)

            model_op = model(imgs)
            model_op_t = tf.transpose(model_op, [1, 0, 2])
            decoder = tf.nn.ctc_greedy_decoder(inputs=model_op_t, sequence_length=inputLengthes)
            recognized = decoderOutputToText(decoder, charList)

            print(processBar(i+1, batch_count, time.time()-startTime), end='')
            # print('Ground truth -> Recognized')
            for j in range(Constants.BATCH_SIZE):
                numWordOK += 1 if labels[j] == recognized[j] else 0
                numWordTotal += 1
                dist = editdistance.eval(recognized[j], labels[j])
                numCharErr += dist
                numCharTotal += len(labels[j])
                # print(bcolors.OKGREEN+'[OK]' if dist==0 else bcolors.FAIL+'[ERR:%d]' % dist,'"' + labels[j] + '"', '->', '"' + recognized[j] + '"'+bcolors.ENDC)

        # print validation result
        charErrorRate = ((numCharTotal-numCharErr) / numCharTotal) * 100
        wordAccuracy = (numWordOK / numWordTotal) * 100
        if wordAccuracy > 85:
            save_model(model, epoch)
        tf.contrib.summary.scalar("character_error_rate", charErrorRate, step=epoch)
        tf.contrib.summary.scalar("word_accuracy", wordAccuracy, step=epoch)
        print('\nEPOCH '+str(epoch)+': Character accuracy rate: %f%%. Word accuracy: %f%%.' % (charErrorRate, wordAccuracy))
        print('----------')
        with open(Constants.ACCURACY_FILE, "a") as f:
            f.write('EPOCH '+str(epoch)+': Character accuracy rate: %f%%. Word accuracy: %f%%.' % (charErrorRate, wordAccuracy)+'\n')
    except:
        raise

def save_model(model, epoch):
    checkpoint_dir = Constants.MODEL_DIR+str(epoch)
    os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    saver = tfe.Saver(model.variables)
    saver.save(checkpoint_prefix)

def toSparse(texts, charList):
    indices = []
    values = []
    shape = [len(texts), 0] # last entry must be max(labelList[i])

    # go over all texts
    for (batchElement, text) in enumerate(texts):
        # text = text.decode("utf-8")
        # convert to string of label (i.e. class-ids)
        labelStr = [charList.index(c) for c in text]
        # sparse tensor must have size of max. label-string
        if len(labelStr) > shape[1]:
            shape[1] = len(labelStr)
        # put each label into sparse tensor
        for (i, label) in enumerate(labelStr):
            indices.append([batchElement, i])
            values.append(label)

    sparseText = tf.SparseTensor(indices, values, shape)
    return sparseText

def decoderOutputToText(ctcOutput, charList):
    # contains string of labels for each batch element
    encodedLabelStrs = [[] for _ in range(Constants.BATCH_SIZE)]

    decoded=ctcOutput[0][0]

    # go over all indices and save mapping: batch -> values
    for (idx, idx2d) in enumerate(decoded.indices):
        label = decoded.values[idx]
        batchElement = idx2d[0] # index according to [b,t]
        encodedLabelStrs[batchElement].append(label)

    # map labels to chars for all batch elements
    decodedText = [str().join([charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]
    return     decodedText

class OutputBar(object):
    def __init__(self, number=50, decimal=2):
        self.decimal = decimal
        self.number = number
        self.a = 100 / number
        self.totalLoss = 0
        self.totalTime = 0

    def __call__(self, now, total, cost, loss=0):
        percentage = round(now / total * 100, self.decimal)

        well_num = int(percentage / self.a)
        # print("well_num: ", well_num, percentage)

        progress_bar_num = self.progress_bar(well_num)

        self.totalTime+=cost
        if not loss==0:
            self.totalLoss+=loss
            result = "\r%s %s %d/%d Batches; Loss: %s; Time: %s" % (str(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())),
                                                                    progress_bar_num,
                                                                    now,
                                                                    total,
                                                                    str(round(self.totalLoss/(now+1),4)),
                                                                    str(round(self.totalTime/(now+1),4)))
        else:
            result = "\r%s %s %d/%d Batches; ; Time: %s" % (str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                                                            progress_bar_num,
                                                            now,
                                                            total,
                                                            str(round(self.totalTime/(now+1),4)))
        return result

    def progress_bar(self, num):
        well_num = ">" * num
        space_num = " " * (self.number - num)

        return '[%s%s]' % (well_num, space_num)
