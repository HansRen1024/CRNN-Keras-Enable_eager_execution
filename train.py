import os, time, json
import tensorflow as tf
from CRNN.model import Constants, crnn, train_on_batch, validate_on_batch

tf.compat.v1.enable_eager_execution()

def main(argv=None):
    if not os.path.exists(Constants.MODEL_DIR):
        os.makedirs(Constants.MODEL_DIR)
    if not os.path.exists(Constants.TENSORBOARD_DIR):
        os.makedirs(Constants.TENSORBOARD_DIR)

    with open(Constants.CHARLIST_FILE, "r") as fp:
        charList = fp.readlines()
    for ind in range(len(charList)):
        charList[ind] = charList[ind].strip('\n')
    model = crnn(len(charList))

    starter_learning_rate = 0.1
    optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=starter_learning_rate)
    epoch = 1
    summary_writer = tf.contrib.summary.create_file_writer(Constants.TENSORBOARD_DIR)

    with open(Constants.TRAIN_TFRECORD, 'r', encoding='utf-8') as imgf:
        Train_images = imgf.readlines()
    for key, value in json.loads(Train_images[0].strip('\r\n')).items():
        print(key, ':', value)
    Train_images.pop(0)

    with open(Constants.VAL_TFRECORDS, 'r', encoding='utf-8') as imgf:
        Val_images = imgf.readlines()
    for key, value in json.loads(Val_images[0].strip('\r\n')).items():
        print(key, ':', value)
    Val_images.pop(0)

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        while True:
            print("Epoch", epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            loss = train_on_batch(model, charList, optimizer, epoch, Train_images)
            validate_on_batch(model, charList, epoch, Val_images)
            epoch += 1

if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    tf.compat.v1.app.run()
