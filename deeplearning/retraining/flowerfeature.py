# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf
import flowerreader as reader
import retrainmodel as model

INPUT_WIDTH = 299
INPUT_HEIGHT = 299
INPUT_DEPTH = 3
INPUT_TENSOR_NAME = 'ResizeBilinear:0'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
CATEGORIES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

def main():

    datapath = 'e:\\prmldata\\tensorflow\\flower_photos'
    inputmodelpath = 'e:\\prmldata\\tensorflow\\models\\inception-2015-12-05\\classify_image_graph_def.pb'

    graph, bottleneck_tensor, input_tensor = model.loadmodel(inputmodelpath, BOTTLENECK_TENSOR_NAME,
                                                             INPUT_TENSOR_NAME)

    imageset = reader.load_dataset(datapath, CATEGORIES, dtype=np.uint8,
                                   needresize=True, newsize=[INPUT_WIDTH, INPUT_HEIGHT],
                                   vectorize=False, grayscale=False)

    with tf.Session() as sess:
        #  One thing to note is that the Inception v3 graph has a bug which prevents running batches
        #  larger than 1 through it. You just need to change one shape constant, I believe it's
        # Node: pool_3/_reshape's first argument, to have a shape of -1 for the batch dimension.
        #  I don't have a code change for that yet though.
        #  https://groups.google.com/a/tensorflow.org/d/msg/discuss/ee8UBKRjdQM/JVk_rJPaEAAJ
        for imagepath in imageset.images:
            imagedata, _ = imageset.next_batch(1, shuffle=False)
            bottleneck = model.getbottlenecks(sess, imagedata, input_tensor, bottleneck_tensor)
            bottleneckpath = imagepath + '.bottleneck'

            print('Creating bottleneck at ' + bottleneckpath)
            bottleneck_string = ','.join(str(x) for x in bottleneck)
            with open(bottleneckpath, 'w') as fp:
                fp.write(bottleneck_string)


if __name__ == "__main__":
    main()