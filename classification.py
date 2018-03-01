import numpy as np
import tensorflow as tf
import cv2
import sys

# Wanna take in an image
# prep it for TF
# use TF to classify it

# Step 1: Given just an image of a digit,
# erode, threshold, prep image
# Step 2: run it through the classifier

def threshold(img, blur_filter=False):
    if (blur_filter):
        blur = cv2.GaussianBlur(img,(5,5),0)
        return cv2.adaptiveThreshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    else:
        return cv2.adaptiveThreshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 

def erode(img):
    # note that this makes the dark area bigger
    # best done on a binary image - see threshold
    # Use a 5x5 erosion kernel
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def classify(img):
    return 0

def process_image(img):
    # do we need to resize
    #image = cv2.resize()

    img = img.astype('float32')
    img = np.multiply(img, 1.0/255.0)
    return img


def main(args):
    # set up tensorflow object
    sess = tf.Session()
    model = tf.train.import_meta_graph('C:\\Users\\damckinn\\Documents\\Tensorflow\\MNIST\\model.ckpt-20000.meta')
    model.restore(sess, tf.train.latest_checkpoint('C:\\Users\\damckinn\\Documents\\Tensorflow\\MNIST'))

    model.predict()

    graph = tf.get_default_graph()

    #y_pred = graph.get_tensor_by_name("softmax_tensor")

    #x = graph.get_tensor_by_name("x:0")
    #y_true = graph.get_tensor_by_name("y_true:0")
    #y_test_images = np.zeroes((1,2))


    # image processing
    image = cv2.imread("test.jpg")
    img = process_image(erode(threshold(image)))


    # stuff
    # the eff does this do?
    feed_dict_testing = {x: img, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    print(str(result))

    # take in image name
    


if __name__ == "__main__":
    main(sys.argv)