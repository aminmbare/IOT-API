import tensorflow as tf
import os 
import numpy as np
from flask import Flask , request ,jsonify
import pickle
import traceback
import logging
NUM_CLASSES = 1

PATH_TO_CKPT = os.path.join(os.getcwd(),"models","leaf_detection","frozen_model","frozen_inference_graph.pb")
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
def select_boxes(boxes, classes, scores, score_threshold=0.5):
    """

    :param boxes:
    :param classes:
    :param scores:
    :param target_class: default traffic light id in COCO dataset is 10
    :return:
    """

    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    sel_id =  sq_scores > score_threshold

    return sq_boxes[sel_id]
def load_graph():
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        #od_graph_def = tf.saved_model.load()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

class LeafDetector(object):
      def __init__(self):

        self.detection_graph = load_graph()
        self.extract_graph_components()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph) 

        # run the first session to "warm up"
        dummy_image = np.zeros((100, 100, 3))
        self.detect_multi_object(dummy_image,0.1)
        self.traffic_light_box = None
        self.classified_index = 0

      def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
      def detect_multi_object(self, image_np, score_threshold):
        """
        Return detection boxes in a image

        :param image_np:
        :param score_threshold:
        :return:
        """

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        print(np.shape(image_np_expanded))
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        sel_boxes = select_boxes(boxes=boxes, classes=classes, scores=scores,
                                 score_threshold=score_threshold)
        return sel_boxes

def crop_roi_image(image_np, sel_box):
    im_height, im_width, _ = image_np.shape
    (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                  sel_box[0] * im_height, sel_box[2] * im_height)
    cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
    return cropped_image

def predict_1(image_np): 
    LD=LeafDetector()
    boxes=LD.detect_multi_object(image_np,score_threshold=0.5)
    cropped_image = crop_roi_image(image_np ,boxes[0])
    return cropped_image
def preproccess_1(img):

    image_np = np.asarray(img)
    return image_np

app = Flask(__name__)
@app.route('/leaf_detection', methods = ["GET", "POST"])
def leaf_detection():
    if request.method == "POST": 
        file = request.files['image']
        if file is  None or file.filename == "": 
            return jsonify({"error":"no file"})
        
        try: 
            img = pickle.loads(file.read()) 
            
            image = preproccess_1(img)

            result = predict_1(image)
            return pickle.dumps(result)
        except Exception as e:
            logging.error(traceback.format_exc())
            return jsonify({{'error':str(e)}})
    return 'OK'

if __name__ == "__main__": 
    app.run(host='127.0.0.1', port=5000 )
