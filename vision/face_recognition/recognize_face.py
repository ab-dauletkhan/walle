"""
Run to recognize people that were scanned previously. When a person with name "X" is recognized, the program will give an
auidio output saying "Hello X".

If you do not have an Edge TPU or you want to see the performance difference, change the
variable ifEdgeTPU_1_else_0 in main() to 0.
"""

import io
import re
import os
import time
from tflite_runtime.interpreter import load_delegate

from annotation import Annotator

import numpy as np

from PIL import Image
from PIL import ImageDraw
import cv2




from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960

def main():

  ifEdgeTPU_1_else_0 = 1
  
  labels = load_labels('coco_labels.txt')
  people_lables = load_labels('people_labels.txt')
  
  preloaded_embeddings = preload_embeddings('scanned_people/')
  
  #get interpreter for face detection model
  if ifEdgeTPU_1_else_0 == 1:
      interpreter = Interpreter(model_path = 'models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite',
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
  else:
      interpreter = Interpreter(model_path = 'models/ssd_mobilenet_v2_face_quant_postprocess.tflite')
  
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  
  #get interpreter for face embedding model
  if ifEdgeTPU_1_else_0 == 1:
      interpreter_emb = Interpreter(model_path = 'models/Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite',
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
  else:
      interpreter_emb = Interpreter(model_path = 'models/Mobilenet1_triplet1589223569_triplet_quant.tflite')

  interpreter_emb.allocate_tensors()

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
  cap.set(cv2.CAP_PROP_FPS, 30)

  try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Rotate 270 degrees
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert BGR (OpenCV) to RGB (PIL), then resize for inference
        image_large = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image_large.convert('RGB').resize(
            (input_width, input_height), Image.LANCZOS)

        start_time = time.monotonic()
        results = detect_objects(interpreter, image, 0.5)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Draw annotations
        annotate_objects(frame, results, labels, CAMERA_WIDTH, CAMERA_HEIGHT)
        cv2.putText(frame, '%.1fms' % elapsed_ms, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Face Recognition', frame)

        ymin, xmin, ymax, xmax, score = get_best_box_param(results, CAMERA_WIDTH, CAMERA_HEIGHT)
        
        frame_height, frame_width = frame.shape[:2]

        print(f"frame_width: {frame_width}, frame_height: {frame_height}")
        if score > 0.80:
            print(
                f"x_left_face: {xmin}, x_right_face: {xmax}, "
                f"y_left_face: {ymin}, y_right_face: {ymax}"
            )
        else:
            print("face: NOT DETECTED")

        if score > 0.80:
            img = np.array(image_large)
            img_cut = img[ymin:ymax, xmin:xmax, :]
            if img_cut is None or img_cut.size == 0 or img_cut.shape[0] == 0 or img_cut.shape[1] == 0:
                continue  # skip this frame, face crop is invalid
            img_cut = cv2.resize(img_cut, dsize=(96, 96),
                                 interpolation=cv2.INTER_CUBIC).astype('uint8')
            img_cut = img_cut.reshape(1, 96, 96, 3) / 255.
            emb = img_to_emb(interpreter_emb, img_cut)
            get_person_from_embedding(people_lables, emb, preloaded_embeddings)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

  finally:
    cap.release()
    cv2.destroyAllWindows()
    
def preload_embeddings(path, num_emb_check=20):
    """Load all embeddings from disk once at startup."""
    folders = sorted(os.listdir(path))
    embeddings = {}
    for folder in folders:
        emb_path = os.path.join(path, folder, 'embeddings')
        files = sorted(os.listdir(emb_path))[:num_emb_check]
        embeddings[folder] = [np.load(os.path.join(emb_path, f)) for f in files]
        print(f"Loaded {len(embeddings[folder])} embeddings for {folder}")
    return embeddings

def get_person_from_embedding(people_lables, emb, preloaded_embeddings):
    folders = sorted(preloaded_embeddings.keys())
    averages = np.zeros(len(folders))
    
    for i, folder in enumerate(folders):
        embs = preloaded_embeddings[folder]
        # vectorized — no Python loop over files
        diffs = np.array([np.sum((emb - e) ** 2) for e in embs])
        averages[i] = diffs.mean()

    who_is_on_pic = 0
    lowest_norm_found = 999
    for run, average in enumerate(averages):
        if average < 0.9 and average < lowest_norm_found:
            lowest_norm_found = average
            who_is_on_pic = run + 1
        print(average)
    print("person on pic: ", people_lables[who_is_on_pic])



def load_labels(path):
  #Loads the labels file. Supports files with or without index numbers.
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  #Sets the input tensor.
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  #Returns the output tensor at the given index.
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def set_input_tensor_emb(interpreter, input):
    #Sets input sensor for face embedding model
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    scale, zero_point = input_details['quantization']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.uint8(input/scale + zero_point)



def img_to_emb(interpreter,input):
    #returns embedding vector, using the face embedding model
    set_input_tensor_emb(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    #emb = np.squeeze(interpreter.get_tensor(output_details['index']))
    emb = interpreter.get_tensor(output_details['index'])
    scale, zero_point = output_details['quantization']
    emb = scale * (emb - zero_point)
    return emb

def detect_objects(interpreter, image, threshold):
  #Returns a list of detection results, each a dictionary of object info.
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def get_best_box_param(results,CAMERA_WIDTH, CAMERA_HEIGHT):
    #Returns the box parameters for the box with the highest score
    best_boxvalue = 0
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    for obj in results:
        if obj['score'] > best_boxvalue:
            best_boxvalue = obj['score']
            ymin, xmin, ymax, xmax = obj['bounding_box']
            if xmin < 0:
                xmin = 0
            if xmax > 1:
                xmax = 1
            if ymin < 0:
                ymin = 0
            if ymax > 1:
                ymax = 1
            xmin = int(xmin * CAMERA_WIDTH)
            xmax = int(xmax * CAMERA_WIDTH)
            ymin = int(ymin * CAMERA_HEIGHT)
            ymax = int(ymax * CAMERA_HEIGHT)
    #print("score: ", best_boxvalue)
    return ymin, xmin, ymax, xmax, best_boxvalue

def annotate_objects(frame, results, labels, CAMERA_WIDTH, CAMERA_HEIGHT):
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, '%s %.2f' % (labels[obj['class_id']], obj['score']),
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    



if __name__ == '__main__':
  main()
