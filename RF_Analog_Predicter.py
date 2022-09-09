import os
import torch
from torchvision import models
import re
import cv2
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
import shutil
import math
import winsound
import requests
from PIL import Image
import io
from requests_toolbelt.multipart.encoder import MultipartEncoder
import yaml
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid


# User parameters
SAVE_NAME_OD = "./Models-OD/RF_Analog-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
SAVE_ANNOTATED_VIDEOS   = True
MIN_SCORE               = 0.7 # Minimum object detection score


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


# Creates class folder
def makeDir(dir, classes_2):
    for classIndex, className in enumerate(classes_2):
        os.makedirs(dir + className, exist_ok=True)



# Starting stopwatch to see how long process takes
start_time = time.time()

# Windows beep settings
frequency = 700  # Set Frequency To 2500 Hertz
duration = 150  # Set Duration To 1000 ms == 1 second

# Email info
settings        = yaml.safe_load( open("config.yaml") )
camera_ip_info  = settings['camera_ip_info']
from_addr       = settings['from_addr']
to_addr         = settings['to_addr']
password        = settings['password']

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)


dataset_path = DATASET_PATH

#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
num_classes = len(categories.keys())

classes = [i[1]['name'] for i in categories.items()]



# lets load the faster rcnn model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD, map_location=map_location)
    model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()
torch.cuda.empty_cache()

transforms = A.Compose([
    ToTensorV2()
])


# Start FPS timer
fps_start_time = time.time()

color_list =['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
pred_dict = {}
ii = 0
for video_name in os.listdir(TO_PREDICT_PATH):
    video_path = os.path.join(TO_PREDICT_PATH, video_name)
    
    
    video_capture = cv2.VideoCapture(video_path)
    
    success, image_b4_color = video_capture.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(PREDICTED_PATH + video_name, fourcc, 20.0, 
                                (int(image_b4_color.shape[1]), 
                                 int(image_b4_color.shape[0])
                                 )
                                )
    
    count = 1
    while success:
        success, image_b4_color = video_capture.read()
        if not success:
            break
        
        # Inference through Roboflow
        # -----------------------------------------------------------------------------
        # Load Image with PIL
        image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(image)
        
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        pilImage.save(buffered, quality=100, format="JPEG")
        
        # Construct the URL
        upload_url = "".join([
            "https://detect.roboflow.com/blood-cell-detection-1ekwu/1",
            "?api_key=umichXAeCyw6nlBsDZIt",
            "&confidence=" + str(MIN_SCORE)
        ])
        
        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        
        response = requests.post(upload_url, 
                                 data=m, 
                                 headers={'Content-Type': m.content_type},
                                 )
        
        predictions = response.json()['predictions']
        # -----------------------------------------------------------------------------
        
        die_coordinates = []
        labels_found = []
        confidence_level_list = []
        for prediction in predictions:
            x1 = prediction['x']
            y1 = prediction['y']
            x2 = x1 + prediction['width']
            y2 = y1 + prediction['height']
            die_coordinates.append([x1, y1, x2, y2])
            
            label = prediction['class']
            labels_found.append(label)
        
        
        if "Center" in labels_found and "Needle_Tip" in labels_found:
            
            # Center
            center_indexes = [index for index, x in enumerate(labels_found) if x == "Center"]
            center_coordinates = die_coordinates[center_indexes[0]]
            
            # Center of center bbox
            center_x_center = int(center_coordinates[0] 
                                  + (center_coordinates[2] - center_coordinates[0])/2
                                  )
            center_y_center = int(center_coordinates[1] 
                                  + (center_coordinates[3] - center_coordinates[1])/2
                                  )
            
            # Needle_Tip
            needle_tip_indexes = [index for index, x in enumerate(labels_found) if x == "Needle_Tip"]
            needle_tip_coordinates = die_coordinates[needle_tip_indexes[0]]
            
            # Center of needle tip bbox
            center_x_needle_tip = int(needle_tip_coordinates[0] 
                                      + (needle_tip_coordinates[2] - needle_tip_coordinates[0])/2
                                      )
            center_y_needle_tip = int(needle_tip_coordinates[1] 
                                      + (needle_tip_coordinates[3] - needle_tip_coordinates[1])/2
                                      )
            
            # Finds angle
            dy = center_y_needle_tip - center_y_center
            dx = center_x_needle_tip - center_x_center
            theta = math.atan2(dy, dx)
            theta = math.degrees(theta)
            theta = round(theta)
            
            # Changes negative thetat to appropriate value
            if theta < 0:
                theta *= -1
                theta = (180 - theta) + 180
            
            # Sets new starting point
            theta = theta - 90
            
            # Changes negative thetat to appropriate value
            if theta < 0:
                theta *= -1
                theta = theta + 270
            
            # theta of 74 is 500 psi and theta of 173 is 2,000 psi
            if theta <= 74 or theta >= 173:
                winsound.Beep(frequency, duration)
            
            # Puts cooridnates on center label
            for label_index, label in enumerate(labels_found):
                if "Center" in label:
                    labels_found[label_index] = label + " " +  str(theta) + " deg"
        
        if SAVE_ANNOTATED_VIDEOS:
            # predicted_image = draw_bounding_boxes(transformed_image,
            #     boxes = die_coordinates,
            #     # labels = [classes[i] for i in die_class_indexes], 
            #     # labels = [str(round(i,2)) for i in die_scores], # SHOWS SCORE IN LABEL
            #     width = line_width,
            #     colors = [color_list[i] for i in die_class_indexes],
            #     font = "arial.ttf",
            #     font_size = 10
            #     )
            
            # predicted_image_cv2 = image.permute(1,2,0).contiguous().numpy()
            # predicted_image_cv2 = cv2.cvtColor(predicted_image_cv2, cv2.COLOR_RGB2BGR)
            
            for dieCoordinate_index, dieCoordinate in enumerate(die_coordinates):
                start_point = ( int(dieCoordinate[0]), int(dieCoordinate[1]) )
                end_point = ( int(dieCoordinate[2]), int(dieCoordinate[3]) )
                color_1 = (255, 0, 255)
                color_2 = (255, 255, 255)
                thickness = 3
                
                cv2.rectangle(image_b4_color, start_point, end_point, color_1, thickness)
                
                # Draws line from needle base to tip
                cv2.line(image_b4_color, 
                         (center_x_center, center_y_center), 
                         (center_x_needle_tip, center_y_needle_tip), 
                         color_2, 
                         thickness=thickness
                         )
                
                start_point_text = (start_point[0], max(start_point[1]-5,0) )
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1.0
                thickness = 2
                cv2.putText(image_b4_color, 
                            labels_found[dieCoordinate_index], 
                            start_point_text, font, fontScale, color_2, thickness)
            
            # Saves video with bounding boxes
            video_out.write(image_b4_color)
        
        
        tenScale = 10
    
        ii += 1
        if ii % tenScale == 0:
            fps_end_time = time.time()
            fps_time_lapsed = fps_end_time - fps_start_time
            print("  " + str(ii) + " of " 
                  + str(len(os.listdir(TO_PREDICT_PATH))), 
                  "-",  round(tenScale/fps_time_lapsed, 2), "FPS")
            fps_start_time = time.time()
        
        count += 1
        
    video_out.release()


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)