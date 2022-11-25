import os
import cv2
import time
import shutil
import math
import winsound
import requests
from PIL import Image
import io
from requests_toolbelt.multipart.encoder import MultipartEncoder


# User parameters
TO_PREDICT_PATH         = "./To_Predict_Videos/"
PREDICTED_PATH          = "./Predicted_Videos/"
MIN_SCORE               = 0.4 # Minimum object detection score



# Main()
# =============================================================================
# Windows beep settings - Used for alarm if psi threshold met
frequency = 700  # Hertz
duration = 150  # Millisecond

# Loops through each video found in TO_PREDICT_PATH folder
for video_name in os.listdir(TO_PREDICT_PATH):
    video_path = os.path.join(TO_PREDICT_PATH, video_name)
    
    video_capture = cv2.VideoCapture(video_path)
    
    # Video frame count and fps needed for VideoWriter settings
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = round( video_capture.get(cv2.CAP_PROP_FPS) )
    
    # If successful and image of frame
    success, image_b4_color = video_capture.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(PREDICTED_PATH + video_name, fourcc, video_fps, 
                                (int(image_b4_color.shape[1]), 
                                 int(image_b4_color.shape[0])
                                 )
                                )
    
    # If still seeing video, loops through each frame in video
    while success:
        success, image_b4_color = video_capture.read()
        if not success:
            break
        
        # Inference through Roboflow section
        # -----------------------------------------------------------------------------
        # Load Image with PIL
        image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(image)
        
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        pilImage.save(buffered, quality=100, format="JPEG")
        
        # Construct the URL
        settings = yaml.safe_load( open("config.yaml") )
        upload_url = settings['upload_url'] + str(MIN_SCORE)
        
        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        
        response = requests.post(upload_url, 
                                 data=m, 
                                 headers={'Content-Type': m.content_type},
                                 )
        
        predictions = response.json()['predictions']
        # -----------------------------------------------------------------------------
        
        # Creates lists from inferenced frames
        object_coordinates = []
        labels_found = []
        confidence_level_list = []
        for prediction in predictions:
            x1 = prediction['x'] - prediction['width']/2
            y1 = prediction['y'] - prediction['height']/2
            x2 = x1 + prediction['width']
            y2 = y1 + prediction['height']
            object_coordinates.append([x1, y1, x2, y2])
            
            label = prediction['class']
            labels_found.append(label)
        
        
        # If both Center and Needle Tip found, then calculate angle
        if "Center" in labels_found and "Needle_Tip" in labels_found:
            
            # Grabs "Center" label coordinates
            center_indexes = [index for index, x in enumerate(labels_found) if x == "Center"]
            center_coordinates = object_coordinates[center_indexes[0]]
            
            # Finds center x and y coordinates for "Center" label bbox
            center_x_center = int(center_coordinates[0] 
                                  + (center_coordinates[2] - center_coordinates[0])/2
                                  )
            center_y_center = int(center_coordinates[1] 
                                  + (center_coordinates[3] - center_coordinates[1])/2
                                  )
            
            # Grabs "Needle_Tip" label coordinates
            needle_tip_indexes = [index for index, x in enumerate(labels_found) if x == "Needle_Tip"]
            needle_tip_coordinates = object_coordinates[needle_tip_indexes[0]]
            
            # Finds center x and y coordinates for "Needle_Tip" label bbox
            center_x_needle_tip = int(needle_tip_coordinates[0] 
                                      + (needle_tip_coordinates[2] - needle_tip_coordinates[0])/2
                                      )
            center_y_needle_tip = int(needle_tip_coordinates[1] 
                                      + (needle_tip_coordinates[3] - needle_tip_coordinates[1])/2
                                      )
            
            # Finds angle - look at triginometry and arctangent
            dy = center_y_needle_tip - center_y_center
            dx = center_x_needle_tip - center_x_center
            theta = math.atan2(dy, dx)
            theta = math.degrees(theta)
            theta = round(theta)
            
            # Changes negative theta to appropriate value
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
            
            # In video, creates label for cooridnates on center bbox 
            #  and number on needle tip bbox
            for label_index, label in enumerate(labels_found):
                if "Center" in label:
                    labels_found[label_index] = label + " " +  str(theta) + " deg"
                if "Needle_Tip" in label:
                    psi = int(15.21*theta-638.21)
                    labels_found[label_index] = label + " " +  str(psi) + " psi"
        
        # Writes text and boxes on each frame - used for boxes, degrees, and psi
        for object_coordinate_index, object_coordinate in enumerate(object_coordinates):
            
            # Recangle settings
            start_point = ( int(object_coordinate[0]), int(object_coordinate[1]) )
            end_point = ( int(object_coordinate[2]), int(object_coordinate[3]) )
            color_1 = (255, 0, 255) # Magenta
            color_2 = (255, 255, 255) # White
            thickness = 1
            
            cv2.rectangle(image_b4_color, start_point, end_point, 
                          color_1, thickness)
            
            # For text
            start_point_text = (start_point[0], max(start_point[1]-5,0) )
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            
            cv2.putText(image_b4_color, 
                        labels_found[object_coordinate_index], 
                        start_point_text, font, fontScale, color_2, thickness)
            
        # Saves video with bounding boxes
        video_out.write(image_b4_color)
        
    video_out.release()


print("Done!")
# =============================================================================
