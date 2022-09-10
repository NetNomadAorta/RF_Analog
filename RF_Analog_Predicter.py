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
# For email section
import yaml
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid


# User parameters
TO_PREDICT_PATH         = "./To_Predict_Videos/"
PREDICTED_PATH          = "./Predicted_Videos/"
SAVE_ANNOTATED_VIDEOS   = True
MIN_SCORE               = 0.4 # Minimum object detection score


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



# Main()
# =============================================================================
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

# Start FPS timer
fps_start_time = time.time()

for video_name in os.listdir(TO_PREDICT_PATH):
    video_path = os.path.join(TO_PREDICT_PATH, video_name)
    
    
    video_capture = cv2.VideoCapture(video_path)
    
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = round( video_capture.get(cv2.CAP_PROP_FPS) )
    
    success, image_b4_color = video_capture.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(PREDICTED_PATH + video_name, fourcc, video_fps, 
                                (int(image_b4_color.shape[1]), 
                                 int(image_b4_color.shape[0])
                                 )
                                )
    
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
            "https://detect.roboflow.com/analog/10",
            "?api_key=kAGiAjfXg1MNA0NfST4F",
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
            x1 = prediction['x'] - prediction['width']/2
            y1 = prediction['y'] - prediction['height']/2
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
            
            # Puts cooridnates on center label and number on needle tip
            for label_index, label in enumerate(labels_found):
                if "Center" in label:
                    labels_found[label_index] = label + " " +  str(theta) + " deg"
                if "Needle_Tip" in label:
                    psi = int(15.21*theta-638.21)
                    labels_found[label_index] = label + " " +  str(psi) + " psi"
        
        if SAVE_ANNOTATED_VIDEOS:
            
            for dieCoordinate_index, dieCoordinate in enumerate(die_coordinates):
                start_point = ( int(dieCoordinate[0]), int(dieCoordinate[1]) )
                end_point = ( int(dieCoordinate[2]), int(dieCoordinate[3]) )
                color_1 = (255, 0, 255)
                color_2 = (255, 255, 255)
                thickness = 1
                
                cv2.rectangle(image_b4_color, start_point, end_point, color_1, thickness)
                
                if "Center" in labels_found and "Needle_Tip" in labels_found:
                    # Draws line from needle base to tip
                    cv2.line(image_b4_color, 
                             (center_x_center, center_y_center), 
                             (center_x_needle_tip, center_y_needle_tip), 
                             color_2, 
                             thickness=thickness
                             )
                
                start_point_text = (start_point[0], max(start_point[1]-5,0) )
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                thickness = 1
                cv2.putText(image_b4_color, 
                            labels_found[dieCoordinate_index], 
                            start_point_text, font, fontScale, color_2, thickness)
            
        # Saves video with bounding boxes
        video_out.write(image_b4_color)
        
    video_out.release()


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)
# =============================================================================
