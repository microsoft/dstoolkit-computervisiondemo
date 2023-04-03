#########################
# Import Libraries 
#########################
from flask import Flask, request, render_template
import numpy as np
import json
import os
import ssl
import sys
import time 
import cv2
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes


#########################
# Define Common Functions 
#########################
def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
allowSelfSignedHttps(True)

def cleanDir(dir):
    """Cleans a directory to save storage and space on server. Cleans based on time - after a given number of time since
    creation of file, its deleted."""
    file_dir = os.listdir(dir)
    current_time = time.time()
    for file in file_dir: # clean up old files
        file_timeCreation = os.path.getmtime(dir + file)
        time_difference_mins = (current_time - file_timeCreation) / 60
        if time_difference_mins >= 10: # mins
            os.remove(dir + file)

def saveImg(dir, image, prefix=None, image_name=None):
    """Saved image to given directory. Returns image name."""
    if prefix == None:
        image_name = image.filename
        image.save(dir + image_name)
        return image_name 
    else:
        image.save(dir + prefix + image_name)
        return  prefix + image_name 

#########################
# Define Web App Backend 
#########################
app = Flask(__name__)
microsoft_colours = ["#f25022", "#80ba01", "#02a4ef", "#ffb902"]

app_path = sys.path[0] # file paths
app.config["JSON_PATH"] = app_path + "/static/assets/endpoints.json"
app.config["IMAGE_UPLOADS"] = app_path + "/static/assets/img_upload/"
app.config["IMAGE_WEB"] = app_path + "/static/assets/img/"

with open(app.config["JSON_PATH"]) as file: # Parameters
    json_file = json.load(file)
endpoint = json_file['endpoint']
key = json_file['key']
region = json_file['region']

# Setup CV multi resource credentials
credentials = CognitiveServicesCredentials(key)
client = ComputerVisionClient(
    endpoint=endpoint,
    credentials=credentials)

@app.route('/', methods=['GET'])
def home():
    cleanDir(app.config["IMAGE_UPLOADS"])
    return render_template('home.html')

@app.route("/ocr", methods=['GET', 'POST'])
def ocr():
    if request.method == 'POST':
        cleanDir(app.config["IMAGE_UPLOADS"])
        image = request.files["image"]
        fontSizeProp = float(request.form['fontSizeProp'])
        boxWidthProp = int(fontSizeProp + 2)
        hex = request.form['HexProp']
        RGBProp = hex_to_rgb(hex)
        RGBProp = (RGBProp[2], RGBProp[1], RGBProp[0]) # reorder for openCV (seems to be BGR)
        image_name = saveImg(app.config["IMAGE_UPLOADS"], image)
    
        with open(app.config["IMAGE_UPLOADS"] + image_name, "rb") as image_stream:
            rawHttpResponse = client.read_in_stream(image=image_stream, language="en", mode="Printed", raw=True) # local

        # Get ID from returned headers
        numberOfCharsInOperationId = 36
        operationLocation = rawHttpResponse.headers["Operation-Location"]
        idLocation = len(operationLocation) - numberOfCharsInOperationId
        operationId = operationLocation[idLocation:]
        result = client.get_read_result(operationId)
        
        bounding_box_text = {}
        while result.status.lower() in ['notstarted', 'running']:
            time.sleep(1)
            result = client.get_read_result(operationId)
            if result.status == OperationStatusCodes.succeeded:
                for line in result.analyze_result.read_results[0].lines:
                    bounding_box_text[line.text] = line.bounding_box
        
        uploaded_img = cv2.imread(app.config["IMAGE_UPLOADS"] + image_name)    
        if len(bounding_box_text.keys())>0:
            for key in list(bounding_box_text.keys()):
                bounding_box_list = bounding_box_text[key]
                topX, topY, botX, botY = int(bounding_box_list[0]),  int(bounding_box_list[1]),  int(bounding_box_list[4]),  int(bounding_box_list[5])
                cv2.rectangle(uploaded_img,(topX, topY),(botX, botY),RGBProp,boxWidthProp)
                cv2.putText(uploaded_img, text=key, org=(topX+5, topY+15), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontSizeProp, color=RGBProp,thickness=1)
                cv2.imwrite(app.config["IMAGE_UPLOADS"] + "OCR_RETURNED_" + image_name, uploaded_img)
            image_return_caption = "OCR Returned Image"

        else:
            cv2.imwrite(app.config["IMAGE_UPLOADS"] + "OCR_RETURNED_" + image_name, uploaded_img)
            image_return_caption = "OCR Found No Text"
                        
        return render_template("ocr.html", 
                               image_upload = "img_upload/" + image_name, 
                               image_return = "img_upload/" + "OCR_RETURNED_" + image_name,
                               image_upload_caption = "User Supplied Image",
                               image_return_caption = image_return_caption,
                               fontSizeProp=fontSizeProp, 
                               HexProp=hex)
    else:
        return render_template("ocr.html", 
                               image_upload = "img/" + "OCR_pre.jpeg", 
                               image_return = "img/" + "OCR_post.jpeg",
                               image_upload_caption = "Example: User Supplied Image",
                               image_return_caption = "Example: OCR Returned Image",
                               fontSizeProp=1, 
                               HexProp="#ffffff")
    
@app.route('/objectdetection', methods=['GET', 'POST'])
def objectdetection():
    if request.method == 'POST':
        cleanDir(app.config["IMAGE_UPLOADS"])
        image = request.files["image"]
        fontSizeProp = float(request.form['fontSizeProp'])
        boxWidthProp = int(fontSizeProp + 2)
        hex = request.form['HexProp']
        ModelConfidence = int(request.form['ModelConfidence'])
        RGBProp = hex_to_rgb(hex)
        RGBProp = (RGBProp[2], RGBProp[1], RGBProp[0]) # reorder for openCV (seems to be BGR)
        image_name = saveImg(app.config["IMAGE_UPLOADS"], image)
                
        with open(app.config["IMAGE_UPLOADS"] + image_name, "rb") as image_stream:
            rawHttpResponse = client.analyze_image_in_stream(image=image_stream, visual_features=[VisualFeatureTypes.objects]) 

        label_dict = {}
        for num, objectDet in enumerate(rawHttpResponse.objects):
            obj_confidence = objectDet.confidence
            dict_name = f"obj_{num}"
            if (obj_confidence * 100) >= ModelConfidence:
                bounding_box_dict = objectDet.rectangle
                topX, topY, botX, botY = int(bounding_box_dict.x),  int(bounding_box_dict.y),  int(bounding_box_dict.x) + int(bounding_box_dict.w), int(bounding_box_dict.y) +  int(bounding_box_dict.h) 
                mini_dict = {}
                mini_dict['topX'] = topX
                mini_dict['topY'] = topY
                mini_dict['botX'] = botX
                mini_dict['botY'] = botY
                mini_dict['obj_confidence'] = np.round(obj_confidence, 2)
                mini_dict['name']= objectDet.object_property
                label_dict[dict_name] = mini_dict
                
        uploaded_img = cv2.imread(app.config["IMAGE_UPLOADS"] + image_name)   
        if len(label_dict.keys())>0:
            for dict_name in label_dict.keys():
                mini_dict = label_dict[dict_name]
                cv2.rectangle(uploaded_img,(mini_dict['topX'], mini_dict['topY']),(mini_dict['botX'], mini_dict['botY']),RGBProp,boxWidthProp)
                cv2.putText(uploaded_img, text=f"{mini_dict['name']} {100 * mini_dict['obj_confidence']}%", org=(mini_dict['topX']+5, (mini_dict['topY']+35)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontSizeProp, color=RGBProp,thickness=2)
                cv2.imwrite(app.config["IMAGE_UPLOADS"] + "OD_RETURNED_" + image_name, uploaded_img)
            image_return_caption = "Object Detection Returned Image"
        else:
            cv2.imwrite(app.config["IMAGE_UPLOADS"] + "OD_RETURNED_" + image_name, uploaded_img)
            image_return_caption = f"Object Detection Found No Objects At Model Confidence Level {ModelConfidence}%"
                    
        return render_template("objectdetection.html", 
                               image_upload = "img_upload/" + image_name, 
                               image_return = "img_upload/" + "OD_RETURNED_" + image_name,
                               image_upload_caption = "User Supplied Image",
                               image_return_caption = image_return_caption,
                               fontSizeProp=fontSizeProp, 
                               HexProp=hex,
                               ModelConfidence=ModelConfidence)
    else:
        return render_template("objectdetection.html", 
                               image_upload = "img/" + "OD_pre.jpg", 
                               image_return = "img/" + "OD_post.jpg",
                               image_upload_caption = "Example: User Supplied Image",
                               image_return_caption = "Example: Object Detection Returned Image",
                               fontSizeProp=1, 
                               HexProp="#ffffff",
                               ModelConfidence=70)
 
@app.route('/brandsdetection', methods=['GET', 'POST'])
def brandsdetection():
    if request.method == 'POST':
        cleanDir(app.config["IMAGE_UPLOADS"])
        image = request.files["image"]
        fontSizeProp = float(request.form['fontSizeProp'])
        boxWidthProp = int(fontSizeProp + 2)
        hex = request.form['HexProp']
        ModelConfidence = int(request.form['ModelConfidence'])
        RGBProp = hex_to_rgb(hex)
        RGBProp = (RGBProp[2], RGBProp[1], RGBProp[0]) # reorder for openCV (seems to be BGR)
        image_name = saveImg(app.config["IMAGE_UPLOADS"], image)
                
        with open(app.config["IMAGE_UPLOADS"] + image_name, "rb") as image_stream:
            rawHttpResponse = client.analyze_image_in_stream(image=image_stream, visual_features=[VisualFeatureTypes.brands]) 

        label_dict = {}
        for num, objectDet in enumerate(rawHttpResponse.brands):
            obj_confidence = objectDet.confidence
            dict_name = f"obj_{num}"
            if (obj_confidence * 100) >= ModelConfidence:
                bounding_box_dict = objectDet.rectangle
                topX, topY, botX, botY = int(bounding_box_dict.x),  int(bounding_box_dict.y),  int(bounding_box_dict.x) + int(bounding_box_dict.w), int(bounding_box_dict.y) +  int(bounding_box_dict.h) 
                mini_dict = {}
                mini_dict['topX'] = topX
                mini_dict['topY'] = topY
                mini_dict['botX'] = botX
                mini_dict['botY'] = botY
                mini_dict['obj_confidence'] = np.round(obj_confidence, 2)
                mini_dict['name']= objectDet.name
                label_dict[dict_name] = mini_dict
                
        uploaded_img = cv2.imread(app.config["IMAGE_UPLOADS"] + image_name)   
        if len(label_dict.keys())>0:
            for dict_name in label_dict.keys():
                mini_dict = label_dict[dict_name]
                print("\n\n\n", mini_dict, "\n\n\n")
                cv2.rectangle(uploaded_img,(mini_dict['topX'], mini_dict['topY']),(mini_dict['botX'], mini_dict['botY']),RGBProp,boxWidthProp)
                cv2.putText(uploaded_img, text=f"{mini_dict['name']} {100 * mini_dict['obj_confidence']}%", org=(mini_dict['topX']+5, (mini_dict['topY']-10)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontSizeProp, color=RGBProp,thickness=2)
                cv2.imwrite(app.config["IMAGE_UPLOADS"] + "OD_RETURNED_" + image_name, uploaded_img)
            image_return_caption = "Brands Detection Returned Image"
        else:
            cv2.imwrite(app.config["IMAGE_UPLOADS"] + "OD_RETURNED_" + image_name, uploaded_img)
            image_return_caption = f"Brands Detection Found No Objects At Model Confidence Level {ModelConfidence}%"

        return render_template("brandsdetection.html", 
                               image_upload = "img_upload/" + image_name, 
                               image_return = "img_upload/" + "OD_RETURNED_" + image_name,
                               image_upload_caption = "User Supplied Image",
                               image_return_caption = image_return_caption,
                               fontSizeProp=fontSizeProp, 
                               HexProp=hex,
                               ModelConfidence=ModelConfidence)
    else:
        return render_template("brandsdetection.html", 
                               image_upload = "img/" + "BD_pre.jpg", 
                               image_return = "img/" + "BD_post.jpg",
                               image_upload_caption = "Example: User Supplied Image",
                               image_return_caption = "Example: Brands Detection Returned Image",
                               fontSizeProp=1, 
                               HexProp="#ffffff",
                               ModelConfidence=70)

@app.route('/tagdescribe', methods=['GET', 'POST'])
def tagdescribe():
    if request.method == 'POST':
        cleanDir(app.config["IMAGE_UPLOADS"])
        image = request.files["image"]
        image_name = saveImg(app.config["IMAGE_UPLOADS"], image)
                
        with open(app.config["IMAGE_UPLOADS"] + image_name, "rb") as image_stream:
            rawHttpResponse = client.analyze_image_in_stream(image=image_stream, visual_features=[VisualFeatureTypes.description]) 
        
        tags = rawHttpResponse.description.tags
        tag_return = "Tags: " + ','.join(tags)

        caption_confidence = {}
        for desc in rawHttpResponse.description.captions:
            caption_confidence[desc.text] = desc.confidence
        
        desc_return = "Caption: "
        for num, desc_key in enumerate(caption_confidence.keys()):
            if num != 0:  
                desc_return = desc_return + " | "
            desc_return = desc_return + f"{desc_key} ({str(round(100 * caption_confidence[desc_key], 2))}%)"
    
        return render_template("tagdescribe.html", 
                               image_upload = "img_upload/" + image_name, 
                               image_description = desc_return,
                               image_tags = tag_return)
    else:
        return render_template("tagdescribe.html", 
                               image_upload = "img/" + "cap_pre.jpg", 
                               image_description = "Example Caption: a body of water with trees and a cloudy sky (49.89%)",
                               image_tags = "Example Tags: outdoor,sky,water,nature,clouds,pond,lake")


@app.route('/imageclassification', methods=['GET', 'POST'])
def imageclassification():
    if request.method == 'POST':
        cleanDir(app.config["IMAGE_UPLOADS"])
        image = request.files["image"]
        image_name = saveImg(app.config["IMAGE_UPLOADS"], image)    
        model_type = "landmarks" # request.form['model_selection']
        with open(app.config["IMAGE_UPLOADS"] + image_name, "rb") as image_stream:
            rawHttpResponse = client.analyze_image_by_domain_in_stream(image=image_stream, model=model_type) 
        try:
            top_result = rawHttpResponse.result['landmarks'][0]
            name = top_result["name"]
            confidence = round(top_result["confidence"]* 100, 2) 
            return_text = f"The model classifies this image as {name} with a confidence of {confidence}%."
        except:
            return_text = f"The landmarks model cannot classify the contents based on the image provided."
          
        return render_template("imageclassification.html", 
                               image_upload = "img_upload/" + image_name, 
                               image_upload_caption = "User Supplied Image", 
                               image_classification_text = return_text)
    else:
        return render_template("imageclassification.html", 
                               image_upload = "img/" + "IC_pre.jpg", 
                               image_upload_caption = "Example: User Supplied Image",
                               image_classification_text = "Example: The model classifies this image as Blackpool Tower with a confidence of 99.99%.")


@app.route('/detectsensitive', methods=['GET', 'POST'])
def detectsensitive():
    if request.method == 'POST':
        cleanDir(app.config["IMAGE_UPLOADS"])
        image_name = request.form["select_img"]
        with open(app.config["IMAGE_WEB"] + image_name + ".jpg", "rb") as image_stream:
            rawHttpResponse = client.analyze_image_in_stream(image=image_stream, visual_features=[VisualFeatureTypes.adult]) 
        results = rawHttpResponse.adult
        adult_score = str(round(results.adult_score * 100, 2)) + "%"
        racy_score = str(round(results.racy_score * 100, 2)) + "%"
        gore_score = str(round(results.gore_score * 100, 2)) + "%"

        adult_string = f"Adult Score: {adult_score}"
        racy_string = f"Racy Score: {racy_score}"
        gore_string = f"Gore Score: {gore_score}"
          
        return render_template("detectsensitive.html", 
                               main_image_name = image_name,
                               adult_string = adult_string,
                               racy_string = racy_string,
                               gore_string = gore_string)
    else:
        return render_template("detectsensitive.html",
                               main_image_name = "",
                               adult_string = "",
                               racy_string = "",
                               gore_string = "")
        
        
@app.route('/smartcropping', methods=['GET', 'POST'])
def smartcropping():
    if request.method == 'POST':
        cleanDir(app.config["IMAGE_UPLOADS"])
        image = request.files["image"]
        image_name = saveImg(app.config["IMAGE_UPLOADS"], image)

        with open(app.config["IMAGE_UPLOADS"] + image_name, "rb") as image_stream:
            rawHttpResponse = client.get_area_of_interest_in_stream(image=image_stream, smart_cropping=True) 

        bounding_box = rawHttpResponse.area_of_interest
        topX, topY, botX, botY = int(bounding_box.x),  int(bounding_box.y),  int(bounding_box.x) + int(bounding_box.w), int(bounding_box.y) +  int(bounding_box.h) 
        uploaded_img = cv2.imread(app.config["IMAGE_UPLOADS"] + image_name)   
        crop_img = uploaded_img[topY:botY, topX:botX]
        cv2.imwrite(app.config["IMAGE_UPLOADS"] + "smartCropRTN_" + image_name, crop_img)
        
        return render_template("smartcropping.html", 
                               image_upload = "img_upload/" + image_name, 
                               image_return = "img_upload/" + "smartCropRTN_" + image_name,
                               image_arrow = "img/" + "downarrow.png", 
                               image_upload_caption = "User Supplied Image",
                               image_return_caption = "Smart Cropped User Supplied Image"
                                )   
    else:
        return render_template("smartcropping.html", 
                               image_upload = "img/" + "SC_pre.jpg", 
                               image_return = "img/" + "SC_post.png", 
                               image_arrow = "img/" + "downarrow.png", 
                               image_upload_caption = "Example: User Supplied Image",
                               image_return_caption = "Example: Smart Cropped User Supplied Image"
                               )    
          
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
    