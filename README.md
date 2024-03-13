# Microsoft Computer Vision Demo

## Introduction 
This repo aims to provide an insight into computer vision models running on Azure. It covers a wide range of use cases utilizing custom segmentation models (trained on Azure Machine Learning Studio) and Azure Computer Vision models on Cognitive Services. The demo website is deployed on the [Data Science Toolkit](https://www.ds-toolkit.com/).

## Getting Started
The demo website can be accessed [here](https://computer-vision.victorioussea-621a1cbd.westeurope.azurecontainerapps.io/), but if you wish to setup and host your own web app, follow these steps:

- I. Clone this repository on a vm or local machine.
- II. In the terminal, create a new Python enviroment (3.10 was used in demo) and activate it.
    - Use the command: pip install virtualenv
    - Use the command: python -m virtualenv env
    - Use the command: env\Scripts\Activate
- III. Open cloned folder in terminal.
    - Use the command: cd "folder path"
- IV. Install all libraries listed in requirements.txt. 
    - Use the command: pip install -r .\requirements.txt
- V. Replace Cognitive Services enviroment API keys with your own:
    - The Azure Cognitive Services resource required is: "Azure AI services multi-service account". Perform V.i OR V.ii. 
    - V.i Set enviroment variables called: "ai-multiaccount-endpoint" and "ai-multiaccount-apikey"
    - V.ii Add a json file to "/static/assets/endpoints.json" with values for "endpoint" and "key" and uncomment code at the top of app.py. 
- VI. Run the web app
    - Use the command: python flask_app\app.py

## Build and Test
The website is created using HTML/ CSS and Python (Flask). Run the app.py and load the website in your browser. 
Apart from the home page, there are nine demos. Each demo has a seperate HTML page associated with it. The majority of the demos allow you to upload your own images to test them out.
Each demo is outined below:

1. Optical Character Recognition - uses Azure cognitive services to read text from images. 
2. Object Detection - uses Azure cognitive services to idenitify and locate objects in images.
3. Semantic Segmentation - uses a custom trained model using UNET architecture (trained on Azure Machine Learning Studio) to locate and classify every pixel belonging to a single class (in this case, cats).
4. Image Captioning - uses Azure cognitive services to describe images.
5. Image Classification - uses Azure cognitive services to classify images as belonging to a single class (landmarks in this case).
6. Detect Sensitive Content - uses Azure cognitive services to idenitify and flag sensitive content (Adult, Gore, Racy).
7. Smart Cropping - uses Azure cognitive services to idenitify an area of interest and crop an image to this area.
8. Brands Detection - uses Azure cognitive services to idenitify and locate brands in an image.
9. Motion Detection (Video) - uses simple computer vision techniques to idenitify the motion of objects in a video.

## Contribute
Please reach out to henrytaylor@microsoft.com, for any questions, suggestions, or improvements. Thank you!

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
Please follow our [Contrubution Guidence](https://docs.opensource.microsoft.com/releasing/#contribution-guidance)

Special thanks to walter.grasselli@microsoft.com for support in hosting the web application.

## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.