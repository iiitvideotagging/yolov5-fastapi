from pathlib import Path

import streamlit
import streamlit as st
import torch
from detect import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time
import uuid
import random

import os
import subprocess


## CFG
cfg_model_path = "models/best.pt"

cfg_enable_url_download = False    #True
if cfg_enable_url_download:
    # url = "https://archive.org/download/yoloTrained/yoloTrained.pt"  # Configure this if you set
    # cfg_enable_url_download to True
    url = "https://archive.org/download/yolov5_custom_202302/best.pt"
    cfg_model_path = f"models/{url.split('/')[-1:][0]}"  # config model path from url name


## END OF CFG
def imageInput(device, src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True)
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # --Display prediction

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')


def videoInput(device, src, iou_score, confidence_score):
    i = -1
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    tracking_required = st.sidebar.radio("Do we need to track objects?", ['Yes', 'No'], disabled=False, index=0)
    print(tracking_required)
    if isinstance(uploaded_video, streamlit.runtime.uploaded_file_manager.UploadedFile):
        print(uploaded_video)
        print(type(uploaded_video))
        print("i:" + str(i))
        submit = st.button("Predict!")

        if submit:
            ts = random.randrange(20, 50000, 3)  # datetime.timestamp(datetime.now())
            generated_file_name = str(ts).replace("-", "_") + uploaded_video.name
            img_path = os.path.join('data', 'uploads', generated_file_name)
            output_path = os.path.join(Path.cwd(), 'data', 'output',
                                       generated_file_name)  # os.path.basename(img_path))

            with open(img_path, mode='wb') as f:
                f.write(uploaded_video.read())  # save video to disk

            st_video = open(img_path, 'rb')
            video_bytes = st_video.read()
            st.write("Uploaded Video")
            st.video(video_bytes)

            if device == 'cuda':
                output_path = detect(weights=cfg_model_path, source=img_path, device=0, iou_thres=iou_score,
                                     conf_thres=confidence_score, line_thickness=1)
            else:
                output_path = detect(weights=cfg_model_path, source=img_path, device='cpu', iou_thres=iou_score,
                                     conf_thres=confidence_score, line_thickness=1)

            print("Output path", output_path)

            new_video_file_name = str(ts).replace("-", "_") + uploaded_video.name
            new_video_path = os.path.join(Path.cwd(), 'data', 'outputs', new_video_file_name)
            # os.chdir('C://Users/Alex/')
            # subprocess.call(['ffmpeg', '-i', output_path, '-vcodec libx264 ', new_video_path ])
            # call_with_output(['ffmpeg', '-i', output_path, '-vcodec libx264 ', new_video_path])
            st.write("Model Prediction")

            cmd = "ffmpeg -i " + output_path + " -vcodec libx264 " + new_video_path
            os.system(cmd)

            with open(new_video_path, 'rb') as v:
                st.video(v)
            # st_video2 = open(output_path, 'rb')
            # video_bytes2 = st_video2.read()
            # st.video(video_bytes2)
            i = -1


def call_with_output(command):
    success = False
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
        success = True
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
    except Exception as e:
        # check_call can raise other exceptions, such as FileNotFoundError
        output = str(e)
    return success, output

def main():
    # -- Sidebar
    st.sidebar.title('⚙️User Configurations')
    option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    datasrc = 'Upload your own data.'
    iou_score = st.sidebar.slider('Select IoU threshold', min_value=0.5, max_value=1.0, step=0.05)
    confidence_score = st.sidebar.slider('Select confidence threshold', min_value=0.1, max_value=1.0, step=0.01)
    deviceoption = 'cuda'

    """
    datasrc = st.sidebar.radio("Select input source.", ['From BDD100K Dataset.', 'Upload your own data.'],
                        disabled=True, index=1, label_visibility=False)
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled=False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled=False, index=0)
    """
    # -- End of Sidebar
    st.header(':oncoming_automobile: IIIT - Multi Object Detection')
    st.subheader('👈🏽 Select options left-handed menu bar.')

    if option == "Image":
        imageInput(deviceoption, datasrc)
    elif option == "Video":
        videoInput(deviceoption, datasrc, iou_score, confidence_score)


if __name__ == '__main__':
    main()


# Downlaod Model from url.
@st.cache
def loadModel():
    start_dl = time.time()
    model_file = wget.download(url, out="models/")
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl - start_dl}")


if cfg_enable_url_download:
    loadModel()
