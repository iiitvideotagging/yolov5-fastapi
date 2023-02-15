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
from detect_or_track_y5 import detect_run_updated
from obj_det_and_trk_streamlit import detect_and_track

cfg_model_path = "models/best.pt"

cfg_enable_url_download = True  # True
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
            # model.cuda() if device == 'cuda' else model.cpu()
            model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # --Display prediction

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key


def prepare_classes_string(selected_options):
    bdd100k_classes = {0: 'traffic light',
                       1: 'traffic sign',
                       2: 'car',
                       3: 'pedestrian',
                       4: 'bus',
                       5: 'truck',
                       6: 'rider',
                       7: 'bicycle',
                       8: 'motorcycle',
                       9: 'train',
                       10: 'other vehicle',
                       11: 'other person',
                       12: 'trailer'}
    class_string_required = ''
    if selected_options is not None:
        class_string_required = [get_key(bdd100k_classes, option) for option in selected_options]
        # for option in selected_options:
        #    class_string_required = class_string_required + ' ' + get_key(bdd100k_classes, option)
    # print(class_string_required)
    # class_string_required = [0, 2, 3]
    return class_string_required


def videoInput(device, src, iou_score, confidence_score):
    i = -1
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    container = st.sidebar.container()
    all = st.sidebar.checkbox("Select all")
    options = ['traffic light', 'traffic sign', 'car', 'pedestrian', 'bus',
               'truck', 'rider', 'bicycle', 'motorcycle', 'train', 'trailer']
    tracking_required = st.sidebar.radio("Do we need to track objects?", ['Yes', 'No'], disabled=False, index=0)
    # print(f"Require tracking:{tracking_required}")
    save_output_video = 'Yes'  # st.sidebar.radio("Save output video?", ['Yes', 'No'], disabled=True, index=0)

    if all:
        selected_options = container.multiselect("Objects of interest on road scene:",
                                                 options, options)
    else:
        selected_options = container.multiselect("Objects of interest on road scene:",
                                                 options, ['car', 'bicycle'])
    classes_string = prepare_classes_string(selected_options)
    # print(selected_options)

    if save_output_video == 'Yes':
        no_save = False
        display_labels = False
    else:
        no_save = True
        display_labels = True

    if isinstance(uploaded_video, streamlit.runtime.uploaded_file_manager.UploadedFile):
        # print(uploaded_video)
        # print(type(uploaded_video))
        # print("i:" + str(i))

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

        submit = st.button("Predict!")
        if submit:
            # print(f"Tracking required just before start:{tracking_required}")
            if tracking_required == "Yes":
                stframe = st.empty()
                st.markdown("""<h4 style="color:black;"> Memory Overall Statistics</h4>""", unsafe_allow_html=True)
                kpi5, kpi6 = st.columns(2)

                with kpi5:
                    st.markdown("""<h6 style="color:black;">CPU Utilization</h6>""", unsafe_allow_html=True)
                    kpi5_text = st.markdown("0")

                with kpi6:
                    st.markdown("""<h6 style="color:black;">Memory Usage</h6>""", unsafe_allow_html=True)
                    kpi6_text = st.markdown("0")

                output_path = detect_run_updated(
                    weights=cfg_model_path,
                    source=img_path,
                    conf_thres=confidence_score,
                    device="cpu",
                    nosave=no_save,
                    hide_labels=False,
                    classes=classes_string,
                    track=True,
                    unique_track_color=True,
                    stframe=stframe,
                    kpi5_text=kpi5_text,
                    kpi6_text=kpi6_text
                )

                # output_path = detect_and_track(weights=cfg_model_path,
                #                                source=img_path,
                #                                stframe=stframe,
                #                                kpi5_text=kpi5_text,
                #                                kpi6_text=kpi6_text,
                #                                conf_thres=confidence_score,
                #                                device="cpu",
                #                                nosave=no_save,
                #                                display_labels=display_labels,
                #                                classes=classes_string)

            else:
                hide_labels = False if display_labels else True
                output_path = detect(weights=cfg_model_path, source=img_path, device='cpu', iou_thres=iou_score,
                                     conf_thres=confidence_score, line_thickness=1, nosave=no_save,
                                     hide_labels=hide_labels, classes=classes_string)

            # print("Output path final ", output_path)
            new_video_file_name = str(ts).replace("-", "_") + "_improved_video.mp4"
            new_video_path = os.path.join(Path.cwd(), 'data', 'outputs', new_video_file_name)
            # os.chdir('C://Users/Alex/')
            # subprocess.call(['ffmpeg', '-i', output_path, '-vcodec libx264 ', new_video_path ])
            # call_with_output(['ffmpeg', '-i', output_path, '-vcodec libx264 ', new_video_path])
            st.markdown("""<h4 style="color:black;"> Final Video Generated </h4>""", unsafe_allow_html=True)
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
    option = st.sidebar.radio("Select input type.", ['Video', 'Image'], index=0)
    datasrc = 'Upload your own data.'
    iou_score = st.sidebar.slider('Select IoU threshold', min_value=0.5, max_value=1.0, step=0.05)
    confidence_score = st.sidebar.slider('Select confidence threshold', min_value=0.1, max_value=1.0, step=0.01)
    deviceoption = 'cuda'

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
    # print(f"Model Downloaded, ETA:{finished_dl - start_dl}")


if cfg_enable_url_download:
    loadModel()
