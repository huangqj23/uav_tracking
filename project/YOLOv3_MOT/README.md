



#### IMPORTANT! ####
# Install NVIDIA Driver
If you have an NVIDIA GPU on your computer, please ensure that you have correctly installed the NVIDIA driver for your GPU. If you haven't installed it yet, you can find the driver download link on the official NVIDIA website ("https://www.nvidia.com/download/index.aspx").  Download and install the driver according to your GPU model and operating system. 

# Download YOLOv3 Weights
To use MOT with YOLOv3, you must download the "yolov3.weights" file from the following link: "https://pjreddie.com/media/files/yolov3.weights" and then save it to the following path: "/MOT_PROJECTS_BOOK/Chapter5/yolov3/weights/".



# To execute MOT using YOLOv3, run the following commands:

- for the first video:
        python mot_yolov3.py --video-path ./videos/video1.mp4

- for the second video:
        python mot_yolov3.py --video-path ./videos/video2.mp4


# To execute MOT using pre-saved YOLOv3, run the following commands:
- for the first video:
        python mot_presaved.py --source-path ./detections/video1
- for the second video:
        python mot_presaved.py --source-path ./detections/video2

