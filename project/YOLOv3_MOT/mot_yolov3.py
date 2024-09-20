import torch
import numpy as np
from yolov3.yolov3 import YOLOv3NET
import cv2
from yolov3.load_weights import load_weights
from utils.utils import colors_list,PutText
import argparse
from yolov3.utils.utils import post_processing, load_json
from tracker.tracker import Tracker

def main(args):

    configs=load_json(args.detector_config)
    num_classes=configs["num_classes"]    
    yolo_dim = configs["im_size"]
    output_dim = args.out_dim    
    conf_thres = args.conf_thres
    nms_thresh = args.nms_thres
    weights_path = args.weights_path 
    
    model = YOLOv3NET(args.detector_config)
    model = load_weights(model,weights_path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() \
                                                and args.device != 'cpu' else "cpu")        
    model.to(device)
    
    cuda = device.type == 'cuda'

    tracker_params = load_json(args.tracker_params)
    tracker = Tracker(tracker_params)
    
    #Read the video and start tracking
    frame_num=0
    try:
        # Open the video file for reading using the OpenCV library
        VideoCap = cv2.VideoCapture(args.video_path)
        
        # Check if the video file was opened successfully
        if not VideoCap.isOpened():
            raise Exception("Failed to open the video file")

        while True:            
            ret, frame = VideoCap.read()
            if not ret:
                break
            print(f"Frame: {frame_num}")

            # resize the frame to the expected output display dimensions
            outframe = cv2.resize(frame, output_dim)

            # resize the frame to the size of YOLOv3 input 
            image = cv2.resize(frame, yolo_dim)                 
            image = torch.from_numpy(image.astype('float32')).permute(2, 0, 1)/255.0
            
            # Add a batch dimension to the tensor            
            image = image.unsqueeze(0)
            image = image.to(device)

            with torch.no_grad():    
                model.eval()
                outputs = model(x=image, CUDA=cuda)            
            
            detections = post_processing(outputs, conf_thres, num_classes, \
                                       nms_conf = nms_thresh)

            detections = detections.cpu()
            detections = np.array(detections)

            # Select object types to track
            to_filter = [2, 5, 7] # car, bus and truck
            filters = np.logical_or.reduce([detections[:, 6] == v for v in to_filter])
            detections = detections[filters]

            pred_box=detections[:,0:4]
            info_box=detections[:,4:7]
            dets_all = {'dets': pred_box, 'info': info_box}
            
            tracker.manage_tracks(dets_all)    

            for track in tracker.tracks:         
                if len(track.trace) > 0 and track.num_lost_dets <= 1:
                    t_id = track.track_id
                    boxes = track.trace[0][:4].ravel()

                    # Find the scale size between the output display and 
                    # YOLOv3 input dimensions
                    rw = outframe.shape[1] / yolo_dim[1]
                    rh = outframe.shape[0] / yolo_dim[0]

                    # Transform bounding box coordinates relative to the output display
                    x1y1 = (int(boxes[0] * rw ), int(boxes[1] * rh))
                    x2y2 = (int(boxes[2] * rw ), int(boxes[3] * rh))

                    outframe = cv2.rectangle(outframe, x1y1, x2y2, \
                                                        colors_list[int(t_id)], 2)

                    txt = f"Id:{str(int(t_id))}"

                    # Put Id number on each track
                    outframe = PutText(outframe,
                                    text=txt,
                                    pos=x1y1,
                                    text_color=(255, 255, 255),
                                    bg_color=colors_list[int(t_id)],
                                    scale=0.5,
                                    thickness=1,
                                    margin=2,
                                    transparent=True,
                                    alpha=0.5)

            # Write frame number on the scene                                                                  
            outframe=PutText(outframe,text=f"Frame:{frame_num}",
                    pos=(20,40),     
                    text_color=(15,15,255),
                    bg_color=(255,255,255),
                    scale=0.5,
                    thickness=1,
                    margin=3,                   
                    transparent=True,
                    alpha=0.8)        
            frame_num +=1  

            cv2.imshow("MOT-machinelearningspace.com",outframe)              
            if cv2.waitKey(1) & 0xFF == ord('q'):                            
                break 
    
    except Exception as e:
        print(f"Error: {e}")

    finally:
        cv2.destroyAllWindows()
        VideoCap.release()
        print('MOT have been performed successfully.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOT System Using Yolov3')
    
    parser.add_argument('--weights-path',type=str, \
                        default='./yolov3/weights/yolov3.weights',
                        help='weigths path')
    parser.add_argument('--out-dim', type=tuple, \
                        default=(860, 640), help='A tuple of expected output dimensions')
    parser.add_argument('--device', default='0', \
                        help='gpu number, ex: 0 or 1, or cpu')
    parser.add_argument('--detector-config', type=str, \
                        default='./yolov3/data/config.json')
    parser.add_argument('--tracker-params', type=str, \
                        default='./tracker/tracker_params.json')
    parser.add_argument('--conf-thres', type=float, \
                        default=0.5, help='confidence threshold')
    parser.add_argument('--nms-thres', type=float, \
                        default=0.5, help='NMS IoU threshold')
    parser.add_argument('--video-path', type=str, \
                        default='./videos/video1.mp4')    
    args = parser.parse_args()      

    main(args)