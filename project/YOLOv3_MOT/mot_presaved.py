import cv2
from tracker.tracker import Tracker
from utils.utils import colors_list, PutText
from tracker.tracker_utils import load_json
import argparse
import numpy as np
import os
import glob

def main(args):

    configs=load_json(args.detector_config)
    yolo_dim=configs["im_size"]
    output_dim = args.out_dim
    tracker_params = load_json(args.tracker_params)
    tracker = Tracker(tracker_params)
    all_dets= np.load(f"{args.source_path}/labels/detection.npy")
    img_folder_path = f"{args.source_path}/images"
    image_files = glob.glob(os.path.join(img_folder_path, '*.jpg'))  
    num_frame = len(image_files)

    for frame_num in range(num_frame):
        # print(f"Frame: {frame_num}")
        # Load current image        
        img_file='{:06d}.jpg'.format(frame_num)
        im_path=f"{args.source_path}/images/image{img_file}"        
        img = cv2.imread(im_path)

        outframe = cv2.resize(img, output_dim)

        detections = all_dets[all_dets[:, 0] == frame_num]
       
        # Select object types to track
        to_filter = [2, 5, 7] # car, bus and truck   (3=>motorbike)
        filters = np.logical_or.reduce([detections[:, 7] == v for v in to_filter])
        detections = detections[filters]

        det_boxes=detections[:,1:5]
        info_boxes=detections[:,5:]
        dets_all = {'dets': det_boxes, 'info': info_boxes}        

        tracker.manage_tracks(dets_all)   
    
        for track in tracker.tracks:            
            if len(track.trace) > 0  and track.num_lost_dets <= 1:                
                t_id = track.track_id
                boxes = track.trace[0][:4].squeeze()
                
                # Find scale size between the output display and YOLOv3 input dimension
                rw = outframe.shape[1] / yolo_dim[1]
                rh = outframe.shape[0] / yolo_dim[0]

                # Transform bounding box coordinates relative to the output display
                x1y1 = (int(boxes[0] * rw ), int(boxes[1] * rh))
                x2y2 = (int(boxes[2] * rw ), int(boxes[3] * rh))

                outframe = cv2.rectangle(outframe, x1y1, x2y2, colors_list[int(t_id)], 2)

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
        
        cv2.imshow("MOT-machinelearningspace.com",outframe)         
        if cv2.waitKey(1) & 0xFF == ord('q'):                                            
                break 
        
    print('MOT have been performed successfully.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOT System Using Presaved YOLOV3')
    parser.add_argument('--detector-config', type=str, \
                        default='./data/config.json')
    parser.add_argument('--out-dim', type=tuple, \
                        default=(860, 640), help='A tuple of expected output dimensions')
    parser.add_argument('--tracker-params', type=str, \
                        default='tracker/tracker_params.json')
    parser.add_argument('--source-path', type=str, \
                        default='./detections/video1')      
    args = parser.parse_args()  
    main(args)