import numpy as np
from scipy.optimize import linear_sum_assignment
import json

def load_json(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def bbox_iou(box1, box2): 

    # Calculate the area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate the coordinates of the intersection box
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate the area of the intersection box
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Calculate the union area
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU and return the result
    iou = inter_area / union_area   
    
    return iou

def cost_matrix_iou(tracks, detections):
    M = len(tracks)
    N = len(detections)
    cost = np.zeros(shape=(M,N))    
    for i in range(M):
        for j in range(N):              
             cost[i][j] = bbox_iou(tracks[i].predicted[:4], detections[j])                 
    return cost

def track_association(tracks, detections, iou_th = 0.1):

    if not tracks:   
        return [], np.arange(len(detections)), []
           
    cost_matrix = cost_matrix_iou(tracks, detections)  
    
    # Find the optimal matches between detections and existing tracks
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    paired_indices =list(zip(row_ind, col_ind))

    # Find unmatched detections and trackers
    unpaired_tracks = [d for d, _ in enumerate(tracks) if d not in row_ind]
    unpaired_detections = [t for t, _ in enumerate(detections) if t not in col_ind]

    # Select the valid matches if IoU is greater than or equal to the threshold
    pairs = []
    for i,j in paired_indices:
        if cost_matrix[i][j] > iou_th:
            pairs.append((i,j))    
        else:
            unpaired_tracks.append(i)
            unpaired_detections.append(j)        
                        
    return pairs, unpaired_detections, unpaired_tracks