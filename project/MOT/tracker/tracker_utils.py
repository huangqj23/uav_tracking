import numpy as np
from scipy.optimize import linear_sum_assignment

def cost_matrix(tracks,detections):
    M = len(tracks)
    N = len(detections)
    cost = np.zeros(shape=(M, N))    
    for i in range(M):
        for j in range(N):                 
            dist = np.array(tracks[i].predicted[:2] - detections[j].reshape((2,1)))            
            cost[i][j] = np.sqrt(dist[0] ** 2 + dist[1] ** 2)
    
    # Normalize cost matrix                
    max_cost = np.max(cost)
    cost = cost / max_cost
    return cost
            
def track_association(tracks, detections, dist_th = 0.5):
    
    if not tracks:        
        return [], np.arange(len(detections)), []

    cost = cost_matrix(tracks, detections)            

    # Use the Hungarian algorithm to find the optimal matches 
    row_ind, col_ind = linear_sum_assignment(cost)
    paired_indices =list(zip(row_ind, col_ind))

    # Find unpaired detections and trackers
    unpaired_tracks=[d for d, _ in enumerate(tracks) if d not in row_ind]
    unpaired_detections=[t for t, _ in enumerate(detections) if t not in col_ind]

    # Filter out matches with a distance greater than the threshold
    pairs = []
    for i,j in paired_indices:
        if cost[i][j] < dist_th:
            pairs.append((i,j))
            
        else:            
            unpaired_tracks.append(i)
            unpaired_detections.append(j)
            
    return pairs, unpaired_detections, unpaired_tracks