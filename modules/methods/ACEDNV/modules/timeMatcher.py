# This function corresponds the gaze signals with the frames based on the caputered timestamps

from cmath import inf
import numpy as np
from config import VIDEO_SIZE

def timeMatcher(timestamps, mat,lbls):

    res = []
    f=1 #frame inspection pointer (tranverses through timestamp)
    g=1 #gaze inspection pointer (tranverses through mat)

    while f < len(timestamps):
        inspecF = f
        minVal = inf
        while ((f==inspecF) and (g<len(mat))):
            dist = abs(timestamps[f] - mat[g,0])
            if dist <= minVal: #distance is decreasing, then continue travers
                minVal = dist
                g = g+1
            else:
                res.append(np.insert(mat[g-1], 0,[timestamps[f],inspecF, lbls[g-1]]).tolist())
                f = f+1
        if g==len(mat): #if no gaze left to assign to frame, repear the final gaze location #TODO: Improve assigning
            res.append(np.insert(mat[g-1], 0,[timestamps[f],inspecF, lbls[g-1]]).tolist())
            f = f+1

    res = np.array(res)        
    return res[:, [3,4]], res[:, 0], res[:, 1], res[:, 2]
