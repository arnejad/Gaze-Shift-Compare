import cv2

cap = cv2.VideoCapture("/media/ash/Expansion/data/Saccade-Detection-Methods/p5/world.mp4")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print("Total frames:", frame_count)