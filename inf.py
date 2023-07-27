import os
os.chdir(r"E:/")
import cv2
from detectron2dsp import Detectron2Inference


#keypoint_rcnn_R_50_FPN_3x
#mask_rcnn_R_50_FPN_3x
#faster_rcnn_R_50_FPN_3x

detector = Detectron2Inference(task='pose_estimation', model='keypoint_rcnn_R_50_FPN_3x')

image = cv2.imread("person.jpg")

# Perform object detection
boxes, classes, scores, output_image = detector.infer(image)
#output_image = cv2.resize(output_image , (1200,800))

# Display output image
cv2.imshow("Output", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()












