import rospy
import cv2
from ultralytics import YOLO
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from demo_yolo.srv import GetDepthComand,GetDepthComandRequest
#import torch
#print(torch.backends.mps.is_available())


model = YOLO("train15/weights/last.pt")

class ObjectDetection(object):

    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("Chicken_detect", anonymous=True)
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.update_frame_callback)
        rospy.wait_for_message("/camera/rgb/image_raw", Image)
        self.get_depth_command= rospy.ServiceProxy('/vision/get_depth',GetDepthComand)

    def update_frame_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8") 

    def main(self):        

        while not rospy.is_shutdown():
            frame = self.image
            results = model(frame)#, device="mps")
            result = results[0]
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype = "int")
            classes = np.array(result.boxes.cls.cpu(), dtype = "int")
            keypoints = np.array(result.keypoints.xy.cpu(), dtype = "int")

            for keypoint, cls, bbox in zip(keypoints, classes, bboxes):

                (xb, yb, xb2, yb2) = bbox
                (k1,k2,k3) = keypoint

                xk1,yk1 = k1
                xk2,yk2 = k2
                xk3,yk3 = k3

                print("x = ",xb,"y = ",yb,"x2 = ",xb2,"y2 = ",yb2)
                print("k1 =", k1,"k2 =", k2,"k3 =", k3)

                cv2.putText(frame, str(cls), (xb,yb-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

                cv2.rectangle(frame, (xb,yb), (xb2,yb2), (0,0,255), 2)

                cv2.circle(frame, k1, 0, (255,0,0), 10)
                cv2.circle(frame, k2, 0, (255,100,0), 10)
                cv2.circle(frame, k3, 0, (255,200,0), 10)
                request = GetDepthComandRequest()
                request.position = [xk1, yk1] # สร้างอินสแตนซ์ของ GetDepthCommandRequest
                response = self.get_depth_command(request)  # เรียกใช้บริการ get_depth_command
        
            cv2.imshow("img", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break


if __name__ == "__main__":
    obj = ObjectDetection()
    obj.main()