#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer


class StereoMatcherNode(Node):
    def __init__(self, left_yaml, right_yaml):
        super().__init__('stereo_matcher_node')

        self.bridge = CvBridge()

        # --- Cargar calibraciones ---
        self.K1, self.D1, self.R1, self.P1, self.width1, self.height1 = self.load_camera_params(left_yaml)
        self.K2, self.D2, self.R2, self.P2, self.width2, self.height2 = self.load_camera_params(right_yaml)

        # --- Publicadores ---
        self.pub_rect_left = self.create_publisher(Image, '/cam0/image_rect', 1)
        self.pub_rect_right = self.create_publisher(Image, '/cam1/image_rect', 1)
        self.pub_matches_all = self.create_publisher(Image, '/stereo/matches_all', 1)
        self.pub_matches_good = self.create_publisher(Image, '/stereo/matches_good', 1)

        # --- Sincronizar cámaras ---
        sub_l = Subscriber(self, Image, '/cam0/image_raw')
        sub_r = Subscriber(self, Image, '/cam1/image_raw')
        self.ts = ApproximateTimeSynchronizer([sub_l, sub_r], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.maps_computed = False

        self.get_logger().info("Nodo de matching estéreo listo.")

    def load_camera_params(self, yaml_file):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix']['data']).reshape(3,3)
        D = np.array(data['distortion_coefficients']['data']).ravel()
        R = np.array(data['rectification_matrix']['data']).reshape(3,3)
        P = np.array(data['projection_matrix']['data']).reshape(3,4)
        width = data['image_width']
        height = data['image_height']
        return K, D, R, P, width, height

    def compute_rectification_maps(self, size):
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1[:, :3], size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2[:, :3], size, cv2.CV_32FC1)
        self.maps_computed = True
        self.get_logger().info("Mapas de rectificación listos.")

    def callback(self, msg_l, msg_r):
        """Rectifica y publica resultados."""
        imgL = self.bridge.imgmsg_to_cv2(msg_l, desired_encoding='mono8')
        imgR = self.bridge.imgmsg_to_cv2(msg_r, desired_encoding='mono8')

        h, w = imgL.shape[:2]
        if not self.maps_computed:
            self.compute_rectification_maps((w, h))

        # --- Rectificación ---
        rectL = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # --- Publicar imágenes ---
        self.pub_rect_left.publish(self.bridge.cv2_to_imgmsg(rectL, encoding='mono8'))
        self.pub_rect_right.publish(self.bridge.cv2_to_imgmsg(rectR, encoding='mono8'))
       

# --- MAIN ---
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True, help='Ruta a left.yaml')
    parser.add_argument('--right', required=True, help='Ruta a right.yaml')
    args = parser.parse_args()

    rclpy.init()
    node = StereoMatcherNode(args.left, args.right)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
