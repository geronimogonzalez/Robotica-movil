#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer


class StereoFeatureNode(Node):
    def __init__(self, left_yaml, right_yaml):
        super().__init__('stereo_feature_node')

        self.bridge = CvBridge()

        # --- Cargar calibración ---
        self.K1, self.D1, self.R1, self.P1, self.width1, self.height1 = self.load_camera_params(left_yaml)
        self.K2, self.D2, self.R2, self.P2, self.width2, self.height2 = self.load_camera_params(right_yaml)

        # --- Publicadores ---
        self.pub_left = self.create_publisher(Image, '/cam0/image_rect', 1)
        self.pub_right = self.create_publisher(Image, '/cam1/image_rect', 1)
        self.pub_left_kp = self.create_publisher(Image, '/cam0/image_kp', 1)
        self.pub_right_kp = self.create_publisher(Image, '/cam1/image_kp', 1)

        # --- Suscriptores sincronizados ---
        sub_l = Subscriber(self, Image, '/cam0/image_raw')
        sub_r = Subscriber(self, Image, '/cam1/image_raw')
        self.ts = ApproximateTimeSynchronizer([sub_l, sub_r], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        # --- Detector ORB ---
        self.orb = cv2.ORB_create(nfeatures=500)

        self.maps_computed = False
        self.frame_count = 0
        self.get_logger().info("Nodo de extracción de features listo.")

    def load_camera_params(self, yaml_file):
        """Carga los parámetros intrínsecos y de proyección desde un archivo YAML."""
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
        """Calcula los mapas de rectificación para ambas cámaras."""
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1[:, :3], size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2[:, :3], size, cv2.CV_32FC1)
        self.maps_computed = True
        self.get_logger().info("Mapas de rectificación calculados.")

    def callback(self, msg_l, msg_r):
        """Procesa imágenes sincronizadas, rectifica y extrae features."""
        imgL = self.bridge.imgmsg_to_cv2(msg_l, desired_encoding='mono8')
        imgR = self.bridge.imgmsg_to_cv2(msg_r, desired_encoding='mono8')

        h, w = imgL.shape[:2]
        if not self.maps_computed:
            self.compute_rectification_maps((w, h))

        # --- Rectificación ---
        rectL = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # --- Extracción de keypoints y descriptores ---
        kpL, desL = self.orb.detectAndCompute(rectL, None)
        kpR, desR = self.orb.detectAndCompute(rectR, None)

        # --- Dibujar keypoints ---
        img_kpL = cv2.drawKeypoints(rectL, kpL, None, color=(0, 255, 0), flags=0)
        img_kpR = cv2.drawKeypoints(rectR, kpR, None, color=(0, 255, 0), flags=0)

        # --- Publicar imágenes ---
        self.pub_left.publish(self.bridge.cv2_to_imgmsg(rectL, encoding='mono8'))
        self.pub_right.publish(self.bridge.cv2_to_imgmsg(rectR, encoding='mono8'))
        self.pub_left_kp.publish(self.bridge.cv2_to_imgmsg(img_kpL, encoding='bgr8'))
        self.pub_right_kp.publish(self.bridge.cv2_to_imgmsg(img_kpR, encoding='bgr8'))

        self.frame_count += 1
        self.get_logger().info(f"Frame {self.frame_count}: {len(kpL)} / {len(kpR)} keypoints detectados.")


# --- MAIN ---
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True, help='Ruta a left.yaml')
    parser.add_argument('--right', required=True, help='Ruta a right.yaml')
    args = parser.parse_args()

    rclpy.init()
    node = StereoFeatureNode(args.left, args.right)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
