#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer


class StereoTriangulationNode(Node):
    def __init__(self, left_yaml, right_yaml):
        super().__init__('stereo_triangulation_node')

        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.K1, self.D1, self.R1, self.P1, self.width1, self.height1 = self.load_camera_params(left_yaml)
        self.K2, self.D2, self.R2, self.P2, self.width2, self.height2 = self.load_camera_params(right_yaml)

        self.pub_matches_all = self.create_publisher(Image, '/stereo/matches_all', 1)
        self.pub_matches_inliers = self.create_publisher(Image, '/stereo/matches_inliers', 1)
        self.pub_transformed = self.create_publisher(Image, '/stereo/perspective_transform', 1)
        self.pub_pointcloud = self.create_publisher(PointCloud2, '/stereo/pointcloud', 1)

        sub_l = Subscriber(self, Image, '/cam0/image_raw')
        sub_r = Subscriber(self, Image, '/cam1/image_raw')
        self.ts = ApproximateTimeSynchronizer([sub_l, sub_r], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.maps_computed = False
        self.frame_count = 0

        self.get_logger().info("Nodo de triangulación estéreo iniciado correctamente.")

    def load_camera_params(self, yaml_file):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix']['data']).reshape(3, 3)
        D = np.array(data['distortion_coefficients']['data']).ravel()
        R = np.array(data['rectification_matrix']['data']).reshape(3, 3)
        P = np.array(data['projection_matrix']['data']).reshape(3, 4)
        return K, D, R, P, data['image_width'], data['image_height']

    def compute_rectification_maps(self, size):
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1[:, :3], size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2[:, :3], size, cv2.CV_32FC1)
        self.maps_computed = True
        self.get_logger().info("Mapas de rectificación listos.")

    def callback(self, msg_l, msg_r):
        imgL = self.bridge.imgmsg_to_cv2(msg_l, 'mono8')
        imgR = self.bridge.imgmsg_to_cv2(msg_r, 'mono8')
        h, w = imgL.shape

        if not self.maps_computed:
            self.compute_rectification_maps((w, h))

        rectL = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)

        kpL, desL = self.orb.detectAndCompute(rectL, None)
        kpR, desR = self.orb.detectAndCompute(rectR, None)
        if desL is None or desR is None:
            return

        matches = self.bf.match(desL, desR)
        matches = sorted(matches, key=lambda x: x.distance)

        img_matches_all = cv2.drawMatches(rectL, kpL, rectR, kpR, matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.pub_matches_all.publish(self.bridge.cv2_to_imgmsg(img_matches_all, 'bgr8'))

        # --- Filtrado por RANSAC ---
        if len(matches) > 10:
            ptsL_all = np.float32([kpL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            ptsR_all = np.float32([kpR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(ptsL_all, ptsR_all, cv2.RANSAC, 5.0)

            if H is not None:
                mask = mask.ravel().astype(bool)
                inlier_matches = [m for m, valid in zip(matches, mask) if valid]

                img_inliers = cv2.drawMatches(
                    rectL, kpL, rectR, kpR, inlier_matches, None,
                    matchColor=(0,255,0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.pub_matches_inliers.publish(self.bridge.cv2_to_imgmsg(img_inliers, 'bgr8'))

                # --- Puntos proyectados ---
                rectR_color = cv2.cvtColor(rectR, cv2.COLOR_GRAY2BGR)
                ptsL_transformed = cv2.perspectiveTransform(ptsL_all, H)
                for p in ptsL_transformed:
                    cv2.circle(rectR_color, tuple(np.int32(p[0])), 3, (0,0,255), -1)
                self.pub_transformed.publish(self.bridge.cv2_to_imgmsg(rectR_color, 'bgr8'))

                # --- Triangular solo inliers ---
                self.triangulate_and_publish(kpL, kpR, inlier_matches, msg_l.header)

    def triangulate_and_publish(self, kpL, kpR, matches, header):
        if len(matches) < 8:
            self.get_logger().warn("No hay suficientes matches para triangulación.")
            return

        ptsL = np.float32([kpL[m.queryIdx].pt for m in matches]).T
        ptsR = np.float32([kpR[m.trainIdx].pt for m in matches]).T

        points_4d = cv2.triangulatePoints(self.P1, self.P2, ptsL, ptsR)
        points_3d = points_4d[:3, :] / points_4d[3, :]

        # --- Debug ---
        finite_mask = np.isfinite(points_3d).all(axis=0)
        valid_points = np.sum(finite_mask)
        self.get_logger().info(f"Puntos triangulados: {points_3d.shape[1]}, válidos: {valid_points}")

        cloud_points = []
        for i in range(points_3d.shape[1]):
            x, y, z = points_3d[:, i]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and z > 0:
                cloud_points.append((float(x), float(y), float(z)))

        if len(cloud_points) == 0:
            self.get_logger().warn("No se generaron puntos válidos para la nube.")
            return

        header.frame_id = "stereo_camera"
        msg = point_cloud2.create_cloud_xyz32(header, cloud_points)
        self.pub_pointcloud.publish(msg)
        self.get_logger().info(f"✅ Publicada nube con {len(cloud_points)} puntos 3D.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    args = parser.parse_args()

    rclpy.init()
    node = StereoTriangulationNode(args.left, args.right)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
