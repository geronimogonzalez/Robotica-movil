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

from geometry_msgs.msg import PoseStamped
import tf_transformations


class StereoTriangulationNode(Node):
    def __init__(self, left_yaml, right_yaml):
        super().__init__('stereo_triangulation_node')

        self.bridge = CvBridge()
        """ self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) """

        # --- Cargar parámetros de cámaras ---
        self.K1, self.D1, self.R1, self.P1, self.width1, self.height1 = self.load_camera_params(left_yaml)
        self.K2, self.D2, self.R2, self.P2, self.width2, self.height2 = self.load_camera_params(right_yaml)

        # --- Publishers ---
        self.pub_matches_all = self.create_publisher(Image, '/stereo/matches_all', 1)
        self.pub_matches_inliers = self.create_publisher(Image, '/stereo/matches_inliers', 1)
        self.pub_transformed = self.create_publisher(Image, '/stereo/perspective_transform', 1)
        self.pub_pointcloud = self.create_publisher(PointCloud2, '/stereo/global_pointcloud', 1)
        self.pub_disparity = self.create_publisher(Image, '/stereo/disparity', 1)
        self.pub_dense_cloud = self.create_publisher(PointCloud2, '/stereo/dense_pointcloud', 1)
        self.pub_rect_left = self.create_publisher(Image, '/cam0/image_rect', 1)
        self.pub_rect_right = self.create_publisher(Image, '/cam1/image_rect', 1)

        # --- Subscribers con sincronización ---
        sub_l = Subscriber(self, Image, '/cam0/image_raw')
        sub_r = Subscriber(self, Image, '/cam1/image_raw')
        self.ts = ApproximateTimeSynchronizer([sub_l, sub_r], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.maps_computed = False
        self.frame_count = 0

        """ # --- Ground-truth ---
        self.current_pose = np.eye(4)
        self.global_cloud = []
        self.sub_gt = self.create_subscription(
            PoseStamped, '/ground_truth/pose', self.gt_callback, 1
        ) """

        # --- Stereo matcher ---
        self.num_disparities = 64
        self.block_size = 15
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8*3*self.block_size**2,
            P2=32*3*self.block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

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
        # Q derivada de las matrices de proyección
        baseline = - (self.P2[0, 3] / self.P2[0, 0])
        cx = self.P1[0, 2]
        cy = self.P1[1, 2]
        fx = self.P1[0, 0]
        fy = self.P1[1, 1]

        self.Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, fx],
            [0, 0, -1.0 / baseline, 0]
        ])

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1[:, :3], size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2[:, :3], size, cv2.CV_32FC1)

        self.maps_computed = True
        self.get_logger().info("Mapas de rectificación y matriz Q calculados desde P1/P2.")

    """ def gt_callback(self, msg):
        q = msg.pose.orientation
        t = msg.pose.position
        R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        self.current_pose = T """

    def callback(self, msg_l, msg_r):
        imgL = self.bridge.imgmsg_to_cv2(msg_l, 'mono8')
        imgR = self.bridge.imgmsg_to_cv2(msg_r, 'mono8')
        h, w = imgL.shape

        if not self.maps_computed:
            self.compute_rectification_maps((w, h))

        rectL = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # --- Disparity map ---
        disparity = self.stereo.compute(rectL, rectR).astype(np.float32) / 16.0
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.pub_disparity.publish(self.bridge.cv2_to_imgmsg(disp_vis, 'mono8'))

        # --- Reconstrucción densa ---
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        mask = disparity > disparity.min()
        points = points_3d[mask]
        colors = cv2.cvtColor(rectL, cv2.COLOR_GRAY2BGR)[mask]

        # Filtrar valores extremos
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]

        # Publicar nube densa
        if len(points) > 0:
            header = msg_l.header
            header.frame_id = "cam0"
            cloud_points = [(float(x), float(y), float(z)) for x, y, z in points]
            msg = point_cloud2.create_cloud_xyz32(header, cloud_points)
            self.pub_dense_cloud.publish(msg)
            self.get_logger().info(f"✅ Nube densa publicada con {len(cloud_points)} puntos.")

        #Publicar imagenes rectificdas
        self.pub_rect_left.publish(self.bridge.cv2_to_imgmsg(rectL, encoding='mono8'))
        self.pub_rect_right.publish(self.bridge.cv2_to_imgmsg(rectR, encoding='mono8'))

    def triangulate_and_publish(self, kpL, kpR, matches, header):
        if len(matches) < 8:
            return
        ptsL = np.float32([kpL[m.queryIdx].pt for m in matches]).T
        ptsR = np.float32([kpR[m.trainIdx].pt for m in matches]).T
        points_4d = cv2.triangulatePoints(self.P1, self.P2, ptsL, ptsR)
        points_3d = points_4d[:3, :] / points_4d[3, :]

        points_hom = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))
        points_global = self.current_pose @ points_hom
        points_global = points_global[:3, :]

        finite_mask = np.isfinite(points_global).all(axis=0)
        for i in range(points_global.shape[1]):
            if finite_mask[i] and points_global[2, i] > 0:
                x, y, z = points_global[:, i]
                self.global_cloud.append((float(x), float(y), float(z)))

        if len(self.global_cloud) > 0:
            header.frame_id = "world"
            msg = point_cloud2.create_cloud_xyz32(header, self.global_cloud)
            self.pub_pointcloud.publish(msg)


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
