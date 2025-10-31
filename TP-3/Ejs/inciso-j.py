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
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import time
import csv
from bisect import bisect_left

class StereoTriangulationNode(Node):
    def __init__(self, left_yaml, right_yaml, gt_csv=None):
        super().__init__('stereo_triangulation_node')

        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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
        self.traj_pub = self.create_publisher(Marker, '/monocular_trajectory', 1)

        # --- Subscribers con sincronización ---
        sub_l = Subscriber(self, Image, '/cam0/image_raw')
        sub_r = Subscriber(self, Image, '/cam1/image_raw')
        self.ts = ApproximateTimeSynchronizer([sub_l, sub_r], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.maps_computed = False
        self.frame_count = 0

        # --- Ground-truth: soporte por tópico o por CSV ---
        # Si se pasa gt_csv, cargamos la lista de (time, pose4x4) y usamos get_pose_at_time.
        # Si no se pasa gt_csv, nos suscribimos al tópico /ground_truth/pose (gt_callback).
        self.gt_times = []
        self.gt_poses = []
        self.prev_gt_position = None
        self.current_pose = np.eye(4)  # usado si se suscribe al tópico
        if gt_csv is not None:
            try:
                with open(gt_csv, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # saltar encabezado
                    for row in reader:
                        if len(row) == 0:
                            continue
                        row = [c.strip() for c in row]
                        try:
                            # CSV timestamps en nanosegundos UNIX -> convertir a segundos
                            t = float(row[0]) * 1e-9
                            tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
                            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                            R = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]
                            T = np.eye(4)
                            T[:3, :3] = R
                            T[:3, 3] = [tx, ty, tz]
                            self.gt_times.append(t)
                            self.gt_poses.append(T)
                        except ValueError:
                            self.get_logger().warn(f"Línea ignorada (formato inválido): {row}")
                self.get_logger().info(f"Cargadas {len(self.gt_times)} poses desde CSV {gt_csv}")
            except Exception as e:
                self.get_logger().error(f"Error al leer CSV de ground-truth: {e}")
        else:
            # Si no pasaron CSV, subscribir al tópico tradicional
            self.sub_gt = self.create_subscription(PoseStamped, '/ground_truth/pose', self.gt_callback, 1)
            self.get_logger().info("Suscrito a /ground_truth/pose (topic)")

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

        # --- Monocular pose estimation ---
        self.pose_mono = np.eye(4)
        self.trajectory_mono = []
        self.prev_kp = None
        self.prev_des = None
        self.prev_img = None

        # --- Control de frecuencia ---
        self.last_process_time = 0
        self.process_interval = 1.0  # segundos

        self.get_logger().info("Nodo de triangulación estéreo iniciado correctamente.")

    # -------------------- Funciones auxiliares --------------------
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

    def gt_callback(self, msg):
        q = msg.pose.orientation
        t = msg.pose.position
        R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        self.current_pose = T

    def get_pose_at_time(self, t):
        """Devuelve la pose 4x4 más cercana al timestamp t (t en segundos)"""
        if not self.gt_times:
            return np.eye(4)
        idx = bisect_left(self.gt_times, t)
        if idx == 0:
            return self.gt_poses[0]
        if idx >= len(self.gt_times):
            return self.gt_poses[-1]
        # Elegir la más cercana entre idx-1 e idx
        if abs(self.gt_times[idx] - t) < abs(self.gt_times[idx-1] - t):
            return self.gt_poses[idx]
        else:
            return self.gt_poses[idx-1]

    # -------------------- Callback principal --------------------
    def callback(self, msg_l, msg_r):
        # Control de frecuencia
        now = time.time()
        if now - self.last_process_time < self.process_interval:
            return
        self.last_process_time = now

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

        # --- Dense reconstruction ---
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        mask = disparity > disparity.min()
        points = points_3d[mask]

        # Submuestreo
        if len(points) > 10000:
            idx = np.random.choice(len(points), 10000, replace=False)
            points = points[idx]

        if len(points) > 0:
            header = msg_l.header
            header.frame_id = "cam0"
            cloud_points = [(float(x), float(y), float(z)) for x, y, z in points]
            msg_cloud = point_cloud2.create_cloud_xyz32(header, cloud_points)
            self.pub_dense_cloud.publish(msg_cloud)
            self.get_logger().info(f"✅ Nube densa publicada con {len(cloud_points)} puntos.")

        # --- Monocular pose estimation ---
        kp, des = self.orb.detectAndCompute(rectL, None)
        if self.prev_kp is not None and self.prev_des is not None:
            matches = self.bf.match(des, self.prev_des)
            matches = sorted(matches, key=lambda x: x.distance)[:200]  # limitar matches
            pts_curr = np.float32([kp[m.queryIdx].pt for m in matches])
            pts_prev = np.float32([self.prev_kp[m.trainIdx].pt for m in matches])

            if len(pts_curr) >= 5:
                E, mask_E = cv2.findEssentialMat(pts_curr, pts_prev, self.K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t_unit, _ = cv2.recoverPose(E, pts_curr, pts_prev, self.K1)

                    # Escalar usando ground-truth
                    t_stamp = msg_l.header.stamp.sec + msg_l.header.stamp.nanosec * 1e-9
                    if self.gt_times:
                        pose_now = self.get_pose_at_time(t_stamp)
                        if self.prev_gt_position is not None:
                            gt_scale = np.linalg.norm(pose_now[:3, 3] - self.prev_gt_position)
                        else:
                            gt_scale = 1.0
                        self.prev_gt_position = pose_now[:3, 3].copy()
                    else:
                        #si no hay CSV: usar suscripción topic-based
                        if np.all(np.isfinite(self.current_pose)):
                            if self.prev_gt_position is not None:
                                gt_scale = np.linalg.norm(self.current_pose[:3, 3] - self.prev_gt_position)
                            else:
                                gt_scale = 1.0
                            self.prev_gt_position = self.current_pose[:3, 3].copy()
                        else:
                            gt_scale = 1.0

                    t = t_unit * gt_scale
                    T_frame = np.eye(4)
                    T_frame[:3, :3] = R
                    T_frame[:3, 3] = t.flatten()
                    self.pose_mono = self.pose_mono @ T_frame

                    # Guardar y publicar trayectoria
                    pos = self.pose_mono[:3, 3].copy()
                    self.trajectory_mono.append(pos)
                    self.publish_trajectory()

        self.prev_kp = kp
        self.prev_des = des
        self.prev_img = rectL

    # -------------------- Publicar trayectoria --------------------
    def publish_trajectory(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color = ColorRGBA()
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for p in self.trajectory_mono:
            pt = Point()
            pt.x, pt.y, pt.z = p
            marker.points.append(pt)
        self.traj_pub.publish(marker)

# -------------------- Main --------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--csv', required=False, help='Ruta al CSV de ground-truth (opcional).')
    args = parser.parse_args()

    rclpy.init()
    node = StereoTriangulationNode(args.left, args.right, args.csv)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
