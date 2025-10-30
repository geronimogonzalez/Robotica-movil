#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.msg import PointField
fields = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
] #importo todos los campos que voy a usar en la nube de puntos
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
import tf_transformations
import time
import csv
from bisect import bisect_left




class StereoTriangulationNode(Node):
    def __init__(self, left_yaml, right_yaml, gt_csv=None):
        super().__init__('stereo_triangulation_node')

        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.frame_count = 0   # contador de frames

        # --- Cargar parámetros de cámaras ---
        self.K1, self.D1, self.R1, self.P1, self.width1, self.height1 = self.load_camera_params(left_yaml)
        self.K2, self.D2, self.R2, self.P2, self.width2, self.height2 = self.load_camera_params(right_yaml)

        # --- Publishers ---
        self.pub_matches_all = self.create_publisher(Image, '/stereo/matches_all', 1)
        self.pub_matches_inliers = self.create_publisher(Image, '/stereo/matches_inliers', 1)
        self.pub_transformed = self.create_publisher(Image, '/stereo/perspective_transform', 1)
        #self.pub_pointcloud = self.create_publisher(PointCloud2, '/stereo/global_pointcloud', 1) #se comenta porque es de la otra triangulación
        self.pub_disparity = self.create_publisher(Image, '/stereo/disparity', 1)
        self.pub_dense_cloud = self.create_publisher(PointCloud2, '/stereo/dense_pointcloud', 1)
        self.pub_dense_cloudMAP = self.create_publisher(PointCloud2, '/stereo/dense_pointcloudMAP', 1)
        self.pub_rect_left = self.create_publisher(Image, '/cam0/image_rect', 1)
        self.pub_rect_right = self.create_publisher(Image, '/cam1/image_rect', 1)

        # --- Subscribers con sincronización ---
        sub_l = Subscriber(self, Image, '/cam0/image_raw')
        sub_r = Subscriber(self, Image, '/cam1/image_raw')
        self.ts = ApproximateTimeSynchronizer([sub_l, sub_r], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.callback)

        self.maps_computed = False

        # Ground-truth CSV
        self.gt_times = []
        self.gt_poses = []
        if gt_csv is not None:
            with open(gt_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) == 0:
                        continue
                    row = [c.strip() for c in row]
                    try:
                        t = float(row[0]) * 1e-9 #convierto a segundos porque el csv está en nanosegundos UNIX
                        self.gt_times.append(t)
                        # Convertir pose a matriz 4x4
                        tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
                        qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                        R = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = [tx, ty, tz]
                        self.gt_poses.append(T)
                    except ValueError:
                        self.get_logger().warn(f"Línea ignorada (formato inválido): {row}")
                     
        self.global_cloud = []
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

        # --- Control de frecuencia ---
        self.last_process_time = 0.0
        self.process_interval = 0.5  # segundos

        self.get_logger().info("Nodo de triangulación estéreo iniciado correctamente.")
      
    def get_pose_at_time(self, t): #Devuelve la pose 4x4 más cercana al timestamp t
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
        # --- Control de frecuencia --- par no procesar todos los frames, genera ruido. (esto se llama throttling)
        now = time.time()
        if now - self.last_process_time < self.process_interval:
            return
        self.last_process_time = now

        self.frame_count += 1

        imgL = self.bridge.imgmsg_to_cv2(msg_l, 'mono8')
        imgR = self.bridge.imgmsg_to_cv2(msg_r, 'mono8')
        h, w = imgL.shape

        if not self.maps_computed:
            self.compute_rectification_maps((w, h))

        rectL = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)
        #Publicar imagenes rectificdas
        self.pub_rect_left.publish(self.bridge.cv2_to_imgmsg(rectL, encoding='mono8'))
        self.pub_rect_right.publish(self.bridge.cv2_to_imgmsg(rectR, encoding='mono8'))

        # --- Disparity map ---
        disparity = self.stereo.compute(rectL, rectR).astype(np.float32) / 16.0
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.pub_disparity.publish(self.bridge.cv2_to_imgmsg(disp_vis, 'mono8'))

        # --- Reconstrucción densa ---
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        mask = (disparity > disparity.min()) & np.isfinite(disparity) # filtro valores invalidos, puntos sin correspondencia válida, z negativos y puntos con z mayor a 3 m
        points = points_3d[mask]
        if rectL.ndim == 2:
            colors = cv2.cvtColor(rectL, cv2.COLOR_GRAY2BGR)[mask]
        else:
            colors = rectL[mask] #si la imagen ya tiene 3 canales tomo la imagen completa, pero solo voy a usar los cmapos de color

        """ #Submuestreo para reducir cantidad de puntos
        if len(points) > 20000:
            idx = np.random.choice(len(points), 20000, replace=False)
            points = points[idx]
            colors = colors[idx] """

        # Publicar nube densa y nube global
            #gnero y publico nube densa (local)
        if len(points) > 0:
            header = msg_l.header
            header.frame_id = "cam0"
            cloud_points = [(float(x), float(y), float(z), float(b), float(g), float(r)) for (x, y, z), (b, g, r) in zip(points, colors)]
            msg = point_cloud2.create_cloud(header,fields, cloud_points)
            self.pub_dense_cloud.publish(msg)
            self.get_logger().info(f"Nube densa publicada con {len(cloud_points)} puntos.")
         #genero y publico nube densa (global)
            points_hom = np.hstack((points, np.ones((points.shape[0], 1)))) #convierto a cord homogeneas
            pose_now = self.get_pose_at_time(msg_l.header.stamp.sec + msg_l.header.stamp.nanosec * 1e-9)
            if not np.all(np.isfinite(pose_now)):
               self.get_logger().warn("Pose inválida detectada")
            points_global = (pose_now @ points_hom.T).T[:, :3] #trasnformo a cord globales con pose de gt y tomo solo x, y, z de los puntos

            #genero nube gloabl con la acumulación de todos los puntos de cada frame transformados al sistema world
            for (x, y, z), (b, g, r) in zip(points_global, colors):
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and z > 0:
                    self.global_cloud.append((float(x), float(y), float(z), float(b), float(g), float(r)))
            #verifico que la nube no esté vacía y la publico (me parece que se podría hacer de una forma más optimizada, es como que cada vez publico un mensaje mas grande)
            #tamnbién publico solo cada 15 frames para no saturar la memoria
            if len(self.global_cloud) > 0 and self.frame_count % 15 == 0:
                header.frame_id = "world"
                msg = point_cloud2.create_cloud(header, fields, self.global_cloud)
                self.pub_dense_cloudMAP.publish(msg)
                self.get_logger().info(f"Nube densa global publicada con {len(points_global)} puntos nuevos.")

        #lo que está acá abajo es la forma anterior de triangulación, ahora usamos Stereo Matcher (SGBM) que creo que aprovecha geom. epipolar de cámaras rectificadas, ademas hace nube densa no con features de BF.
        """ # --- ORB matches y nube global ---
        kpL, desL = self.orb.detectAndCompute(rectL, None)
        kpR, desR = self.orb.detectAndCompute(rectR, None)
        if desL is not None and desR is not None:
            matches = self.bf.match(desL, desR)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 10:
                ptsL_all = np.float32([kpL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                ptsR_all = np.float32([kpR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(ptsL_all, ptsR_all, cv2.RANSAC, 5.0) #RANSAC para homografía

                if H is not None:
                    mask = mask.ravel().astype(bool)
                    inlier_matches = [m for m, valid in zip(matches, mask) if valid]
                    self.triangulate_and_publish(kpL, kpR, inlier_matches, msg_l.header)

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
            self.get_logger().info(f"🌐 Nube global publicada con {len(self.global_cloud)} puntos.") """

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()

    rclpy.init()
    node = StereoTriangulationNode(args.left, args.right, args.csv)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# Tambien hay que correr el publicador de gt <python3 camera_info_pub ground_truth_publisher --csv ruta/GT_data.csv>
#para ejecutar este programa desde CLI <python3 inciso-f.py   --left ruta/left.yaml     --right ruta/right.yaml>
# publicar tf de word <ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 world map> o cambiar header.frame_id = "world"
