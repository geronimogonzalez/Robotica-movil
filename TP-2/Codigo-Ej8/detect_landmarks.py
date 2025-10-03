#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import numpy as np

class LandmarkDetector(Node):
    def __init__(self):
        super().__init__('landmark_detector')

        # Parámetros
        self.max_range = 10.0   # máximo radio a considerar (m)
        self.cluster_threshold = 0.2  # distancia máxima entre puntos para agrupar

        # Suscripción al LIDAR
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )

        # Publicador de markers
        self.pub_marker = self.create_publisher(Marker, '/landmark', 10)

        self.get_logger().info("Nodo Landmark Detector inicializado")

    def laser_callback(self, msg: LaserScan):
        # Calcular ángulos y rangos válidos
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges)
        valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges < self.max_range)
        if not np.any(valid):
            return

        # Coordenadas cartesianas
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        ang_valid = angles[valid]

        # -------------------------------
        # Agrupar puntos contiguos en clusters
        # -------------------------------
        clusters = []
        current_cluster = [(x[0], y[0], ang_valid[0])]

        for i in range(1, len(x)):
            dx, dy = x[i] - x[i-1], y[i] - y[i-1]
            dist = np.hypot(dx, dy)
            if dist < self.cluster_threshold:
                current_cluster.append((x[i], y[i], ang_valid[i]))
            else:
                if len(current_cluster) > 2:  # guardar cluster si tiene puntos suficientes
                    clusters.append(current_cluster)
                current_cluster = [(x[i], y[i], ang_valid[i])]
        if len(current_cluster) > 2:
            clusters.append(current_cluster)

        # -------------------------------
        # Publicar un marker por cluster
        # -------------------------------
        for i, cluster in enumerate(clusters):
            pts = np.array(cluster)
            
            # Centroide preliminar
            cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
            mean_angle = np.mean(pts[:, 2])

            # Estimar radio como mitad de la distancia entre primer y último punto
            x_first, y_first = pts[0, 0], pts[0, 1]
            x_last, y_last   = pts[-1, 0], pts[-1, 1]
            diameter_est = np.sqrt((x_last - x_first)**2 + (y_last - y_first)**2)
            r_cylinder = diameter_est / 2.0

            # Corregir el centroide (desplazar hacia afuera del cilindro)
            cx += r_cylinder * np.cos(mean_angle)
            cy += r_cylinder * np.sin(mean_angle)

            # Crear marker
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cylinder"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(cx)
            marker.pose.position.y = float(cy)
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2 * r_cylinder
            marker.scale.y = 2 * r_cylinder
            marker.scale.z = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            self.pub_marker.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
