#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import csv

class GroundTruthPublisher(Node):
    def __init__(self, csv_path, rate_hz=50.0):
        super().__init__('ground_truth_publisher')

        self.pub_pose = self.create_publisher(PoseStamped, '/ground_truth/pose', 10)
        self.rate_hz = rate_hz
        self.csv_path = csv_path

        self.get_logger().info(f"Cargando ground-truth desde: {csv_path}")
        self.poses = self.load_csv()
        self.get_logger().info(f"{len(self.poses)} poses cargadas del dataset.")

        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)
        self.index = 0

    def load_csv(self):
        poses = []
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Ignorar encabezados y líneas vacías
                if len(row) == 0 or row[0].startswith("#"):
                    continue
                row = [c.strip() for c in row]  # limpiar espacios
                try:
                    ts = int(row[0])
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    qw = float(row[4])
                    qx = float(row[5])
                    qy = float(row[6])
                    qz = float(row[7])
                    poses.append((ts, x, y, z, qw, qx, qy, qz))
                except ValueError:
                    self.get_logger().warn(f"Línea ignorada (formato inválido): {row}")
        return poses

    def timer_callback(self):
        if self.index >= len(self.poses):
            self.get_logger().info("🏁 Se publicaron todas las poses del ground-truth.")
            self.destroy_timer(self.timer)
            return

        ts, x, y, z, qw, qx, qy, qz = self.poses[self.index]

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()  # usar clock ROS

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = qw
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz

        self.pub_pose.publish(msg)
        self.index += 1

        self.get_logger().info(f"📡 Pose publicada #{self.index}: "
                               f"pos=({x:.2f}, {y:.2f}, {z:.2f})")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Ruta al archivo data.csv')
    parser.add_argument('--rate', type=float, default=10.0, help='Frecuencia de publicación [Hz]')
    args = parser.parse_args()

    rclpy.init()
    node = GroundTruthPublisher(args.csv, rate_hz=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
