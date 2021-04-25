"""Estimate head pose according to the facial landmarks"""
import os

import cv2
import numpy as np


class FacePoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self):

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Mouth left corner
            (150.0, -150.0, -125.0)  # Mouth right corner
        ]) / 4.5

        self.face_model_path = os.path.join(os.path.dirname(__file__), "config", "3d_face.dat")

        self.model_points_68 = self._get_full_model_points()
        # Camera internals
        self.focal_length = None
        self.camera_center = None
        self.camera_matrix = None
        self.size = None

        # Assuming no lens distortion
        self.dist_coefficients = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])

    def infer_camera_internals(self, img_size):
        # Camera internals
        self.size = img_size
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0,                 0,                    1]], dtype="double")

    def _get_full_model_points(self, filename=None):
        """Get all 68 3D model points from file"""
        filename = self.face_model_path if filename == None else filename
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        model_points[:, 2] *= -1
        return model_points

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)

        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        ax.scatter(x, y, z)
        ax.axis('square')
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        assert image_points.shape[0] == self.model_points.shape[0], \
            "3D points and 2D points should be of same number."
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coefficients)

        return (rotation_vector, translation_vector)

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, self.r_vec, self.t_vec) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coefficients)

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coefficients,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coefficients)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axis(self, img, R, t):
        points = np.float32(
            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)

        axis_points, _ = cv2.projectPoints(
            points, R, t, self.camera_matrix, self.dist_coefficients)

        cv2.line(img, tuple(axis_points[3].ravel()), tuple(
            axis_points[0].ravel()), (255, 0, 0), 3)
        cv2.line(img, tuple(axis_points[3].ravel()), tuple(
            axis_points[1].ravel()), (0, 255, 0), 3)
        cv2.line(img, tuple(axis_points[3].ravel()), tuple(
            axis_points[2].ravel()), (0, 0, 255), 3)

    def draw_axes(self, img, r, t):
        cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coefficients, r, t, 30)

    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])  # Nose tip
        pose_marks.append(marks[8])  # Chin
        pose_marks.append(marks[36])  # Left eye left corner
        pose_marks.append(marks[45])  # Right eye right corner
        pose_marks.append(marks[48])  # Mouth left corner
        pose_marks.append(marks[54])  # Mouth right corner
        return pose_marks
