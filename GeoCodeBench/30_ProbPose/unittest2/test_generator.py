"""
Test Data Generator for compute_oks() function.
Generates test cases with keypoints, bboxes, and area.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for compute_oks() function."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def generate_keypoints(self, num_keypoints=17, x_range=(0, 100), y_range=(0, 100), visibility_prob=0.8):
        """Generate random keypoints."""
        keypoints = []
        for i in range(num_keypoints):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            v = 2 if np.random.random() < visibility_prob else 0  # 2 = visible, 0 = not visible
            keypoints.extend([x, y, v])
        return np.array(keypoints)

    def generate_bbox(self, keypoints, padding=10):
        """Generate bounding box from keypoints."""
        kp_array = np.array(keypoints).reshape(-1, 3)
        visible_mask = kp_array[:, 2] > 0
        if np.any(visible_mask):
            visible_kp = kp_array[visible_mask]
            x_min = np.min(visible_kp[:, 0]) - padding
            x_max = np.max(visible_kp[:, 0]) + padding
            y_min = np.min(visible_kp[:, 1]) - padding
            y_max = np.max(visible_kp[:, 1]) + padding
        else:
            # If no visible keypoints, create a default bbox
            x_min, x_max = 0, 100
            y_min, y_max = 0, 100

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return np.array([center_x, center_y, width / 2, height / 2])

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with visible keypoints, use_area=True, per_kpt=False
        gt_kp = self.generate_keypoints(visibility_prob=0.9)
        dt_kp = gt_kp.copy()
        dt_kp[::3] += np.random.normal(0, 2, size=len(dt_kp[::3]))  # Add small noise to x coordinates
        dt_kp[1::3] += np.random.normal(0, 2, size=len(dt_kp[1::3]))  # Add small noise to y coordinates
        bbox = self.generate_bbox(gt_kp)
        area = bbox[2] * bbox[3] * 4  # width * height
        test_cases.append({
            'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
            'dt': {'keypoints': dt_kp},
            'use_area': True,
            'per_kpt': False,
            'description': 'Basic: visible keypoints, use_area=True, per_kpt=False',
        })

        if num_tests > 1:
            # Test 2: per_kpt=True
            gt_kp = self.generate_keypoints(visibility_prob=0.8)
            dt_kp = gt_kp.copy()
            dt_kp[::3] += np.random.normal(0, 3, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, 3, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': True,
                'per_kpt': True,
                'description': 'Per-keypoint: use_area=True, per_kpt=True',
            })

        if num_tests > 2:
            # Test 3: use_area=False
            gt_kp = self.generate_keypoints(visibility_prob=0.7)
            dt_kp = gt_kp.copy()
            dt_kp[::3] += np.random.normal(0, 1.5, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, 1.5, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': False,
                'per_kpt': False,
                'description': 'No area: use_area=False, per_kpt=False',
            })

        if num_tests > 3:
            # Test 4: All visible keypoints
            gt_kp = self.generate_keypoints(visibility_prob=1.0)
            dt_kp = gt_kp.copy()
            dt_kp[::3] += np.random.normal(0, 1, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, 1, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': True,
                'per_kpt': True,
                'description': 'All visible: use_area=True, per_kpt=True',
            })

        if num_tests > 4:
            # Test 5: No visible keypoints (edge case)
            gt_kp = self.generate_keypoints(visibility_prob=0.0)
            dt_kp = gt_kp.copy()
            dt_kp[::3] += np.random.normal(0, 5, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, 5, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp, padding=20)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': True,
                'per_kpt': False,
                'description': 'No visible: use_area=True, per_kpt=False',
            })

        if num_tests > 5:
            # Test 6: Large area
            gt_kp = self.generate_keypoints(visibility_prob=0.6, x_range=(0, 500), y_range=(0, 500))
            dt_kp = gt_kp.copy()
            dt_kp[::3] += np.random.normal(0, 10, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, 10, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp, padding=50)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': True,
                'per_kpt': False,
                'description': 'Large area: use_area=True, per_kpt=False',
            })

        if num_tests > 6:
            # Test 7: Small area
            gt_kp = self.generate_keypoints(visibility_prob=0.9, x_range=(0, 20), y_range=(0, 20))
            dt_kp = gt_kp.copy()
            dt_kp[::3] += np.random.normal(0, 0.5, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, 0.5, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp, padding=2)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': True,
                'per_kpt': True,
                'description': 'Small area: use_area=True, per_kpt=True',
            })

        if num_tests > 7:
            # Test 8: Perfect match
            gt_kp = self.generate_keypoints(visibility_prob=0.8)
            dt_kp = gt_kp.copy()  # Perfect match
            bbox = self.generate_bbox(gt_kp)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': False,
                'per_kpt': True,
                'description': 'Perfect match: use_area=False, per_kpt=True',
            })

        if num_tests > 8:
            # Test 9: Large mismatch
            gt_kp = self.generate_keypoints(visibility_prob=0.7)
            dt_kp = gt_kp.copy()
            dt_kp[::3] += np.random.normal(0, 20, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, 20, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': True,
                'per_kpt': False,
                'description': 'Large mismatch: use_area=True, per_kpt=False',
            })

        if num_tests > 9:
            # Test 10: Mixed visibility
            gt_kp = self.generate_keypoints(visibility_prob=0.5)
            dt_kp = gt_kp.copy()
            dt_kp[::3] += np.random.normal(0, 2, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, 2, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp)
            area = bbox[2] * bbox[3] * 4
            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': True,
                'per_kpt': True,
                'description': 'Mixed visibility: use_area=True, per_kpt=True',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            visibility_prob = 0.3 + (i % 7) * 0.1
            use_area = (i % 2) == 0
            per_kpt = (i % 3) == 0
            gt_kp = self.generate_keypoints(visibility_prob=visibility_prob)
            dt_kp = gt_kp.copy()
            noise_scale = 1 + (i % 5)
            dt_kp[::3] += np.random.normal(0, noise_scale, size=len(dt_kp[::3]))
            dt_kp[1::3] += np.random.normal(0, noise_scale, size=len(dt_kp[1::3]))
            bbox = self.generate_bbox(gt_kp)
            area = bbox[2] * bbox[3] * 4

            test_cases.append({
                'gt': {'keypoints': gt_kp, 'bbox': bbox, 'area': area},
                'dt': {'keypoints': dt_kp},
                'use_area': use_area,
                'per_kpt': per_kpt,
                'description': f'Additional test {i+1}: use_area={use_area}, per_kpt={per_kpt}',
            })

        return test_cases[:num_tests]
