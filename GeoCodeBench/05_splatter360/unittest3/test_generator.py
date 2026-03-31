import torch


class TestDataGenerator:
    """Generate test data for rotation matrix functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different rotation angles."""
        test_cases = []

        # Test 1: Basic angles (0°, 90°, 180°, 270°)
        test_cases.append({
            "angles": [0, 90, 180, 270],
            "description": "Basic angles: 0°, 90°, 180°, 270°",
        })

        if num_tests > 1:
            # Test 2: Common angles
            test_cases.append({
                "angles": [30, 45, 60, 120, 135, 150],
                "description": "Common angles: 30°, 45°, 60°, 120°, 135°, 150°",
            })

        if num_tests > 2:
            # Test 3: Negative angles
            test_cases.append({
                "angles": [-30, -45, -60, -90, -180],
                "description": "Negative angles: -30°, -45°, -60°, -90°, -180°",
            })

        if num_tests > 3:
            # Test 4: Small angles
            test_cases.append({
                "angles": [1, 5, 10, 15, 20, 25],
                "description": "Small angles: 1°, 5°, 10°, 15°, 20°, 25°",
            })

        if num_tests > 4:
            # Test 5: Large angles (>360°)
            test_cases.append({
                "angles": [360, 450, 540, 720, 1080],
                "description": "Large angles: 360°, 450°, 540°, 720°, 1080°",
            })

        if num_tests > 5:
            # Test 6: Arbitrary decimal angles
            test_cases.append({
                "angles": [12.5, 37.8, 67.3, 123.456, 234.567],
                "description": "Decimal angles: 12.5°, 37.8°, 67.3°, 123.456°, 234.567°",
            })

        if num_tests > 6:
            # Test 7: Very small angles
            test_cases.append({
                "angles": [0.1, 0.5, 0.9, 2.3, 3.7],
                "description": "Very small angles: 0.1°, 0.5°, 0.9°, 2.3°, 3.7°",
            })

        if num_tests > 7:
            # Test 8: Angles near 180°
            test_cases.append({
                "angles": [175, 178, 179, 181, 182, 185],
                "description": "Angles near 180°: 175°, 178°, 179°, 181°, 182°, 185°",
            })

        if num_tests > 8:
            # Test 9: Negative large angles
            test_cases.append({
                "angles": [-360, -450, -540, -720],
                "description": "Negative large angles: -360°, -450°, -540°, -720°",
            })

        if num_tests > 9:
            # Test 10: Mixed range
            test_cases.append({
                "angles": [0, -90, 180, 360, 45, -135, 270, -45, 90, -270],
                "description": "Mixed range: 0°, -90°, 180°, 360°, 45°, -135°, 270°, -45°, 90°, -270°",
            })

        # Generate additional random test cases if needed
        for i in range(num_tests - len(test_cases)):
            angles = [
                (i * 37 + 13) % 360,
                -(i * 43 + 7) % 360,
                (i * 61 + 19) % 720,
                (i * 17 + 3.14) % 180,
                -(i * 29 + 11.5) % 180,
            ]
            test_cases.append({
                "angles": angles,
                "description": f"Random test {i+1}: various angles",
            })

        return test_cases[:num_tests]
