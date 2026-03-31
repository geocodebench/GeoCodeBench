"""
Test Data Generator for spherical_unwrap_opacity function
Generates various test cases with different characteristics.
"""

import numpy as np


class TestDataGenerator:
    """Generate test data for spherical unwrapping tests."""
    
    def __init__(self, seed=42):
        """
        Initialize the test data generator.
        
        Args:
            seed (int): Random seed for reproducibility.
        """
        self.seed = seed
        np.random.seed(seed)
    
    def generate_uniform_sphere(self, n_points=1000, radius=1.0):
        """
        Generate points uniformly distributed on a sphere surface.
        
        Args:
            n_points (int): Number of points to generate.
            radius (float): Radius of the sphere.
            
        Returns:
            tuple: (points, opacity) arrays
        """
        # Use Fibonacci sphere algorithm for uniform distribution
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        
        points = np.column_stack([x, y, z])
        opacity = np.random.rand(n_points)
        
        return points, opacity
    
    def generate_random_cloud(self, n_points=1000, scale=1.0):
        """
        Generate random point cloud in a cubic volume.
        
        Args:
            n_points (int): Number of points to generate.
            scale (float): Scale of the cubic volume.
            
        Returns:
            tuple: (points, opacity) arrays
        """
        points = np.random.randn(n_points, 3) * scale
        opacity = np.random.rand(n_points)
        
        return points, opacity
    
    def generate_clustered_points(self, n_clusters=10, points_per_cluster=100, spread=0.2):
        """
        Generate clustered points to test collision handling.
        
        Args:
            n_clusters (int): Number of clusters.
            points_per_cluster (int): Points per cluster.
            spread (float): Spread of points within each cluster.
            
        Returns:
            tuple: (points, opacity) arrays
        """
        all_points = []
        all_opacity = []
        
        for _ in range(n_clusters):
            # Random cluster center on sphere
            theta = np.random.rand() * 2 * np.pi
            phi = np.random.rand() * np.pi
            center = np.array([
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi)
            ])
            
            # Generate points around center
            cluster_points = center + np.random.randn(points_per_cluster, 3) * spread
            cluster_opacity = np.random.rand(points_per_cluster)
            
            all_points.append(cluster_points)
            all_opacity.append(cluster_opacity)
        
        points = np.vstack(all_points)
        opacity = np.concatenate(all_opacity)
        
        return points, opacity
    
    def generate_edge_case_simple(self):
        """
        Generate simple edge case with known behavior.
        
        Returns:
            tuple: (points, opacity) arrays
        """
        # Simple case: 4 points at cardinal directions
        points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0]
        ])
        opacity = np.array([0.5, 0.7, 0.9, 0.3])
        
        return points, opacity
    
    def generate_collision_test(self, n_points=50):
        """
        Generate points that will map to same pixel to test opacity handling.
        
        Args:
            n_points (int): Number of points to generate.
            
        Returns:
            tuple: (points, opacity) arrays
        """
        # Generate points very close to each other (will map to same pixel)
        base_point = np.array([1.0, 0.0, 0.0])
        points = base_point + np.random.randn(n_points, 3) * 0.001
        
        # Create opacity values with clear ordering
        opacity = np.linspace(0.1, 1.0, n_points)
        
        return points, opacity
    
    def generate_test_suite(self, num_tests=5):
        """
        Generate a complete test suite with various test cases.
        
        Args:
            num_tests (int): Number of test cases to generate.
            
        Returns:
            list: List of test cases, each containing (points, opacity, height, width, description)
        """
        test_cases = []
        
        # Test 1: Simple edge case (always included)
        points, opacity = self.generate_edge_case_simple()
        test_cases.append({
            'points': points,
            'opacity': opacity,
            'height': 64,
            'width': 64,
            'description': 'Simple edge case with 4 cardinal points'
        })
        
        if num_tests > 1:
            # Test 2: Uniform sphere distribution
            points, opacity = self.generate_uniform_sphere(n_points=500)
            test_cases.append({
                'points': points,
                'opacity': opacity,
                'height': 128,
                'width': 128,
                'description': 'Uniform distribution on sphere (500 points)'
            })
        
        if num_tests > 2:
            # Test 3: Random cloud
            points, opacity = self.generate_random_cloud(n_points=1000)
            test_cases.append({
                'points': points,
                'opacity': opacity,
                'height': 256,
                'width': 256,
                'description': 'Random point cloud (1000 points)'
            })
        
        if num_tests > 3:
            # Test 4: Clustered points
            points, opacity = self.generate_clustered_points(n_clusters=20, points_per_cluster=50)
            test_cases.append({
                'points': points,
                'opacity': opacity,
                'height': 256,
                'width': 256,
                'description': 'Clustered points (20 clusters, 50 points each)'
            })
        
        if num_tests > 4:
            # Test 5: Collision test
            points, opacity = self.generate_collision_test(n_points=100)
            test_cases.append({
                'points': points,
                'opacity': opacity,
                'height': 128,
                'width': 128,
                'description': 'Collision test (multiple points mapping to same pixel)'
            })
        
        # Generate additional random tests if needed
        for i in range(num_tests - len(test_cases)):
            test_type = i % 3
            if test_type == 0:
                points, opacity = self.generate_uniform_sphere(n_points=500 + i * 100)
                desc = f'Additional uniform sphere test {i+1}'
            elif test_type == 1:
                points, opacity = self.generate_random_cloud(n_points=800 + i * 200)
                desc = f'Additional random cloud test {i+1}'
            else:
                points, opacity = self.generate_clustered_points(
                    n_clusters=10 + i * 5, 
                    points_per_cluster=50
                )
                desc = f'Additional clustered test {i+1}'
            
            test_cases.append({
                'points': points,
                'opacity': opacity,
                'height': 256,
                'width': 256,
                'description': desc
            })
        
        return test_cases[:num_tests]

