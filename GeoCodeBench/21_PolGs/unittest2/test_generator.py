"""
Test Data Generator for rotmat2quaternion and normal2rotation functions.
"""

import torch


class TestDataGenerator:
    """Generate test data for rotmat2quaternion and normal2rotation functions."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_rotation_matrix(self, batch_size):
        """Generate random valid rotation matrices."""
        # Generate random rotation matrices using Gram-Schmidt
        matrices = []
        for _ in range(batch_size):
            # Random 3x3 matrix
            A = torch.randn(3, 3)
            # QR decomposition to get orthogonal matrix
            Q, R = torch.linalg.qr(A)
            # Ensure determinant is +1 (proper rotation)
            if torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            matrices.append(Q)
        return torch.stack(matrices, dim=0)

    def generate_normal_vectors(self, batch_size):
        """Generate random normal vectors."""
        normals = torch.randn(batch_size, 3)
        normals = torch.nn.functional.normalize(normals, dim=-1)
        return normals

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic rotmat2quaternion - small batch
        R = self.generate_rotation_matrix(5)
        test_cases.append({
            'function': 'rotmat2quaternion',
            'R': R,
            'normalize': False,
            'description': 'rotmat2quaternion: batch_size=5, normalize=False',
        })

        if num_tests > 1:
            # Test 2: rotmat2quaternion with normalization
            R = self.generate_rotation_matrix(10)
            test_cases.append({
                'function': 'rotmat2quaternion',
                'R': R,
                'normalize': True,
                'description': 'rotmat2quaternion: batch_size=10, normalize=True',
            })

        if num_tests > 2:
            # Test 3: Basic normal2rotation - small batch
            n = self.generate_normal_vectors(5)
            test_cases.append({
                'function': 'normal2rotation',
                'n': n,
                'description': 'normal2rotation: batch_size=5',
            })

        if num_tests > 3:
            # Test 4: normal2rotation - larger batch
            n = self.generate_normal_vectors(20)
            test_cases.append({
                'function': 'normal2rotation',
                'n': n,
                'description': 'normal2rotation: batch_size=20',
            })

        if num_tests > 4:
            # Test 5: rotmat2quaternion - larger batch
            R = self.generate_rotation_matrix(30)
            test_cases.append({
                'function': 'rotmat2quaternion',
                'R': R,
                'normalize': False,
                'description': 'rotmat2quaternion: batch_size=30, normalize=False',
            })

        if num_tests > 5:
            # Test 6: rotmat2quaternion - single element
            R = self.generate_rotation_matrix(1)
            test_cases.append({
                'function': 'rotmat2quaternion',
                'R': R,
                'normalize': True,
                'description': 'rotmat2quaternion: batch_size=1, normalize=True',
            })

        if num_tests > 6:
            # Test 7: normal2rotation - single element
            n = self.generate_normal_vectors(1)
            test_cases.append({
                'function': 'normal2rotation',
                'n': n,
                'description': 'normal2rotation: batch_size=1',
            })

        if num_tests > 7:
            # Test 8: rotmat2quaternion - large batch
            R = self.generate_rotation_matrix(50)
            test_cases.append({
                'function': 'rotmat2quaternion',
                'R': R,
                'normalize': False,
                'description': 'rotmat2quaternion: batch_size=50, normalize=False',
            })

        if num_tests > 8:
            # Test 9: normal2rotation - special cases
            n = torch.tensor([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.577, 0.577, 0.577],  # normalized [1,1,1]
            ])
            n = torch.nn.functional.normalize(n, dim=-1)
            test_cases.append({
                'function': 'normal2rotation',
                'n': n,
                'description': 'normal2rotation: special axis-aligned normals',
            })

        if num_tests > 9:
            # Test 10: rotmat2quaternion - identity matrices
            R = torch.eye(3).unsqueeze(0).repeat(3, 1, 1)
            test_cases.append({
                'function': 'rotmat2quaternion',
                'R': R,
                'normalize': False,
                'description': 'rotmat2quaternion: identity matrices',
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            if i % 2 == 0:
                R = self.generate_rotation_matrix(5 + i * 3)
                test_cases.append({
                    'function': 'rotmat2quaternion',
                    'R': R,
                    'normalize': i % 4 == 0,
                    'description': f'rotmat2quaternion: additional test {i+1}, batch_size={5 + i * 3}',
                })
            else:
                n = self.generate_normal_vectors(5 + i * 3)
                test_cases.append({
                    'function': 'normal2rotation',
                    'n': n,
                    'description': f'normal2rotation: additional test {i+1}, batch_size={5 + i * 3}',
                })

        return test_cases[:num_tests]
