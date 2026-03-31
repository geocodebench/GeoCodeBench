"""
Test Data Generator for LossGuidance.__call__() method.
"""

import torch

from reference_implementation import MockLossGuidance as RefMockLossGuidance, mock_lpips_fn


class TestDataGenerator:
    """Generate test data for __call__() method."""

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case without guidance masks
        H, W = 64, 64
        diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.5  # [-1, 1]
        guidance_images = torch.rand(1, 3, H, W, device=self.device)  # [0, 1]

        loss_guidance = RefMockLossGuidance(
            w_recon=0.5,
            mean_loss=False,
            ssim_guidance=False,
            lpips_guidance=False,
            guidance_images=guidance_images,
            guidance_masks=None,
            lpips_fn=mock_lpips_fn,
            verbose=False
        )

        test_cases.append({
            'loss_guidance': loss_guidance,
            'diffused_images': diffused_images,
            'ddim_index': 0,
            'batch_idx_start': 0,
            'batch_idx_end': 1,
            'description': f'Basic: no masks, H={H}, W={W}',
        })

        if num_tests > 1:
            # Test 2: With guidance masks
            H, W = 64, 64
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.5
            guidance_images = torch.rand(2, 3, H, W, device=self.device)
            guidance_masks = torch.rand(2, 1, H, W, device=self.device) > 0.5
            guidance_masks = guidance_masks.float()

            loss_guidance = RefMockLossGuidance(
                w_recon=1.0,
                mean_loss=False,
                ssim_guidance=False,
                lpips_guidance=False,
                guidance_images=guidance_images,
                guidance_masks=guidance_masks,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 5,
                'batch_idx_start': 1,
                'batch_idx_end': 2,
                'description': f'With masks: H={H}, W={W}, batch_idx 1:2',
            })

        if num_tests > 2:
            # Test 3: With SSIM guidance
            H, W = 128, 128
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.3
            guidance_images = torch.rand(1, 3, H, W, device=self.device)

            loss_guidance = RefMockLossGuidance(
                w_recon=0.8,
                mean_loss=False,
                ssim_guidance=True,
                lpips_guidance=False,
                guidance_images=guidance_images,
                guidance_masks=None,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 10,
                'batch_idx_start': 0,
                'batch_idx_end': 1,
                'description': f'SSIM guidance: H={H}, W={W}',
            })

        if num_tests > 3:
            # Test 4: With LPIPS guidance
            H, W = 64, 64
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.5
            guidance_images = torch.rand(3, 3, H, W, device=self.device)
            guidance_masks = torch.ones(3, 1, H, W, device=self.device)

            loss_guidance = RefMockLossGuidance(
                w_recon=0.5,
                mean_loss=False,
                ssim_guidance=False,
                lpips_guidance=True,
                guidance_images=guidance_images,
                guidance_masks=guidance_masks,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 15,
                'batch_idx_start': 1,
                'batch_idx_end': 2,
                'description': f'LPIPS guidance: H={H}, W={W}',
            })

        if num_tests > 4:
            # Test 5: All guidance types
            H, W = 96, 96
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.4
            guidance_images = torch.rand(2, 3, H, W, device=self.device)
            guidance_masks = torch.rand(2, 1, H, W, device=self.device) > 0.3
            guidance_masks = guidance_masks.float()

            loss_guidance = RefMockLossGuidance(
                w_recon=1.0,
                mean_loss=False,
                ssim_guidance=True,
                lpips_guidance=True,
                guidance_images=guidance_images,
                guidance_masks=guidance_masks,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 20,
                'batch_idx_start': 0,
                'batch_idx_end': 1,
                'description': f'All guidance: SSIM+LPIPS, H={H}, W={W}',
            })

        if num_tests > 5:
            # Test 6: mean_loss=True
            H, W = 64, 64
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.5
            guidance_images = torch.rand(1, 3, H, W, device=self.device)

            loss_guidance = RefMockLossGuidance(
                w_recon=0.5,
                mean_loss=True,
                ssim_guidance=False,
                lpips_guidance=False,
                guidance_images=guidance_images,
                guidance_masks=None,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 0,
                'batch_idx_start': 0,
                'batch_idx_end': 1,
                'description': f'mean_loss=True: H={H}, W={W}',
            })

        if num_tests > 6:
            # Test 7: Larger resolution
            H, W = 256, 256
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.3
            guidance_images = torch.rand(1, 3, H, W, device=self.device)

            loss_guidance = RefMockLossGuidance(
                w_recon=0.7,
                mean_loss=False,
                ssim_guidance=False,
                lpips_guidance=False,
                guidance_images=guidance_images,
                guidance_masks=None,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 0,
                'batch_idx_start': 0,
                'batch_idx_end': 1,
                'description': f'Large resolution: H={H}, W={W}',
            })

        if num_tests > 7:
            # Test 8: Different w_recon
            H, W = 64, 64
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.5
            guidance_images = torch.rand(1, 3, H, W, device=self.device)

            loss_guidance = RefMockLossGuidance(
                w_recon=2.0,
                mean_loss=False,
                ssim_guidance=False,
                lpips_guidance=False,
                guidance_images=guidance_images,
                guidance_masks=None,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 0,
                'batch_idx_start': 0,
                'batch_idx_end': 1,
                'description': f'High w_recon=2.0: H={H}, W={W}',
            })

        if num_tests > 8:
            # Test 9: Sparse masks
            H, W = 64, 64
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.5
            guidance_images = torch.rand(2, 3, H, W, device=self.device)
            guidance_masks = torch.rand(2, 1, H, W, device=self.device) > 0.9  # Very sparse
            guidance_masks = guidance_masks.float()

            loss_guidance = RefMockLossGuidance(
                w_recon=1.0,
                mean_loss=False,
                ssim_guidance=False,
                lpips_guidance=False,
                guidance_images=guidance_images,
                guidance_masks=guidance_masks,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 0,
                'batch_idx_start': 1,
                'batch_idx_end': 2,
                'description': f'Sparse masks: H={H}, W={W}',
            })

        if num_tests > 9:
            # Test 10: Edge case with very small values
            H, W = 64, 64
            diffused_images = torch.randn(3, 1, H, W, device=self.device) * 0.01
            guidance_images = torch.rand(1, 3, H, W, device=self.device) * 0.1

            loss_guidance = RefMockLossGuidance(
                w_recon=0.5,
                mean_loss=False,
                ssim_guidance=True,
                lpips_guidance=True,
                guidance_images=guidance_images,
                guidance_masks=None,
                lpips_fn=mock_lpips_fn,
                verbose=False
            )

            test_cases.append({
                'loss_guidance': loss_guidance,
                'diffused_images': diffused_images,
                'ddim_index': 0,
                'batch_idx_start': 0,
                'batch_idx_end': 1,
                'description': f'Small values: H={H}, W={W}',
            })

        return test_cases[:num_tests]
