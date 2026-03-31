"""
Reference Implementation for init_reso_scheduler
This serves as the ground truth for testing LLM-generated implementations.
"""

import math
import torch


class TrainingSchedulerMock:
    """
    Mock Training Scheduler class for testing init_reso_scheduler method.
    Contains only the necessary attributes and the method to be tested.
    """
    def __init__(self):
        # Initialize necessary attributes
        self.resolution_mode = "freq"
        self.start_significance_factor = 4
        self.max_reso_scale = 8
        self.reso_sample_num = 32  # Must be no less than 2
        self.max_densify_rate_per_step = 0.2
        self.reso_scales = None
        self.reso_level_significance = None
        self.reso_level_begin = None
        self.densify_until_iter = 15000
        self.increase_reso_until = self.densify_until_iter
        self.next_i = 2
    
    def init_reso_scheduler(self, original_images):
        if self.resolution_mode != "freq":
            print("[ INFO ] Skipped resolution scheduler initialization, the resolution mode is {}".format(self.resolution_mode))
            return

        def compute_win_significance(significance_map: torch.Tensor, scale: float):
            h, w = significance_map.shape[-2:]
            c = ((h + 1) // 2, (w + 1) // 2)
            win_size = (int(h / scale), int(w / scale))
            win_significance = significance_map[..., c[0]-win_size[0]//2: c[0]+win_size[0]//2, c[1]-win_size[1]//2: c[1]+win_size[1]//2].sum().item()
            return win_significance
        
        def scale_solver(significance_map: torch.Tensor, target_significance: float):
            L, R, T = 0., 1., 64
            for _ in range(T):
                mid = (L + R) / 2
                win_significance = compute_win_significance(significance_map, 1 / mid)
                if win_significance < target_significance:
                    L = mid
                else:
                    R = mid
            return 1 / mid
        
        print("[ INFO ] Initializing resolution scheduler...")

        self.max_reso_scale = 8
        self.next_i = 2
        scene_freq_image = None
        
        for img in original_images:
            img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
            img_fft_centered_mod = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()
            scene_freq_image = img_fft_centered_mod if scene_freq_image is None else scene_freq_image + img_fft_centered_mod

            e_total = img_fft_centered_mod.sum().item()
            e_min = e_total / self.start_significance_factor
            self.max_reso_scale = min(self.max_reso_scale, scale_solver(img_fft_centered_mod, e_min))

        modulation_func = math.log

        self.reso_scales = []
        self.reso_level_significance = []
        self.reso_level_begin = []
        scene_freq_image /= len(original_images)
        E_total = scene_freq_image.sum().item()
        E_min = compute_win_significance(scene_freq_image, self.max_reso_scale)
        self.reso_level_significance.append(E_min)
        self.reso_scales.append(self.max_reso_scale)
        self.reso_level_begin.append(0)
        for i in range(1, self.reso_sample_num - 1):
            self.reso_level_significance.append((E_total - E_min) * (i - 0) / (self.reso_sample_num-1 - 0) + E_min)
            self.reso_scales.append(scale_solver(scene_freq_image, self.reso_level_significance[-1]))
            self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
            self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
        self.reso_level_significance.append(modulation_func(E_total / E_min))
        self.reso_scales.append(1.)
        self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
        self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
        self.reso_level_begin.append(self.increase_reso_until)

