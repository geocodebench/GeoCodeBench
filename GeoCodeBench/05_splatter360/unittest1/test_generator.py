import torch


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cpu"),
):
    """Generate attention mask for shifted window attention."""
    from reference_implementation import split_feature

    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


class TestDataGenerator:
    """Generate test data for multi_head_split_window_attention function."""

    def __init__(self, seed=42):
        self.seed = seed
        torch.manual_seed(seed)

    def generate_test_suite(self, num_tests=5):
        """Generate test cases with different configurations."""
        test_cases = []

        # Test 1: Basic case with single head, no shift
        b, h, w, c = 2, 8, 8, 64
        num_splits = 2
        num_head = 1
        test_cases.append({
            "q": torch.randn(b, h * w, c),
            "k": torch.randn(b, h * w, c),
            "v": torch.randn(b, h * w, c),
            "num_splits": num_splits,
            "with_shift": False,
            "h": h,
            "w": w,
            "attn_mask": None,
            "num_head": num_head,
            "description": f"Basic: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}, no shift",
        })

        if num_tests > 1:
            # Test 2: Multi-head attention
            b, h, w, c = 2, 8, 8, 128
            num_splits = 2
            num_head = 4
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": False,
                "h": h,
                "w": w,
                "attn_mask": None,
                "num_head": num_head,
                "description": f"Multi-head: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        if num_tests > 2:
            # Test 3: With window shift
            b, h, w, c = 2, 8, 8, 64
            num_splits = 2
            num_head = 2
            window_size_h = h // num_splits
            window_size_w = w // num_splits
            attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=torch.device("cpu"),
            )
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": True,
                "h": h,
                "w": w,
                "attn_mask": attn_mask,
                "num_head": num_head,
                "description": f"With shift: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        if num_tests > 3:
            # Test 4: Larger resolution
            b, h, w, c = 1, 16, 16, 128
            num_splits = 4
            num_head = 4
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": False,
                "h": h,
                "w": w,
                "attn_mask": None,
                "num_head": num_head,
                "description": f"Large resolution: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        if num_tests > 4:
            # Test 5: More heads with shift
            b, h, w, c = 3, 16, 16, 256
            num_splits = 2
            num_head = 8
            window_size_h = h // num_splits
            window_size_w = w // num_splits
            attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=torch.device("cpu"),
            )
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": True,
                "h": h,
                "w": w,
                "attn_mask": attn_mask,
                "num_head": num_head,
                "description": f"More heads with shift: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        if num_tests > 5:
            # Test 6: Single split (no splitting)
            b, h, w, c = 2, 8, 8, 64
            num_splits = 1
            num_head = 2
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": False,
                "h": h,
                "w": w,
                "attn_mask": None,
                "num_head": num_head,
                "description": f"No split: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        if num_tests > 6:
            # Test 7: Larger batch size
            b, h, w, c = 8, 8, 8, 128
            num_splits = 2
            num_head = 4
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": False,
                "h": h,
                "w": w,
                "attn_mask": None,
                "num_head": num_head,
                "description": f"Large batch: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        if num_tests > 7:
            # Test 8: Small feature dimensions
            b, h, w, c = 2, 4, 4, 32
            num_splits = 2
            num_head = 2
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": False,
                "h": h,
                "w": w,
                "attn_mask": None,
                "num_head": num_head,
                "description": f"Small dims: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        if num_tests > 8:
            # Test 9: More splits
            b, h, w, c = 2, 16, 16, 128
            num_splits = 8
            num_head = 4
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": False,
                "h": h,
                "w": w,
                "attn_mask": None,
                "num_head": num_head,
                "description": f"Many splits: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        if num_tests > 9:
            # Test 10: Non-square resolution
            b, h, w, c = 2, 8, 16, 128
            num_splits = 2
            num_head = 4
            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": False,
                "h": h,
                "w": w,
                "attn_mask": None,
                "num_head": num_head,
                "description": f"Non-square: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        # Generate additional tests if needed
        for i in range(num_tests - len(test_cases)):
            b = 2 + (i % 3)
            h = 8 * (1 + i // 3)
            w = h
            c = 64 * (1 + (i % 2))
            num_splits = 2 ** (i % 3)
            num_head = 2 ** ((i % 3) + 1)

            test_cases.append({
                "q": torch.randn(b, h * w, c),
                "k": torch.randn(b, h * w, c),
                "v": torch.randn(b, h * w, c),
                "num_splits": num_splits,
                "with_shift": False,
                "h": h,
                "w": w,
                "attn_mask": None,
                "num_head": num_head,
                "description": f"Additional test {i+1}: B={b}, H={h}, W={w}, C={c}, heads={num_head}, splits={num_splits}",
            })

        return test_cases[:num_tests]
