"""
Test Runner for coords_grid, yin_to_3d, yang90_from_3d functions
Supports batch testing of multiple LLM implementations.
"""

import torch
import os
import sys
import importlib.util
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementations
try:
    from reference_implementation import (
        coords_grid as ref_coords_grid,
        yin_to_3d as ref_yin_to_3d,
        yang90_from_3d as ref_yang90_from_3d
    )
except ImportError:
    print("Error: reference_implementation.py not found!")
    sys.exit(1)

try:
    from test_generator import TestDataGenerator
except ImportError:
    print("Error: test_generator.py not found!")
    sys.exit(1)

class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-5):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.test_generator = TestDataGenerator()
        self.test_cases = self.test_generator.generate_test_suite(num_tests)
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            funcs = {}
            if hasattr(module, 'coords_grid'):
                funcs['coords_grid'] = module.coords_grid
            if hasattr(module, 'yin_to_3d'):
                funcs['yin_to_3d'] = module.yin_to_3d
            if hasattr(module, 'yang90_from_3d'):
                funcs['yang90_from_3d'] = module.yang90_from_3d
            
            if len(funcs) == 0:
                raise AttributeError(f"No test functions found in {filepath}")
            
            return funcs
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference, func_name):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check type
        if not isinstance(output, torch.Tensor):
            metrics['error'] = f"Output is not a torch.Tensor (got {type(output)})"
            return metrics
        
        # Check shape
        if output.shape != reference.shape:
            metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
            return metrics
        
        # Convert to float for comparison
        output_f = output.float()
        reference_f = reference.float()
        
        # L1 error (Mean Absolute Error)
        l1_error = torch.mean(torch.abs(output_f - reference_f)).item()
        metrics['l1_error'] = l1_error
        
        # L2 error (Root Mean Square Error)
        l2_error = torch.sqrt(torch.mean((output_f - reference_f) ** 2)).item()
        metrics['l2_error'] = l2_error
        
        # Max error
        max_error = torch.max(torch.abs(output_f - reference_f)).item()
        metrics['max_error'] = max_error
        
        # Relative error (avoid division by zero)
        ref_norm = torch.norm(reference_f)
        if ref_norm > 1e-10:
            relative_error = (torch.norm(output_f - reference_f) / ref_norm).item() * 100
        else:
            relative_error = 0.0 if max_error < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass (within tolerance)
        metrics['pass'] = max_error < self.tolerance
        
        return metrics
    
    def test_coords_grid(self, impl_func, test_case):
        """Test coords_grid function."""
        results = {}
        b, h, w = test_case['b'], test_case['h'], test_case['w']
        
        # Test without homogeneous
        try:
            start_time = time.time()
            output = impl_func(b, h, w, homogeneous=False, device=None)
            exec_time = time.time() - start_time
            
            reference = ref_coords_grid(b, h, w, homogeneous=False, device=None)
            metrics = self.compute_error(output, reference, 'coords_grid')
            metrics['execution_time'] = exec_time
            results['without_homogeneous'] = metrics
        except Exception as e:
            results['without_homogeneous'] = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        # Test with homogeneous
        try:
            start_time = time.time()
            output = impl_func(b, h, w, homogeneous=True, device=None)
            exec_time = time.time() - start_time
            
            reference = ref_coords_grid(b, h, w, homogeneous=True, device=None)
            metrics = self.compute_error(output, reference, 'coords_grid')
            metrics['execution_time'] = exec_time
            results['with_homogeneous'] = metrics
        except Exception as e:
            results['with_homogeneous'] = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return results
    
    def test_yin_to_3d(self, impl_func, test_case):
        """Test yin_to_3d function."""
        h, w = test_case['h'], test_case['w']
        n_points = test_case['n_points']
        
        # Generate random 2D grid points
        grid = torch.rand(n_points, 2)
        grid[:, 0] = grid[:, 0] * (w - 1)  # x in [0, w-1]
        grid[:, 1] = grid[:, 1] * (h - 1)  # y in [0, h-1]
        
        try:
            start_time = time.time()
            output = impl_func(grid, h, w)
            exec_time = time.time() - start_time
            
            reference = ref_yin_to_3d(grid, h, w)
            metrics = self.compute_error(output, reference, 'yin_to_3d')
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_yang90_from_3d(self, impl_func, test_case):
        """Test yang90_from_3d function."""
        h, w = test_case['h'], test_case['w']
        n_points = test_case['n_points']
        
        # Generate random 3D points on unit sphere
        theta = torch.rand(n_points) * 2 * torch.pi
        phi = torch.rand(n_points) * torch.pi
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        points = torch.stack([x, y, z], dim=1)
        
        try:
            start_time = time.time()
            output = impl_func(points, h, w)
            exec_time = time.time() - start_time
            
            reference = ref_yang90_from_3d(points, h, w)
            metrics = self.compute_error(output, reference, 'yang90_from_3d')
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_single_implementation(self, impl_path):
        """Test a single LLM implementation."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # Load implementation
        impl_funcs = self.load_llm_implementation(impl_path)
        
        if impl_funcs is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': 0
            }
        
        all_results = []
        
        # Run all test cases
        for i, test_case in enumerate(self.test_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(self.test_cases)}: {test_case['description']}")
            
            test_result = {
                'test_idx': i,
                'description': test_case['description']
            }
            
            # Test coords_grid
            if 'coords_grid' in impl_funcs:
                if self.verbose:
                    print("  Testing coords_grid...")
                coords_result = self.test_coords_grid(impl_funcs['coords_grid'], test_case)
                test_result['coords_grid'] = coords_result
                
                if self.verbose:
                    for variant, metrics in coords_result.items():
                        if metrics.get('pass', False):
                            print(f"    {variant}: ✓ Pass (L1={metrics.get('l1_error', 0):.2e}, L2={metrics.get('l2_error', 0):.2e}, time={metrics.get('execution_time', 0):.4f}s)")
                        else:
                            print(f"    {variant}: ✗ Fail - {metrics.get('error', 'Error exceeds tolerance')}")
            
            # Test yin_to_3d
            if 'yin_to_3d' in impl_funcs:
                if self.verbose:
                    print("  Testing yin_to_3d...")
                yin_result = self.test_yin_to_3d(impl_funcs['yin_to_3d'], test_case)
                test_result['yin_to_3d'] = yin_result
                
                if self.verbose:
                    if yin_result.get('pass', False):
                        print(f"    ✓ Pass (L1={yin_result.get('l1_error', 0):.2e}, L2={yin_result.get('l2_error', 0):.2e}, time={yin_result.get('execution_time', 0):.4f}s)")
                    else:
                        print(f"    ✗ Fail - {yin_result.get('error', 'Error exceeds tolerance')}")
            
            # Test yang90_from_3d
            if 'yang90_from_3d' in impl_funcs:
                if self.verbose:
                    print("  Testing yang90_from_3d...")
                yang_result = self.test_yang90_from_3d(impl_funcs['yang90_from_3d'], test_case)
                test_result['yang90_from_3d'] = yang_result
                
                if self.verbose:
                    if yang_result.get('pass', False):
                        print(f"    ✓ Pass (L1={yang_result.get('l1_error', 0):.2e}, L2={yang_result.get('l2_error', 0):.2e}, time={yang_result.get('execution_time', 0):.4f}s)")
                    else:
                        print(f"    ✗ Fail - {yang_result.get('error', 'Error exceeds tolerance')}")
            
            all_results.append(test_result)
        
        # Compute summary
        summary = self.compute_summary(impl_name, all_results)
        
        if self.verbose:
            self.print_summary(summary)
        
        return summary
    
    def compute_summary(self, impl_name, all_results):
        """Compute summary statistics."""
        summary = {
            'implementation': impl_name,
            'total_tests': len(all_results),
            'results': all_results
        }
        
        # Count passes for each function
        all_pass_rates = []
        total_pass_count = 0
        total_test_count = 0
        
        for func_name in ['coords_grid', 'yin_to_3d', 'yang90_from_3d']:
            passes = []
            l1_errors = []
            l2_errors = []
            exec_times = []
            
            for result in all_results:
                if func_name in result:
                    if func_name == 'coords_grid':
                        # coords_grid has two variants
                        for variant, metrics in result[func_name].items():
                            total_test_count += 1
                            if metrics.get('pass', False):
                                passes.append(True)
                                total_pass_count += 1
                                l1_errors.append(metrics.get('l1_error', 0))
                                l2_errors.append(metrics.get('l2_error', 0))
                                exec_times.append(metrics.get('execution_time', 0))
                            else:
                                passes.append(False)
                    else:
                        total_test_count += 1
                        metrics = result[func_name]
                        if metrics.get('pass', False):
                            passes.append(True)
                            total_pass_count += 1
                            l1_errors.append(metrics.get('l1_error', 0))
                            l2_errors.append(metrics.get('l2_error', 0))
                            exec_times.append(metrics.get('execution_time', 0))
                        else:
                            passes.append(False)
            
            if passes:
                pass_rate = sum(passes) / len(passes) * 100
                summary[f'{func_name}_pass_rate'] = pass_rate
                all_pass_rates.append(pass_rate)
                if l1_errors:
                    summary[f'{func_name}_avg_l1'] = sum(l1_errors) / len(l1_errors)
                    summary[f'{func_name}_avg_l2'] = sum(l2_errors) / len(l2_errors)
                    summary[f'{func_name}_avg_time'] = sum(exec_times) / len(exec_times)
        
        # Calculate overall average pass rate and counts
        if all_pass_rates:
            summary['overall_pass_rate'] = sum(all_pass_rates) / len(all_pass_rates)
        else:
            summary['overall_pass_rate'] = 0.0
        
        summary['total_pass_count'] = total_pass_count
        summary['total_test_count'] = total_test_count
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Summary for {summary['implementation']}:")
        print(f"  Total tests: {summary['total_tests']}")
        
        for func_name in ['coords_grid', 'yin_to_3d', 'yang90_from_3d']:
            pass_rate_key = f'{func_name}_pass_rate'
            if pass_rate_key in summary:
                print(f"\n  {func_name}:")
                print(f"    Pass rate: {summary[pass_rate_key]:.1f}%")
                if f'{func_name}_avg_l1' in summary:
                    print(f"    Avg L1 error: {summary[f'{func_name}_avg_l1']:.2e}")
                    print(f"    Avg L2 error: {summary[f'{func_name}_avg_l2']:.2e}")
                    print(f"    Avg time: {summary[f'{func_name}_avg_time']:.4f}s")
        
        pass_count = summary.get('total_pass_count', 0)
        test_count = summary.get('total_test_count', 0)
        print(f"\n  Overall Average Pass Rate: {summary.get('overall_pass_rate', 0.0):.1f}% ({pass_count}/{test_count} tests passed)")
        print(f"{'='*80}")
    
    def batch_test(self, implementations_dir):
        """Test all implementations in a directory."""
        impl_dir = Path(implementations_dir)
        
        if not impl_dir.exists():
            print(f"Error: Directory {implementations_dir} does not exist")
            return []
        
        # Find all Python files
        impl_files = list(impl_dir.glob("*.py"))
        impl_files = [f for f in impl_files if f.stem not in ['__init__', 'llm_template']]
        
        if not impl_files:
            print(f"No implementation files found in {implementations_dir}")
            return []
        
        print(f"\nFound {len(impl_files)} implementations to test")
        print(f"Running {self.num_tests} test cases per implementation\n")
        
        # Test each implementation
        all_summaries = []
        for impl_file in impl_files:
            summary = self.test_single_implementation(str(impl_file))
            all_summaries.append(summary)
        
        # Print comparison
        self.print_comparison(all_summaries)
        
        # Save results to file
        self.save_results_to_file(all_summaries)

        # Save structured summary to JSON (schema-aligned)
        self.save_summary_to_file(all_summaries)
        
        return all_summaries

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured test summary aligned with schema.json."""
        if output_path is None:
            output_path = Path(__file__).parent / "test_summary.json"
        else:
            output_path = Path(output_path)

        script_dir = Path(__file__).parent
        project_id = script_dir.parent.name
        unittest_id = script_dir.name.replace("unittest", "")
        suite_path = f"{project_id}/{script_dir.name}"

        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        implementations = []
        for summary in all_results or []:
            impl_name = summary.get("implementation", "unknown")
            test_total = summary.get("total_test_count", summary.get("total_tests", 0))
            test_pass = summary.get("total_pass_count", 0)
            implementations.append(
                {
                    "name": impl_name,
                    "test_total": int(test_total or 0),
                    "test_pass": int(test_pass or 0),
                }
            )

        payload = {
            "suite": {
                "project_id": project_id,
                "unittest_id": unittest_id,
                "suite_path": suite_path,
                "num_tests_requested": self.num_tests,
            },
            "timestamp_utc": timestamp_utc,
            "implementations": implementations,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if self.verbose:
            print(f"\n✓ Structured test summary saved to: {output_path}")
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Implementation':<20} {'Function':<20} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation']
            
            # Check if there was an error loading the implementation
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<20} {'ERROR':<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                print(f"{'  → AVERAGE':<20} {'(0/0)':<20} {'0.0%':<12} {'':<12} {'':<12} {'':<12}")
                print("-" * 100)
                continue
            
            for func_name in ['coords_grid', 'yin_to_3d', 'yang90_from_3d']:
                pass_rate_key = f'{func_name}_pass_rate'
                if pass_rate_key in summary:
                    pass_rate = f"{summary[pass_rate_key]:.1f}%"
                    
                    if f'{func_name}_avg_l1' in summary:
                        avg_l1 = f"{summary[f'{func_name}_avg_l1']:.2e}"
                        avg_l2 = f"{summary[f'{func_name}_avg_l2']:.2e}"
                        avg_time = f"{summary[f'{func_name}_avg_time']:.4f}s"
                    else:
                        avg_l1 = "N/A"
                        avg_l2 = "N/A"
                        avg_time = "N/A"
                    
                    print(f"{name:<20} {func_name:<20} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_time:<12}")
                    name = ""  # Only print name once
            
            # Print overall average pass rate for this implementation
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            pass_count = summary.get('total_pass_count', 0)
            test_count = summary.get('total_test_count', 0)
            count_info = f"({pass_count}/{test_count})"
            print(f"{'  → AVERAGE':<20} {count_info:<20} {overall_rate:<12} {'':<12} {'':<12} {'':<12}")
            print("-" * 100)
        
        # Print final summary sorted by overall pass rate
        print(f"\n{'OVERALL RANKING':<20} {'Avg Pass Rate':<20} {'Pass Count':<15}")
        print("-" * 57)
        sorted_summaries = sorted(all_summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
        for i, summary in enumerate(sorted_summaries, 1):
            name = summary['implementation']
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            pass_count = summary.get('total_pass_count', 0)
            test_count = summary.get('total_test_count', 0)
            count_str = f"{pass_count}/{test_count}"
            print(f"{i}. {name:<30} {overall_rate:<20} {count_str:<15}")
        print("-" * 57)
    




    def save_results_to_file(self, all_summaries, output_dir=None):
        """Save test results to a text file."""
        if not all_summaries:
            return None
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"test_results_{timestamp}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*100 + "\n")
                f.write("TEST RESULTS SUMMARY\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of implementations tested: {len(all_summaries)}\n")
                f.write(f"Number of test cases per implementation: {self.num_tests}\n")
                f.write("="*100 + "\n\n")
                
                # Write detailed results for each implementation
                for summary in all_summaries:
                    f.write("="*80 + "\n")
                    f.write(f"Implementation: {summary['implementation']}\n")
                    f.write("="*80 + "\n")
                    
                    # Check if there was an error loading
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"ERROR: {summary['error']}\n")
                        f.write(f"Pass Rate: 0/0 (0.0%)\n")
                        f.write("\n")
                        continue
                    
                    # Write test summary
                    total_checks = summary.get('total_test_count', 0)
                    pass_count = summary.get('total_pass_count', 0)
                    failed_count = max(total_checks - pass_count, 0)
                    f.write(f"Total tests: {summary.get('total_tests', 0)}\n")
                    f.write(f"Successful tests: {pass_count}\n")
                    f.write(f"Failed tests: {failed_count}\n")
                    f.write(f"Overall pass rate: {summary.get('overall_pass_rate', 0.0):.1f}%\n\n")
                    
                    # Write detailed test results if available
                    if 'results' in summary:
                        f.write("Detailed Test Results:\n")
                        f.write("-" * 80 + "\n")
                        for i, test_result in enumerate(summary['results'], 1):
                            test_desc = test_result.get('description', f'Test {i}')
                            f.write(f"\nTest {i}: {test_desc}\n")
                            for func_name in ['coords_grid', 'yin_to_3d', 'yang90_from_3d']:
                                if func_name not in test_result:
                                    continue
                                func_result = test_result[func_name]
                                if func_name == 'coords_grid':
                                    for variant, metrics in func_result.items():
                                        status = "✓" if metrics.get('pass', False) else "✗"
                                        if metrics.get('pass', False):
                                            f.write(
                                                f"  {func_name}.{variant}: {status} "
                                                f"(L1={metrics.get('l1_error', 0):.2e}, "
                                                f"L2={metrics.get('l2_error', 0):.2e}, "
                                                f"time={metrics.get('execution_time', 0):.4f}s)\n"
                                            )
                                        else:
                                            f.write(
                                                f"  {func_name}.{variant}: {status} "
                                                f"{metrics.get('error', 'Error exceeds tolerance')}\n"
                                            )
                                else:
                                    status = "✓" if func_result.get('pass', False) else "✗"
                                    if func_result.get('pass', False):
                                        f.write(
                                            f"  {func_name}: {status} "
                                            f"(L1={func_result.get('l1_error', 0):.2e}, "
                                            f"L2={func_result.get('l2_error', 0):.2e}, "
                                            f"time={func_result.get('execution_time', 0):.4f}s)\n"
                                        )
                                    else:
                                        f.write(
                                            f"  {func_name}: {status} "
                                            f"{func_result.get('error', 'Error exceeds tolerance')}\n"
                                        )
                        
                        f.write("\n" + "-" * 80 + "\n\n")
                    
                    # Write statistics summary if available
                    if summary.get('total_test_count', 0) > 0:
                        f.write("Summary Statistics:\n")
                        for func_name in ['coords_grid', 'yin_to_3d', 'yang90_from_3d']:
                            pass_key = f'{func_name}_pass_rate'
                            if pass_key in summary:
                                f.write(f"  {func_name} pass rate: {summary[pass_key]:.1f}%\n")
                                if f'{func_name}_avg_l1' in summary:
                                    f.write(f"    Avg L1: {summary[f'{func_name}_avg_l1']:.2e}\n")
                                    f.write(f"    Avg L2: {summary[f'{func_name}_avg_l2']:.2e}\n")
                                    f.write(f"    Avg time: {summary[f'{func_name}_avg_time']:.4f}s\n")
                    
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<30} {'Pass Rate':<12} {'Overall Rate':<12} {'Avg L1':<12} {'Avg Time':<12}\n")
                f.write("-" * 80 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<30} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        continue
                    
                    pass_count = summary.get('total_pass_count', 0)
                    total_count = summary.get('total_test_count', 0)
                    pass_rate = f"{pass_count}/{total_count}"
                    overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"

                    avg_l1_list = [
                        summary[key]
                        for key in ['coords_grid_avg_l1', 'yin_to_3d_avg_l1', 'yang90_from_3d_avg_l1']
                        if key in summary
                    ]
                    avg_time_list = [
                        summary[key]
                        for key in ['coords_grid_avg_time', 'yin_to_3d_avg_time', 'yang90_from_3d_avg_time']
                        if key in summary
                    ]
                    avg_l1 = f"{(sum(avg_l1_list) / len(avg_l1_list)):.2e}" if avg_l1_list else "N/A"
                    avg_time = f"{(sum(avg_time_list) / len(avg_time_list)):.4f}s" if avg_time_list else "N/A"
                    
                    f.write(f"{name:<30} {pass_rate:<12} {overall_rate:<12} {avg_l1:<12} {avg_time:<12}\n")
                
                f.write("-" * 80 + "\n")
            
            print(f"\n✓ Results saved to: {output_file}")
            return str(output_file)
        
        except Exception as e:
            print(f"\n✗ Error saving results to file: {e}")
            return None



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for coords_grid, yin_to_3d, yang90_from_3d')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                       help='Error tolerance for pass/fail (default: 1e-5)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    impl_dir = script_dir / args.impl_dir
    
    # Create test runner
    runner = TestRunner(
        num_tests=args.num_tests,
        verbose=not args.quiet,
        tolerance=args.tolerance
    )
    
    # Run tests
    results = runner.batch_test(str(impl_dir))
    
    return results


if __name__ == '__main__':
    main()
