"""
Test Runner for camera ray functions
Supports batch testing of multiple LLM implementations.
"""

import os
import sys
import importlib.util
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
import json

from reference_implementation import (
    sample_camera_rays as ref_sample_camera_rays,
    reflection as ref_reflection,
    sample_cubemap_color as ref_sample_cubemap_color,
    get_refl_color as ref_get_refl_color,
)
from test_generator import TestDataGenerator


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
            
            required_funcs = ['sample_camera_rays', 'reflection', 'sample_cubemap_color', 'get_refl_color']
            for func_name in required_funcs:
                if not hasattr(module, func_name):
                    raise AttributeError(f"No {func_name} function found in {filepath}")
            
            return {
                'sample_camera_rays': module.sample_camera_rays,
                'reflection': module.reflection,
                'sample_cubemap_color': module.sample_cubemap_color,
                'get_refl_color': module.get_refl_color,
            }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference, output_name="Output"):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check type
        if not isinstance(output, torch.Tensor):
            metrics['error'] = f"{output_name} is not a Tensor (got {type(output)})"
            return metrics
        
        # Check shape
        if output.shape != reference.shape:
            metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
            return metrics
        
        # L1 error (Mean Absolute Error)
        l1_error = torch.mean(torch.abs(output - reference)).item()
        metrics['l1_error'] = l1_error
        
        # L2 error (Root Mean Square Error)
        l2_error = torch.sqrt(torch.mean((output - reference) ** 2)).item()
        metrics['l2_error'] = l2_error
        
        # Max error
        max_error = torch.max(torch.abs(output - reference)).item()
        metrics['max_error'] = max_error
        
        # Relative error (avoid division by zero)
        ref_norm = torch.norm(reference)
        if ref_norm > 1e-10:
            relative_error = (torch.norm(output - reference) / ref_norm).item() * 100
        else:
            relative_error = 0.0 if max_error < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass (within tolerance)
        metrics['pass'] = max_error < self.tolerance
        
        return metrics
    
    def test_sample_camera_rays(self, impl_funcs, test_case):
        """Test sample_camera_rays function."""
        HWK = test_case['HWK']
        R = test_case['R']
        T = test_case['T']
        
        try:
            start_time = time.time()
            output = impl_funcs['sample_camera_rays'](HWK, R, T)
            exec_time = time.time() - start_time
            
            reference = ref_sample_camera_rays(HWK, R, T)
            metrics = self.compute_error(output, reference, "sample_camera_rays output")
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_reflection(self, impl_funcs, test_case):
        """Test reflection function."""
        rayd = test_case['rayd']
        normal = test_case['normal']
        
        # Normalize inputs
        rayd = rayd / torch.norm(rayd, dim=-1, keepdim=True)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        
        try:
            start_time = time.time()
            output = impl_funcs['reflection'](rayd, normal)
            exec_time = time.time() - start_time
            
            reference = ref_reflection(rayd, normal)
            metrics = self.compute_error(output, reference, "reflection output")
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_sample_cubemap_color(self, impl_funcs, test_case):
        """Test sample_cubemap_color function."""
        rays_d = test_case['rays_d_cubemap']
        env_map = test_case['env_map']
        
        # Normalize rays
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        try:
            start_time = time.time()
            output = impl_funcs['sample_cubemap_color'](rays_d, env_map)
            exec_time = time.time() - start_time
            
            reference = ref_sample_cubemap_color(rays_d, env_map)
            metrics = self.compute_error(output, reference, "sample_cubemap_color output")
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_get_refl_color(self, impl_funcs, test_case):
        """Test get_refl_color function."""
        envmap = test_case['env_map']
        HWK = test_case['HWK']
        R = test_case['R']
        T = test_case['T']
        normal_map = test_case['normal_map']
        
        try:
            start_time = time.time()
            output = impl_funcs['get_refl_color'](envmap, HWK, R, T, normal_map)
            exec_time = time.time() - start_time
            
            reference = ref_get_refl_color(envmap, HWK, R, T, normal_map)
            metrics = self.compute_error(output, reference, "get_refl_color output")
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
            
            # Test each function
            test_funcs = [
                ('sample_camera_rays', self.test_sample_camera_rays),
                ('reflection', self.test_reflection),
                ('sample_cubemap_color', self.test_sample_cubemap_color),
                ('get_refl_color', self.test_get_refl_color),
            ]
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'results': {}
            }
            
            for func_name, test_func in test_funcs:
                result = test_func(impl_funcs, test_case)
                test_result['results'][func_name] = result
                
                if self.verbose:
                    if result.get('pass', False):
                        print(f"  ✓ {func_name}: Pass (L1={result.get('l1_error', 0):.2e}, "
                              f"L2={result.get('l2_error', 0):.2e}, "
                              f"max={result.get('max_error', 0):.2e}, time={result.get('execution_time', 0):.4f}s)")
                    else:
                        print(f"  ✗ {func_name}: Fail - {result.get('error', 'Error exceeds tolerance')}")
                        if 'max_error' in result:
                            print(f"    Max error: {result['max_error']:.2e} (tolerance: {self.tolerance:.2e})")
            
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
        
        func_stats = {}
        total_passes = []
        
        for test_result in all_results:
            for func_name, result in test_result['results'].items():
                if func_name not in func_stats:
                    func_stats[func_name] = {
                        'passes': [],
                        'l1_errors': [],
                        'l2_errors': [],
                        'max_errors': [],
                        'exec_times': []
                    }
                
                if result.get('pass', False):
                    func_stats[func_name]['passes'].append(True)
                    func_stats[func_name]['l1_errors'].append(result.get('l1_error', 0))
                    func_stats[func_name]['l2_errors'].append(result.get('l2_error', 0))
                    func_stats[func_name]['max_errors'].append(result.get('max_error', 0))
                    func_stats[func_name]['exec_times'].append(result.get('execution_time', 0))
                    total_passes.append(True)
                else:
                    func_stats[func_name]['passes'].append(False)
                    total_passes.append(False)
        
        # Calculate per-function metrics
        summary['function_stats'] = {}
        for func_name, stats in func_stats.items():
            passes = stats['passes']
            func_summary = {
                'pass_rate': sum(passes) / len(passes) * 100 if passes else 0.0,
                'pass_count': sum(passes),
                'total_count': len(passes)
            }
            
            if stats['l1_errors']:
                func_summary['avg_l1'] = sum(stats['l1_errors']) / len(stats['l1_errors'])
                func_summary['avg_l2'] = sum(stats['l2_errors']) / len(stats['l2_errors'])
                func_summary['avg_max'] = sum(stats['max_errors']) / len(stats['max_errors'])
                func_summary['avg_time'] = sum(stats['exec_times']) / len(stats['exec_times'])
            
            summary['function_stats'][func_name] = func_summary
        
        # Overall metrics
        if total_passes:
            summary['overall_pass_rate'] = sum(total_passes) / len(total_passes) * 100
            summary['total_pass_count'] = sum(total_passes)
            summary['total_test_count'] = len(total_passes)
        else:
            summary['overall_pass_rate'] = 0.0
            summary['total_pass_count'] = 0
            summary['total_test_count'] = 0
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Summary for {summary['implementation']}:")
        print(f"  Total test cases: {summary['total_tests']}")
        print(f"  Overall pass rate: {summary.get('overall_pass_rate', 0.0):.1f}%")
        print(f"  Overall: {summary.get('total_pass_count', 0)}/{summary.get('total_test_count', 0)} tests passed")
        
        print(f"\n  Per-function statistics:")
        for func_name, func_stats in summary.get('function_stats', {}).items():
            print(f"\n    {func_name}:")
            print(f"      Pass rate: {func_stats['pass_rate']:.1f}% ({func_stats['pass_count']}/{func_stats['total_count']})")
            if 'avg_l1' in func_stats:
                print(f"      Avg L1: {func_stats['avg_l1']:.2e}, Avg L2: {func_stats['avg_l2']:.2e}")
                print(f"      Avg max: {func_stats['avg_max']:.2e}, Avg time: {func_stats['avg_time']:.4f}s")
        
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
        
        # Save structured summary for aggregation
        self.save_summary_to_file(all_summaries)
        
        return all_summaries
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}\n")
        
        # Overall comparison
        print(f"{'Implementation':<20} {'Overall Pass Rate':<20} {'Pass Count':<15}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation']
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<20} {'0.0%':<20} {'0/0':<15}")
                continue
            
            pass_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            pass_count = summary.get('total_pass_count', 0)
            test_count = summary.get('total_test_count', 0)
            count_str = f"{pass_count}/{test_count}"
            
            print(f"{name:<20} {pass_rate:<20} {count_str:<15}")
        
        print("-" * 100)
        
        # Per-function comparison
        print(f"\nPER-FUNCTION COMPARISON:")
        func_names = ['sample_camera_rays', 'reflection', 'sample_cubemap_color', 'get_refl_color']
        
        for func_name in func_names:
            print(f"\n{func_name}:")
            print(f"{'Implementation':<20} {'Pass Rate':<15} {'Avg L1':<12} {'Avg L2':<12} {'Avg Max':<12}")
            print("-" * 85)
            
            for summary in all_summaries:
                name = summary['implementation']
                
                if 'function_stats' in summary and func_name in summary['function_stats']:
                    stats = summary['function_stats'][func_name]
                    pass_rate = f"{stats['pass_rate']:.1f}%"
                    avg_l1 = f"{stats.get('avg_l1', 0):.2e}" if 'avg_l1' in stats else "N/A"
                    avg_l2 = f"{stats.get('avg_l2', 0):.2e}" if 'avg_l2' in stats else "N/A"
                    avg_max = f"{stats.get('avg_max', 0):.2e}" if 'avg_max' in stats else "N/A"
                else:
                    pass_rate = "0.0%"
                    avg_l1 = avg_l2 = avg_max = "N/A"
                
                print(f"{name:<20} {pass_rate:<15} {avg_l1:<12} {avg_l2:<12} {avg_max:<12}")
        
        print("-" * 100)
        
        # Print ranking
        print(f"\n{'OVERALL RANKING':<25} {'Pass Rate':<15} {'Pass Count':<15}")
        print("-" * 57)
        sorted_summaries = sorted(all_summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
        for i, summary in enumerate(sorted_summaries, 1):
            name = summary['implementation'][:23]
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            pass_count = summary.get('total_pass_count', 0)
            test_count = summary.get('total_test_count', 0)
            count_str = f"{pass_count}/{test_count}"
            print(f"{i}. {name:<30} {overall_rate:<15} {count_str:<15}")
        print("-" * 57)

    


    def save_summary_to_file(self, all_summaries, output_path=None):
        """Save structured test summary JSON aligned with schema.json."""
        if not all_summaries:
            return None
        
        script_dir = Path(__file__).parent  # .../unittest{N}
        project_id = script_dir.parent.name
        unittest_id = script_dir.name.replace("unittest", "")
        suite_path = f"{project_id}/{script_dir.name}"
        
        if output_path is None:
            output_path = script_dir / "test_summary.json"
        else:
            output_path = Path(output_path)
        
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        implementations = []
        for summary in all_summaries:
            name = summary.get("implementation", "")
            test_total = int(summary.get("total_test_count", summary.get("total_tests", 0)) or 0)
            test_pass = int(summary.get("total_pass_count", 0) or 0)
            implementations.append(
                {
                    "name": name,
                    "test_total": test_total,
                    "test_pass": test_pass,
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
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        
        print(f"✓ Structured summary saved to: {output_path}")
        return output_path
    
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
                f.write(f"Error tolerance: {self.tolerance}\n")
                f.write("="*100 + "\n\n")
                
                # Write detailed results for each implementation
                for summary in all_summaries:
                    f.write("="*80 + "\n")
                    f.write(f"Implementation: {summary['implementation']}\n")
                    f.write("="*80 + "\n")
                    
                    # Check if there was an error loading
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"ERROR: {summary['error']}\n")
                        f.write(f"Overall Pass Rate: 0.0% (0/0 tests passed)\n")
                        f.write("\n")
                        continue
                    
                    f.write(f"Total tests: {summary.get('total_tests', 0)}\n\n")
                    
                    # Write per-function statistics - dynamically find function names
                    func_names = [key.replace('_pass_rate', '') for key in summary.keys() 
                                  if key.endswith('_pass_rate')]
                    
                    for func_name in func_names:
                        pass_rate_key = f'{func_name}_pass_rate'
                        if pass_rate_key in summary:
                            f.write(f"{func_name}:\n")
                            f.write(f"  Pass rate: {summary[pass_rate_key]:.1f}%\n")
                            if f'{func_name}_avg_l1' in summary:
                                f.write(f"  Avg L1 error: {summary[f'{func_name}_avg_l1']:.2e}\n")
                                f.write(f"  Avg L2 error: {summary[f'{func_name}_avg_l2']:.2e}\n")
                                f.write(f"  Avg time: {summary[f'{func_name}_avg_time']:.4f}s\n")
                            f.write("\n")
                    
                    # Write overall statistics
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    overall_rate = summary.get('overall_pass_rate', 0.0)
                    f.write(f"Overall Average Pass Rate: {overall_rate:.1f}% ({pass_count}/{test_count} tests passed)\n")
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<20} {'Function':<20} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<30} {'ERROR':<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        f.write(f"{'  → AVERAGE':<20} {'(0/0)':<20} {'0.0%':<12} {'':<12} {'':<12} {'':<12}\n")
                        f.write("-" * 100 + "\n")
                        continue
                    
                    # Find all function names dynamically
                    func_names = [key.replace('_pass_rate', '') for key in summary.keys() 
                                  if key.endswith('_pass_rate')]
                    
                    for func_name in func_names:
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
                            
                            f.write(f"{name:<30} {func_name:<30} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_time:<12}\n")
                            name = ""
                    
                    overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    count_info = f"({pass_count}/{test_count})"
                    f.write(f"{'  → AVERAGE':<20} {count_info:<20} {overall_rate:<12} {'':<12} {'':<12} {'':<12}\n")
                    f.write("-" * 100 + "\n")
                
                # Write ranking
                f.write(f"\n{'OVERALL RANKING':<20} {'Avg Pass Rate':<20} {'Pass Count':<15}\n")
                f.write("-" * 57 + "\n")
                sorted_summaries = sorted(all_summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
                for i, summary in enumerate(sorted_summaries, 1):
                    name = summary['implementation']
                    overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    count_str = f"{pass_count}/{test_count}"
                    f.write(f"{i}. {name:<30} {overall_rate:<20} {count_str:<15}\n")
                f.write("-" * 57 + "\n")
            
            print(f"\n✓ Results saved to: {output_file}")
            return str(output_file)
        
        except Exception as e:
            print(f"\n✗ Error saving results to file: {e}")
            return None



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for camera ray functions')
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

