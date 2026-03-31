"""
Test Runner for depth-disparity conversion functions
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import torch
import os
import sys
import importlib.util
import time
from pathlib import Path
from datetime import datetime, timezone
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementation and test generator
try:
    from reference_implementation import relative_disparity_to_depth as ref_disp_to_depth
    from reference_implementation import depth_to_relative_disparity as ref_depth_to_disp
    from test_generator import TestDataGenerator
except ImportError as e:
    print(f"Error: {e}")
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
            
            if not hasattr(module, 'relative_disparity_to_depth'):
                raise AttributeError(f"No relative_disparity_to_depth function found in {filepath}")
            if not hasattr(module, 'depth_to_relative_disparity'):
                raise AttributeError(f"No depth_to_relative_disparity function found in {filepath}")
            
            return module.relative_disparity_to_depth, module.depth_to_relative_disparity
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check type
        if not isinstance(output, torch.Tensor):
            metrics['error'] = f"Output is not a Tensor (got {type(output)})"
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
    
    def test_functions(self, disp_to_depth_func, depth_to_disp_func, test_case):
        """Test both conversion functions."""
        near = test_case['near']
        far = test_case['far']
        relative_disparity = test_case['relative_disparity']
        depth = test_case['depth']
        eps = test_case['eps']
        
        results = {}
        
        # Test 1: relative_disparity_to_depth
        try:
            start_time = time.time()
            output_depth = disp_to_depth_func(relative_disparity, near, far, eps)
            exec_time_1 = time.time() - start_time
            
            reference_depth = ref_disp_to_depth(relative_disparity, near, far, eps)
            metrics_1 = self.compute_error(output_depth, reference_depth)
            metrics_1['execution_time'] = exec_time_1
            results['disp_to_depth'] = metrics_1
        except Exception as e:
            results['disp_to_depth'] = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        # Test 2: depth_to_relative_disparity
        try:
            start_time = time.time()
            output_disp = depth_to_disp_func(depth, near, far, eps)
            exec_time_2 = time.time() - start_time
            
            reference_disp = ref_depth_to_disp(depth, near, far, eps)
            metrics_2 = self.compute_error(output_disp, reference_disp)
            metrics_2['execution_time'] = exec_time_2
            results['depth_to_disp'] = metrics_2
        except Exception as e:
            results['depth_to_disp'] = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        # Test 3: Round-trip consistency (disparity -> depth -> disparity)
        try:
            output_depth_rt = disp_to_depth_func(relative_disparity, near, far, eps)
            output_disp_rt = depth_to_disp_func(output_depth_rt, near, far, eps)
            
            roundtrip_error = torch.max(torch.abs(output_disp_rt - relative_disparity)).item()
            results['roundtrip_disp'] = {
                'max_error': roundtrip_error,
                'pass': roundtrip_error < self.tolerance * 10,  # More lenient for roundtrip
            }
        except Exception as e:
            results['roundtrip_disp'] = {
                'error': str(e),
                'pass': False,
            }
        
        # Test 4: Round-trip consistency (depth -> disparity -> depth)
        try:
            output_disp_rt2 = depth_to_disp_func(depth, near, far, eps)
            output_depth_rt2 = disp_to_depth_func(output_disp_rt2, near, far, eps)
            
            roundtrip_error2 = torch.max(torch.abs(output_depth_rt2 - depth)).item()
            results['roundtrip_depth'] = {
                'max_error': roundtrip_error2,
                'pass': roundtrip_error2 < self.tolerance * 10,  # More lenient for roundtrip
            }
        except Exception as e:
            results['roundtrip_depth'] = {
                'error': str(e),
                'pass': False,
            }
        
        return results
    
    def test_single_implementation(self, impl_path):
        """Test a single LLM implementation."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # Load implementation
        disp_to_depth_func, depth_to_disp_func = self.load_llm_implementation(impl_path)
        
        if disp_to_depth_func is None or depth_to_disp_func is None:
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
            
            results = self.test_functions(disp_to_depth_func, depth_to_disp_func, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'results': results
            }
            
            if self.verbose:
                # Print disp_to_depth result
                r1 = results.get('disp_to_depth', {})
                if r1.get('pass', False):
                    print(f"  ✓ disp_to_depth Pass (L1={r1.get('l1_error', 0):.2e}, "
                          f"L2={r1.get('l2_error', 0):.2e}, max={r1.get('max_error', 0):.2e}, "
                          f"time={r1.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  ✗ disp_to_depth Fail - {r1.get('error', 'Error exceeds tolerance')}")
                    if 'max_error' in r1:
                        print(f"    Max error: {r1['max_error']:.2e} (tolerance: {self.tolerance:.2e})")
                
                # Print depth_to_disp result
                r2 = results.get('depth_to_disp', {})
                if r2.get('pass', False):
                    print(f"  ✓ depth_to_disp Pass (L1={r2.get('l1_error', 0):.2e}, "
                          f"L2={r2.get('l2_error', 0):.2e}, max={r2.get('max_error', 0):.2e}, "
                          f"time={r2.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  ✗ depth_to_disp Fail - {r2.get('error', 'Error exceeds tolerance')}")
                    if 'max_error' in r2:
                        print(f"    Max error: {r2['max_error']:.2e} (tolerance: {self.tolerance:.2e})")
                
                # Print roundtrip results
                rt1 = results.get('roundtrip_disp', {})
                rt2 = results.get('roundtrip_depth', {})
                if rt1.get('pass', False) and rt2.get('pass', False):
                    print(f"  ✓ Roundtrip OK (disp err={rt1.get('max_error', 0):.2e}, "
                          f"depth err={rt2.get('max_error', 0):.2e})")
                else:
                    print(f"  ⚠ Roundtrip issues detected")
            
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
        
        passes = []
        l1_errors = []
        l2_errors = []
        max_errors = []
        exec_times = []
        
        for test_result in all_results:
            results = test_result['results']
            
            # A test passes if both functions pass
            r1 = results.get('disp_to_depth', {})
            r2 = results.get('depth_to_disp', {})
            both_pass = r1.get('pass', False) and r2.get('pass', False)
            
            passes.append(both_pass)
            
            if both_pass:
                l1_errors.append((r1.get('l1_error', 0) + r2.get('l1_error', 0)) / 2)
                l2_errors.append((r1.get('l2_error', 0) + r2.get('l2_error', 0)) / 2)
                max_errors.append(max(r1.get('max_error', 0), r2.get('max_error', 0)))
                exec_times.append(r1.get('execution_time', 0) + r2.get('execution_time', 0))
        
        # Calculate metrics
        if passes:
            pass_rate = sum(passes) / len(passes) * 100
            summary['pass_rate'] = pass_rate
            summary['total_pass_count'] = sum(passes)
            summary['total_test_count'] = len(passes)
            
            if l1_errors:
                summary['avg_l1'] = sum(l1_errors) / len(l1_errors)
                summary['avg_l2'] = sum(l2_errors) / len(l2_errors)
                summary['avg_max'] = sum(max_errors) / len(max_errors)
                summary['avg_time'] = sum(exec_times) / len(exec_times)
        else:
            summary['pass_rate'] = 0.0
            summary['total_pass_count'] = 0
            summary['total_test_count'] = 0
        
        summary['overall_pass_rate'] = summary.get('pass_rate', 0.0)
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Summary for {summary['implementation']}:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Pass rate: {summary.get('pass_rate', 0.0):.1f}%")
        
        if 'avg_l1' in summary:
            print(f"  Avg L1 error: {summary['avg_l1']:.2e}")
            print(f"  Avg L2 error: {summary['avg_l2']:.2e}")
            print(f"  Avg max error: {summary['avg_max']:.2e}")
            print(f"  Avg time: {summary['avg_time']:.4f}s")
        
        pass_count = summary.get('total_pass_count', 0)
        test_count = summary.get('total_test_count', 0)
        print(f"  Overall: {summary.get('overall_pass_rate', 0.0):.1f}% ({pass_count}/{test_count} tests passed)")
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
        
        # Save structured test summary (schema.json compatible)
        self.save_summary_to_file(all_summaries)
        
        return all_summaries
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg Max':<12} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_l1 = f"{summary.get('avg_l1', 0):.2e}" if 'avg_l1' in summary else "N/A"
            avg_l2 = f"{summary.get('avg_l2', 0):.2e}" if 'avg_l2' in summary else "N/A"
            avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_max:<12} {avg_time:<12}")
        
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

    def save_summary_to_file(self, all_summaries, output_path=None):
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
        for summary in all_summaries or []:
            name = summary.get("implementation", "unknown")
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

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"✓ Structured summary saved to: {output_path}")
        return str(output_path)



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for depth-disparity conversion')
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

