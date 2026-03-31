"""
Test Runner for finite_cone_formulation and points_in_finite_cone functions
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import torch
import os
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from test_generator import TestDataGenerator

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementation
try:
    from reference_implementation import finite_cone_formulation as ref_finite_cone_formulation
    from reference_implementation import points_in_finite_cone as ref_points_in_finite_cone
except ImportError:
    print("Error: reference_implementation.py not found!")
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
            
            if not hasattr(module, 'finite_cone_formulation') or not hasattr(module, 'points_in_finite_cone'):
                raise AttributeError(f"Missing required functions in {filepath}")
            
            return module.finite_cone_formulation, module.points_in_finite_cone
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def compute_error(self, output, reference, test_name):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        if test_name == "finite_cone_formulation":
            # Check if output is a tuple
            if not isinstance(output, tuple) or len(output) != 3:
                metrics['error'] = f"Output is not a tuple of 3 elements (got {type(output)})"
                return metrics
            
            direction, height, half_angle_degrees = output
            ref_direction, ref_height, ref_half_angle_degrees = reference
            
            # Check types
            if not isinstance(direction, torch.Tensor):
                metrics['error'] = f"Direction is not a tensor (got {type(direction)})"
                return metrics
            if not isinstance(height, (float, int, torch.Tensor)):
                metrics['error'] = f"Height is not a number (got {type(height)})"
                return metrics
            if not isinstance(half_angle_degrees, (float, int, torch.Tensor)):
                metrics['error'] = f"Half angle degrees is not a number (got {type(half_angle_degrees)})"
                return metrics
            
            # Convert to tensors for comparison
            if isinstance(height, torch.Tensor):
                height = height.item()
            if isinstance(half_angle_degrees, torch.Tensor):
                half_angle_degrees = half_angle_degrees.item()
            if isinstance(ref_height, torch.Tensor):
                ref_height = ref_height.item()
            if isinstance(ref_half_angle_degrees, torch.Tensor):
                ref_half_angle_degrees = ref_half_angle_degrees.item()
            
            # Check shapes
            if direction.shape != ref_direction.shape:
                metrics['error'] = f"Direction shape mismatch: {direction.shape} vs {ref_direction.shape}"
                return metrics
            
            # Compute errors
            direction_error = torch.norm(direction - ref_direction).item()
            height_error = abs(height - ref_height)
            angle_error = abs(half_angle_degrees - ref_half_angle_degrees)
            
            metrics['direction_error'] = direction_error
            metrics['height_error'] = height_error
            metrics['angle_error'] = angle_error
            metrics['max_error'] = max(direction_error, height_error, angle_error)
            
            # Check if pass (within tolerance)
            metrics['pass'] = metrics['max_error'] < self.tolerance
            
        elif test_name == "points_in_finite_cone":
            # Check if output is a tensor
            if not isinstance(output, torch.Tensor):
                metrics['error'] = f"Output is not a tensor (got {type(output)})"
                return metrics
            
            # Check shape
            if output.shape != reference.shape:
                metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
                return metrics
            
            # Check dtype
            if output.dtype != reference.dtype:
                metrics['error'] = f"Dtype mismatch: {output.dtype} vs {reference.dtype}"
                return metrics
            
            # Compute errors
            # For boolean tensors, we compare element-wise
            if output.dtype == torch.bool:
                # Count mismatches
                mismatches = (output != reference).sum().item()
                total = output.numel()
                error_rate = mismatches / total if total > 0 else 0
                metrics['error_rate'] = error_rate
                metrics['mismatches'] = mismatches
                metrics['total'] = total
                metrics['max_error'] = error_rate
            else:
                # For numerical tensors
                l1_error = torch.mean(torch.abs(output.float() - reference.float())).item()
                l2_error = torch.sqrt(torch.mean((output.float() - reference.float()) ** 2)).item()
                max_error = torch.max(torch.abs(output.float() - reference.float())).item()
                
                metrics['l1_error'] = l1_error
                metrics['l2_error'] = l2_error
                metrics['max_error'] = max_error
            
            # Check if pass (within tolerance)
            if output.dtype == torch.bool:
                metrics['pass'] = metrics['error_rate'] < self.tolerance
            else:
                metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_finite_cone_formulation(self, impl_func, test_case):
        """Test finite_cone_formulation function."""
        top_point = test_case['top_point']
        base_center = test_case['base_center']
        radius = test_case['radius']
        
        try:
            start_time = time.time()
            output = impl_func(top_point, base_center, radius)
            exec_time = time.time() - start_time
            
            reference = ref_finite_cone_formulation(top_point, base_center, radius)
            metrics = self.compute_error(output, reference, "finite_cone_formulation")
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_points_in_finite_cone(self, impl_func, test_case):
        """Test points_in_finite_cone function."""
        top_point = test_case['top_point']
        base_center = test_case['base_center']
        radius = test_case['radius']
        
        # Generate test points
        points = self.test_generator.generate_points_for_cone_test(top_point, base_center, radius)
        
        # Get cone parameters from reference implementation
        ref_direction, ref_height, ref_angle_cosine = ref_finite_cone_formulation(top_point, base_center, radius)
        
        try:
            start_time = time.time()
            output = impl_func(points, top_point, ref_direction, ref_angle_cosine, ref_height)
            exec_time = time.time() - start_time
            
            reference = ref_points_in_finite_cone(points, top_point, ref_direction, ref_angle_cosine, ref_height)
            metrics = self.compute_error(output, reference, "points_in_finite_cone")
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
        finite_cone_func, points_in_cone_func = self.load_llm_implementation(impl_path)
        
        if finite_cone_func is None or points_in_cone_func is None:
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
            
            # Test finite_cone_formulation
            cone_result = self.test_finite_cone_formulation(finite_cone_func, test_case)
            
            # Test points_in_finite_cone
            points_result = self.test_points_in_finite_cone(points_in_cone_func, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'cone_result': cone_result,
                'points_result': points_result,
                'overall_pass': cone_result.get('pass', False) and points_result.get('pass', False)
            }
            
            if self.verbose:
                cone_status = "✓" if cone_result.get('pass', False) else "✗"
                points_status = "✓" if points_result.get('pass', False) else "✗"
                overall_status = "✓" if test_result['overall_pass'] else "✗"
                
                print(f"  {cone_status} finite_cone_formulation: ", end="")
                if cone_result.get('pass', False):
                    print(f"max_error={cone_result.get('max_error', 0):.2e}, time={cone_result.get('execution_time', 0):.4f}s")
                else:
                    print(f"FAIL - {cone_result.get('error', 'Error exceeds tolerance')}")
                
                print(f"  {points_status} points_in_finite_cone: ", end="")
                if points_result.get('pass', False):
                    if 'error_rate' in points_result:
                        print(f"error_rate={points_result.get('error_rate', 0):.2e}, time={points_result.get('execution_time', 0):.4f}s")
                    else:
                        print(f"max_error={points_result.get('max_error', 0):.2e}, time={points_result.get('execution_time', 0):.4f}s")
                else:
                    print(f"FAIL - {points_result.get('error', 'Error exceeds tolerance')}")
                
                print(f"  {overall_status} Overall: {'PASS' if test_result['overall_pass'] else 'FAIL'}")
            
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
        
        cone_passes = []
        points_passes = []
        overall_passes = []
        cone_errors = []
        points_errors = []
        cone_times = []
        points_times = []
        
        for test_result in all_results:
            cone_result = test_result['cone_result']
            points_result = test_result['points_result']
            
            if cone_result.get('pass', False):
                cone_passes.append(True)
                cone_errors.append(cone_result.get('max_error', 0))
                cone_times.append(cone_result.get('execution_time', 0))
            else:
                cone_passes.append(False)
            
            if points_result.get('pass', False):
                points_passes.append(True)
                if 'error_rate' in points_result:
                    points_errors.append(points_result.get('error_rate', 0))
                else:
                    points_errors.append(points_result.get('max_error', 0))
                points_times.append(points_result.get('execution_time', 0))
            else:
                points_passes.append(False)
            
            overall_passes.append(test_result['overall_pass'])
        
        # Calculate metrics
        if cone_passes:
            cone_pass_rate = sum(cone_passes) / len(cone_passes) * 100
            summary['cone_pass_rate'] = cone_pass_rate
            summary['cone_pass_count'] = sum(cone_passes)
            
            if cone_errors:
                summary['avg_cone_error'] = sum(cone_errors) / len(cone_errors)
                summary['avg_cone_time'] = sum(cone_times) / len(cone_times)
        else:
            summary['cone_pass_rate'] = 0.0
            summary['cone_pass_count'] = 0
        
        if points_passes:
            points_pass_rate = sum(points_passes) / len(points_passes) * 100
            summary['points_pass_rate'] = points_pass_rate
            summary['points_pass_count'] = sum(points_passes)
            
            if points_errors:
                summary['avg_points_error'] = sum(points_errors) / len(points_errors)
                summary['avg_points_time'] = sum(points_times) / len(points_times)
        else:
            summary['points_pass_rate'] = 0.0
            summary['points_pass_count'] = 0
        
        if overall_passes:
            overall_pass_rate = sum(overall_passes) / len(overall_passes) * 100
            summary['overall_pass_rate'] = overall_pass_rate
            summary['total_pass_count'] = sum(overall_passes)
            summary['total_test_count'] = len(overall_passes)
        else:
            summary['overall_pass_rate'] = 0.0
            summary['total_pass_count'] = 0
            summary['total_test_count'] = 0
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Summary for {summary['implementation']}:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  finite_cone_formulation pass rate: {summary.get('cone_pass_rate', 0.0):.1f}%")
        print(f"  points_in_finite_cone pass rate: {summary.get('points_pass_rate', 0.0):.1f}%")
        print(f"  Overall pass rate: {summary.get('overall_pass_rate', 0.0):.1f}%")
        
        if 'avg_cone_error' in summary:
            print(f"  Avg cone error: {summary['avg_cone_error']:.2e}")
            print(f"  Avg cone time: {summary['avg_cone_time']:.4f}s")
        
        if 'avg_points_error' in summary:
            print(f"  Avg points error: {summary['avg_points_error']:.2e}")
            print(f"  Avg points time: {summary['avg_points_time']:.4f}s")
        
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
        
        # Save structured per-run summary (schema-aligned)
        self.save_summary_to_file(all_summaries)
        
        return all_summaries
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*120}")
        print("COMPARISON SUMMARY")
        print(f"{'='*120}\n")
        
        # Header
        print(f"{'Implementation':<25} {'Overall':<10} {'Cone':<10} {'Points':<10} {'Cone Error':<12} {'Points Error':<12}")
        print("-" * 120)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<10} {'0.0%':<10} {'0.0%':<10} {'N/A':<12} {'N/A':<12}")
                continue
            
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            cone_rate = f"{summary.get('cone_pass_rate', 0.0):.1f}%"
            points_rate = f"{summary.get('points_pass_rate', 0.0):.1f}%"
            
            cone_error = f"{summary.get('avg_cone_error', 0):.2e}" if 'avg_cone_error' in summary else "N/A"
            points_error = f"{summary.get('avg_points_error', 0):.2e}" if 'avg_points_error' in summary else "N/A"
            
            print(f"{name:<25} {overall_rate:<10} {cone_rate:<10} {points_rate:<10} {cone_error:<12} {points_error:<12}")
        
        print("-" * 120)
        
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

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured schema-aligned test summary JSON."""
        if not all_results:
            return None

        script_dir = Path(__file__).parent
        project_id = script_dir.parent.name
        unittest_id = script_dir.name.replace("unittest", "")
        suite_path = f"{project_id}/{script_dir.name}"

        suite = {
            "project_id": project_id,
            "unittest_id": unittest_id,
            "suite_path": suite_path,
            "num_tests_requested": int(self.num_tests),
        }

        implementations = []
        for summary in all_results:
            impl_name = summary.get("implementation", "unknown")
            test_total = summary.get("total_test_count", summary.get("total_tests", 0)) or 0
            test_pass = summary.get("total_pass_count", 0) or 0

            implementations.append(
                {
                    "name": impl_name,
                    "test_total": int(test_total),
                    "test_pass": int(test_pass),
                }
            )

        payload = {
            "suite": suite,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": implementations,
        }

        if output_path is None:
            output_path = script_dir / "test_summary.json"
        else:
            output_path = Path(output_path)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if self.verbose:
                print(f"\n✓ Structured summary saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving structured summary: {e}")
            return None



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for finite_cone_formulation and points_in_finite_cone')
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
