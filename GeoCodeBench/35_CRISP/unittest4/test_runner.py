"""
Test Runner for shape_recovery_from_pc_cvxpy() function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import json
import torch
import numpy as np
import os
import sys
import importlib.util
import time
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementation and test generator
try:
    from reference_implementation import shape_recovery_from_pc_cvxpy as RefShapeRecovery
except ImportError:
    print("Error: reference_implementation.py not found!")
    sys.exit(1)
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-4, device='cpu'):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.device = device
        self.test_generator = TestDataGenerator(device=device)
        self.test_cases = self.test_generator.generate_test_suite(num_tests)
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'shape_recovery_from_pc_cvxpy'):
                raise AttributeError(f"No shape_recovery_from_pc_cvxpy function found in {filepath}")
            
            return module.shape_recovery_from_pc_cvxpy
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check if output is a dictionary
        if not isinstance(output, dict) or not isinstance(reference, dict):
            metrics['error'] = f"Type mismatch: expected dict, got {type(output)} vs {type(reference)}"
            return metrics
        
        # Check required keys
        required_keys = ['postcrt_shape_code', 'postcrt_shape_coeffs', 'solver_statuss']
        for key in required_keys:
            if key not in output:
                metrics['error'] = f"Missing key in output: {key}"
                return metrics
            if key not in reference:
                metrics['error'] = f"Missing key in reference: {key}"
                return metrics
        
        # Compare postcrt_shape_code
        out_code = output['postcrt_shape_code']
        ref_code = reference['postcrt_shape_code']
        if not isinstance(out_code, torch.Tensor):
            metrics['error'] = f"postcrt_shape_code is not a tensor (got {type(out_code)})"
            return metrics
        if out_code.shape != ref_code.shape:
            metrics['error'] = f"postcrt_shape_code shape mismatch: {out_code.shape} vs {ref_code.shape}"
            return metrics
        
        code_l1 = torch.mean(torch.abs(out_code - ref_code)).item()
        code_l2 = torch.sqrt(torch.mean((out_code - ref_code) ** 2)).item()
        code_max = torch.max(torch.abs(out_code - ref_code)).item()
        
        metrics['code_l1_error'] = code_l1
        metrics['code_l2_error'] = code_l2
        metrics['code_max_error'] = code_max
        
        # Compare postcrt_shape_coeffs
        out_coeffs = output['postcrt_shape_coeffs']
        ref_coeffs = reference['postcrt_shape_coeffs']
        if not isinstance(out_coeffs, torch.Tensor):
            metrics['error'] = f"postcrt_shape_coeffs is not a tensor (got {type(out_coeffs)})"
            return metrics
        if out_coeffs.shape != ref_coeffs.shape:
            metrics['error'] = f"postcrt_shape_coeffs shape mismatch: {out_coeffs.shape} vs {ref_coeffs.shape}"
            return metrics
        
        coeffs_l1 = torch.mean(torch.abs(out_coeffs - ref_coeffs)).item()
        coeffs_l2 = torch.sqrt(torch.mean((out_coeffs - ref_coeffs) ** 2)).item()
        coeffs_max = torch.max(torch.abs(out_coeffs - ref_coeffs)).item()
        
        metrics['coeffs_l1_error'] = coeffs_l1
        metrics['coeffs_l2_error'] = coeffs_l2
        metrics['coeffs_max_error'] = coeffs_max
        
        # Overall max error
        metrics['max_error'] = max(code_max, coeffs_max)
        
        # Relative error
        ref_code_norm = torch.norm(ref_code).item()
        ref_coeffs_norm = torch.norm(ref_coeffs).item()
        if ref_code_norm > 1e-10:
            code_relative = (torch.norm(out_code - ref_code).item() / ref_code_norm) * 100
        else:
            code_relative = 0.0 if code_max < self.tolerance else 100.0
        if ref_coeffs_norm > 1e-10:
            coeffs_relative = (torch.norm(out_coeffs - ref_coeffs).item() / ref_coeffs_norm) * 100
        else:
            coeffs_relative = 0.0 if coeffs_max < self.tolerance else 100.0
        
        metrics['code_relative_error'] = code_relative
        metrics['coeffs_relative_error'] = coeffs_relative
        metrics['relative_error'] = max(code_relative, coeffs_relative)
        
        # Check if pass
        metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_function(self, impl_func, test_case, test_data):
        """Test shape_recovery_from_pc_cvxpy function."""
        try:
            start_time = time.time()
            with torch.no_grad():
                output = impl_func(
                    sdf_model=test_data['sdf_model'],
                    initial_shape_code=test_data['initial_shape_code'],
                    nocs=test_data['nocs'],
                    masks=test_data['masks'],
                    shape_code_library=test_data['shape_code_library'],
                    use_L1_reg=test_case['use_L1_reg'],
                    use_onehot=test_case['use_onehot'],
                    use_initial_shape_code_basis=test_case['use_initial_shape_code_basis'],
                    normalize_F_matrix=test_case['normalize_F_matrix'],
                    L1_weight=test_case['L1_weight'],
                )
            exec_time = time.time() - start_time
            
            # Get reference output
            with torch.no_grad():
                reference = RefShapeRecovery(
                    sdf_model=test_data['sdf_model'],
                    initial_shape_code=test_data['initial_shape_code'],
                    nocs=test_data['nocs'],
                    masks=test_data['masks'],
                    shape_code_library=test_data['shape_code_library'],
                    use_L1_reg=test_case['use_L1_reg'],
                    use_onehot=test_case['use_onehot'],
                    use_initial_shape_code_basis=test_case['use_initial_shape_code_basis'],
                    normalize_F_matrix=test_case['normalize_F_matrix'],
                    L1_weight=test_case['L1_weight'],
                )
            
            metrics = self.compute_error(output, reference)
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
            import traceback
            if self.verbose:
                traceback.print_exc()
        
        return metrics
    
    def test_single_implementation(self, impl_path):
        """Test a single LLM implementation."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # Load implementation
        impl_func = self.load_llm_implementation(impl_path)
        
        if impl_func is None:
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
            
            # Generate test data
            test_data = self.test_generator.create_test_data(test_case)
            
            result = self.test_function(impl_func, test_case, test_data)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (code_max={result.get('code_max_error', 0):.2e}, "
                          f"coeffs_max={result.get('coeffs_max_error', 0):.2e}, "
                          f"time={result.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
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
        
        passes = []
        code_l1_errors = []
        code_l2_errors = []
        code_max_errors = []
        coeffs_l1_errors = []
        coeffs_l2_errors = []
        coeffs_max_errors = []
        max_errors = []
        exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                code_l1_errors.append(result.get('code_l1_error', 0))
                code_l2_errors.append(result.get('code_l2_error', 0))
                code_max_errors.append(result.get('code_max_error', 0))
                coeffs_l1_errors.append(result.get('coeffs_l1_error', 0))
                coeffs_l2_errors.append(result.get('coeffs_l2_error', 0))
                coeffs_max_errors.append(result.get('coeffs_max_error', 0))
                max_errors.append(result.get('max_error', 0))
                exec_times.append(result.get('execution_time', 0))
            else:
                passes.append(False)
        
        # Calculate metrics
        if passes:
            pass_rate = sum(passes) / len(passes) * 100
            summary['pass_rate'] = pass_rate
            summary['total_pass_count'] = sum(passes)
            summary['total_test_count'] = len(passes)
            
            if code_l1_errors:
                summary['avg_code_l1'] = sum(code_l1_errors) / len(code_l1_errors)
                summary['avg_code_l2'] = sum(code_l2_errors) / len(code_l2_errors)
                summary['avg_code_max'] = sum(code_max_errors) / len(code_max_errors)
                summary['avg_coeffs_l1'] = sum(coeffs_l1_errors) / len(coeffs_l1_errors)
                summary['avg_coeffs_l2'] = sum(coeffs_l2_errors) / len(coeffs_l2_errors)
                summary['avg_coeffs_max'] = sum(coeffs_max_errors) / len(coeffs_max_errors)
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
        
        if 'avg_code_max' in summary:
            print(f"  Avg code max error: {summary['avg_code_max']:.2e}")
            print(f"  Avg coeffs max error: {summary['avg_coeffs_max']:.2e}")
            print(f"  Avg overall max error: {summary['avg_max']:.2e}")
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

        # Save structured test summary (schema.json-aligned)
        self.save_summary_to_file(all_summaries)
        
        return all_summaries

    def save_summary_to_file(self, all_results, output_path=None):
        """Save a structured `test_summary.json` aligned with `schema.json`."""
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
            # If an implementation failed to load, keep counts at 0.
            name = summary.get("implementation", "unknown")
            test_total = summary.get("total_test_count", summary.get("total_tests", 0))
            test_pass = summary.get("total_pass_count", 0)

            implementations.append(
                {
                    "name": name,
                    "test_total": int(test_total) if test_total is not None else 0,
                    "test_pass": int(test_pass) if test_pass is not None else 0,
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

        return str(output_path)
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg Max':<12} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_max:<12} {avg_time:<12}")
        
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
                    
                    # Write overall statistics
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    overall_rate = summary.get('overall_pass_rate', 0.0)
                    f.write(f"Overall Average Pass Rate: {overall_rate:.1f}% ({pass_count}/{test_count} tests passed)\n")
                    
                    if 'avg_code_max' in summary:
                        f.write(f"  Avg code max error: {summary['avg_code_max']:.2e}\n")
                        f.write(f"  Avg coeffs max error: {summary['avg_coeffs_max']:.2e}\n")
                        f.write(f"  Avg overall max error: {summary['avg_max']:.2e}\n")
                        f.write(f"  Avg time: {summary['avg_time']:.4f}s\n")
                    
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<20} {'Pass Rate':<12} {'Avg Max':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<30} {'ERROR':<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12}\n")
                        f.write("-" * 100 + "\n")
                        continue
                    
                    overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
                    avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
                    avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
                    
                    f.write(f"{name:<30} {overall_rate:<12} {avg_max:<12} {avg_time:<12}\n")
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
    
    parser = argparse.ArgumentParser(description='Test runner for shape_recovery_from_pc_cvxpy()')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Error tolerance for pass/fail (default: 1e-4)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    impl_dir = script_dir / args.impl_dir
    
    # Create test runner
    runner = TestRunner(
        num_tests=args.num_tests,
        verbose=not args.quiet,
        tolerance=args.tolerance,
        device=args.device
    )
    
    # Run tests
    results = runner.batch_test(str(impl_dir))
    
    return results


if __name__ == '__main__':
    main()
