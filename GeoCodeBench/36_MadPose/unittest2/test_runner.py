"""
Test Runner for solve_shift_and_scale_shared_focal() function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import numpy as np
import os
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

from reference_implementation import solve_shift_and_scale_shared_focal as ref_solve_shift_and_scale_shared_focal
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
            
            if not hasattr(module, 'solve_shift_and_scale_shared_focal'):
                raise AttributeError(f"No solve_shift_and_scale_shared_focal function found in {filepath}")
            
            return module.solve_shift_and_scale_shared_focal
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check if output is a list
        if not isinstance(output, list):
            metrics['error'] = f"Output is not a list (got {type(output)})"
            return metrics
        
        if not isinstance(reference, list):
            metrics['error'] = f"Reference is not a list (got {type(reference)})"
            return metrics
        
        # If both are empty lists, consider it a pass
        if len(output) == 0 and len(reference) == 0:
            metrics['pass'] = True
            metrics['mse'] = 0.0
            metrics['max_error'] = 0.0
            metrics['relative_error'] = 0.0
            metrics['num_solutions_match'] = True
            return metrics
        
        # Check if number of solutions matches
        num_solutions_match = (len(output) == len(reference))
        metrics['num_solutions_match'] = num_solutions_match
        
        if len(output) == 0:
            metrics['error'] = f"Output is empty but reference has {len(reference)} solutions"
            metrics['pass'] = False
            return metrics
        
        if len(reference) == 0:
            metrics['error'] = f"Reference is empty but output has {len(output)} solutions"
            metrics['pass'] = False
            return metrics
        
        # For each solution, compute errors
        # Each solution is a tuple: (a1, b1, a2, b2, f)
        all_errors = []
        all_mse_values = []
        
        # Try to match solutions (find closest match for each reference solution)
        matched_outputs = []
        matched_references = []
        
        for ref_sol in reference:
            best_match = None
            best_error = float('inf')
            
            for out_sol in output:
                if not isinstance(out_sol, tuple) or len(out_sol) != 5:
                    continue
                
                # Compute error for this pair
                error = (
                    np.abs(out_sol[0] - ref_sol[0]) +
                    np.abs(out_sol[1] - ref_sol[1]) +
                    np.abs(out_sol[2] - ref_sol[2]) +
                    np.abs(out_sol[3] - ref_sol[3]) +
                    np.abs(out_sol[4] - ref_sol[4])
                )
                
                if error < best_error:
                    best_error = error
                    best_match = out_sol
            
            if best_match is not None:
                matched_outputs.append(best_match)
                matched_references.append(ref_sol)
                
                # Compute detailed errors
                diff = np.array([
                    best_match[0] - ref_sol[0],
                    best_match[1] - ref_sol[1],
                    best_match[2] - ref_sol[2],
                    best_match[3] - ref_sol[3],
                    best_match[4] - ref_sol[4]
                ])
                
                mse = np.mean(diff ** 2)
                max_err = np.max(np.abs(diff))
                
                all_errors.append(max_err)
                all_mse_values.append(mse)
        
        if len(matched_outputs) == 0:
            metrics['error'] = "No valid solutions found in output"
            metrics['pass'] = False
            return metrics
        
        # Compute aggregate metrics
        metrics['mse'] = np.mean(all_mse_values) if all_mse_values else float('inf')
        metrics['max_error'] = np.max(all_errors) if all_errors else float('inf')
        
        # Relative error
        ref_norm_sq = sum(sum(s**2 for s in sol) for sol in matched_references)
        ref_norm = np.sqrt(ref_norm_sq) if ref_norm_sq > 0 else 0.0
        
        if ref_norm > 1e-10:
            diff_norm_sq = sum(sum((o - r)**2 for o, r in zip(out_sol, ref_sol)) 
                              for out_sol, ref_sol in zip(matched_outputs, matched_references))
            diff_norm = np.sqrt(diff_norm_sq)
            metrics['relative_error'] = (diff_norm / ref_norm) * 100
        else:
            metrics['relative_error'] = 0.0 if metrics['max_error'] < self.tolerance else 100.0
        
        # Check if pass
        metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_function(self, impl_func, test_case, ref_func):
        """Test solve_shift_and_scale_shared_focal function."""
        x1 = test_case['x1']
        x2 = test_case['x2']
        d1 = test_case['d1']
        d2 = test_case['d2']
        
        try:
            start_time = time.time()
            output = impl_func(x1.copy(), x2.copy(), d1.copy(), d2.copy())
            exec_time = time.time() - start_time
            
            reference = ref_func(x1.copy(), x2.copy(), d1.copy(), d2.copy())
            
            metrics = self.compute_error(output, reference)
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
            
            result = self.test_function(impl_func, test_case, ref_solve_shift_and_scale_shared_focal)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (MSE={result.get('mse', 0):.2e}, "
                          f"max={result.get('max_error', 0):.2e}, "
                          f"rel={result.get('relative_error', 0):.2f}%, "
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
        mse_errors = []
        max_errors = []
        relative_errors = []
        exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                mse_errors.append(result.get('mse', 0))
                max_errors.append(result.get('max_error', 0))
                relative_errors.append(result.get('relative_error', 0))
                exec_times.append(result.get('execution_time', 0))
            else:
                passes.append(False)
        
        # Calculate metrics
        if passes:
            pass_rate = sum(passes) / len(passes) * 100
            summary['pass_rate'] = pass_rate
            summary['total_pass_count'] = sum(passes)
            summary['total_test_count'] = len(passes)
            
            if mse_errors:
                summary['avg_mse'] = sum(mse_errors) / len(mse_errors)
                summary['avg_max'] = sum(max_errors) / len(max_errors)
                summary['avg_relative'] = sum(relative_errors) / len(relative_errors)
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
        
        if 'avg_mse' in summary:
            print(f"  Avg MSE: {summary['avg_mse']:.2e}")
            print(f"  Avg max error: {summary['avg_max']:.2e}")
            print(f"  Avg relative error: {summary['avg_relative']:.2f}%")
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
        
        # Save structured results to test_summary.json
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
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg MSE':<12} {'Avg Max':<12} {'Avg Rel%':<12} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_mse = f"{summary.get('avg_mse', 0):.2e}" if 'avg_mse' in summary else "N/A"
            avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
            avg_rel = f"{summary.get('avg_relative', 0):.2f}%" if 'avg_relative' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_mse:<12} {avg_max:<12} {avg_rel:<12} {avg_time:<12}")
        
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
                    
                    if 'avg_mse' in summary:
                        f.write(f"Average MSE: {summary['avg_mse']:.2e}\n")
                        f.write(f"Average Max Error: {summary['avg_max']:.2e}\n")
                        f.write(f"Average Relative Error: {summary['avg_relative']:.2f}%\n")
                        f.write(f"Average Time: {summary['avg_time']:.4f}s\n")
                    
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<20} {'Pass Rate':<12} {'Avg MSE':<12} {'Avg Max':<12} {'Avg Rel%':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<30} {'ERROR':<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        f.write("-" * 100 + "\n")
                        continue
                    
                    pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
                    avg_mse = f"{summary.get('avg_mse', 0):.2e}" if 'avg_mse' in summary else "N/A"
                    avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
                    avg_rel = f"{summary.get('avg_relative', 0):.2f}%" if 'avg_relative' in summary else "N/A"
                    avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
                    
                    f.write(f"{name:<30} {pass_rate:<12} {avg_mse:<12} {avg_max:<12} {avg_rel:<12} {avg_time:<12}\n")
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
        """
        Save schema-aligned structured test results to `test_summary.json`.
        """
        if not all_summaries:
            return None

        # Determine output path
        if output_path is None:
            output_path = Path(__file__).parent / "test_summary.json"
        else:
            output_path = Path(output_path)

        script_dir = Path(__file__).parent
        project_id = script_dir.parent.name
        unittest_id = script_dir.name.replace("unittest", "")

        payload = {
            "suite": {
                "project_id": project_id,
                "unittest_id": unittest_id,
                "suite_path": f"{project_id}/{script_dir.name}",
                "num_tests_requested": int(self.num_tests),
            },
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": [],
        }

        for summary in all_summaries:
            impl_name = summary.get("implementation", "unknown")

            test_total = int(summary.get("total_test_count", summary.get("total_tests", 0)) or 0)
            test_pass = int(summary.get("total_pass_count", 0) or 0)

            payload["implementations"].append(
                {
                    "name": impl_name,
                    "test_total": test_total,
                    "test_pass": test_pass,
                }
            )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if getattr(self, "verbose", True):
                print(f"\n✓ Structured summary saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            if getattr(self, "verbose", True):
                print(f"\n✗ Error saving structured summary to file: {e}")
            return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for solve_shift_and_scale_shared_focal()')
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
