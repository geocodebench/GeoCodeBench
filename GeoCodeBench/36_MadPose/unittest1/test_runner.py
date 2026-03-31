"""
Test Runner for solve_shift_and_scale() function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import numpy as np
import os
import sys
import importlib.util
import time
import json
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timezone

from reference_implementation import solve_shift_and_scale as ref_solve_shift_and_scale
from test_generator import TestDataGenerator


def _execute_impl_once(impl_path, x1, x2, d1, d2, result_queue):
    """Execute one implementation call in an isolated process."""
    try:
        spec = importlib.util.spec_from_file_location("llm_impl_timeout", impl_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "solve_shift_and_scale"):
            raise AttributeError(f"No solve_shift_and_scale function found in {impl_path}")

        impl_func = module.solve_shift_and_scale
        start_time = time.time()
        output = impl_func(x1, x2, d1, d2)
        exec_time = time.time() - start_time
        result_queue.put({
            "ok": True,
            "output": output,
            "execution_time": exec_time,
        })
    except Exception as e:
        result_queue.put({
            "ok": False,
            "error": str(e),
            "execution_time": 0.0,
        })


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-5, timeout_per_test=3.0):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.timeout_per_test = timeout_per_test
        self.test_generator = TestDataGenerator()
        self.test_cases = self.test_generator.generate_test_suite(num_tests)
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'solve_shift_and_scale'):
                raise AttributeError(f"No solve_shift_and_scale function found in {filepath}")
            
            return module.solve_shift_and_scale
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
        
        # If both are empty, consider it a match
        if len(output) == 0 and len(reference) == 0:
            metrics['l1_error'] = 0.0
            metrics['l2_error'] = 0.0
            metrics['max_error'] = 0.0
            metrics['relative_error'] = 0.0
            metrics['pass'] = True
            return metrics
        
        # If one is empty and the other is not, it's a failure
        if len(output) == 0 or len(reference) == 0:
            metrics['error'] = f"Solution count mismatch: {len(output)} vs {len(reference)}"
            metrics['pass'] = False
            return metrics
        
        # Check if all elements are tuples of length 4
        for i, sol in enumerate(output):
            if not isinstance(sol, tuple) or len(sol) != 4:
                metrics['error'] = f"Output solution {i} is not a tuple of length 4 (got {type(sol)}, length {len(sol) if isinstance(sol, tuple) else 'N/A'})"
                return metrics
        
        for i, sol in enumerate(reference):
            if not isinstance(sol, tuple) or len(sol) != 4:
                metrics['error'] = f"Reference solution {i} is not a tuple of length 4"
                return metrics
        
        # Find best matching solution pairs
        # For each reference solution, find the closest output solution
        matched_pairs = []
        used_output_indices = set()
        
        for ref_idx, ref_sol in enumerate(reference):
            best_match_idx = None
            best_match_error = float('inf')
            
            for out_idx, out_sol in enumerate(output):
                if out_idx in used_output_indices:
                    continue
                
                # Compute error for this pair
                ref_arr = np.array(ref_sol)
                out_arr = np.array(out_sol)
                
                # L2 error
                error = np.linalg.norm(ref_arr - out_arr)
                
                if error < best_match_error:
                    best_match_error = error
                    best_match_idx = out_idx
            
            if best_match_idx is not None:
                matched_pairs.append((ref_idx, best_match_idx))
                used_output_indices.add(best_match_idx)
        
        # Compute errors for matched pairs
        if len(matched_pairs) == 0:
            metrics['error'] = "No matching solutions found"
            metrics['pass'] = False
            return metrics
        
        l1_errors = []
        l2_errors = []
        max_errors = []
        
        for ref_idx, out_idx in matched_pairs:
            ref_sol = reference[ref_idx]
            out_sol = output[out_idx]
            
            ref_arr = np.array(ref_sol)
            out_arr = np.array(out_sol)
            
            # L1 error
            l1 = np.mean(np.abs(ref_arr - out_arr))
            l1_errors.append(l1)
            
            # L2 error (MSE)
            l2 = np.sqrt(np.mean((ref_arr - out_arr) ** 2))
            l2_errors.append(l2)
            
            # Max error
            max_err = np.max(np.abs(ref_arr - out_arr))
            max_errors.append(max_err)
        
        # Average errors
        metrics['l1_error'] = np.mean(l1_errors) if l1_errors else float('inf')
        metrics['l2_error'] = np.mean(l2_errors) if l2_errors else float('inf')
        metrics['max_error'] = np.max(max_errors) if max_errors else float('inf')
        
        # Relative error
        ref_norms = [np.linalg.norm(np.array(sol)) for sol in reference[:len(matched_pairs)]]
        total_ref_norm = sum(ref_norms)
        
        if total_ref_norm > 1e-10:
            out_diff_norms = [np.linalg.norm(np.array(reference[ref_idx]) - np.array(output[out_idx])) 
                            for ref_idx, out_idx in matched_pairs]
            total_diff_norm = sum(out_diff_norms)
            relative_error = (total_diff_norm / total_ref_norm) * 100
        else:
            relative_error = 0.0 if metrics['max_error'] < self.tolerance else 100.0
        
        metrics['relative_error'] = relative_error
        metrics['solution_count_match'] = len(matched_pairs) == len(reference) and len(matched_pairs) == len(output)
        metrics['matched_solutions'] = len(matched_pairs)
        metrics['reference_solutions'] = len(reference)
        metrics['output_solutions'] = len(output)
        
        # Check if pass
        metrics['pass'] = metrics['max_error'] < self.tolerance and metrics['solution_count_match']
        
        return metrics
    
    def test_solve_shift_and_scale(self, impl_func, test_case, impl_path=None):
        """Test solve_shift_and_scale function."""
        x1 = test_case['x1']
        x2 = test_case['x2']
        d1 = test_case['d1']
        d2 = test_case['d2']
        
        try:
            if self.timeout_per_test is not None and impl_path is not None:
                ctx = mp.get_context("spawn")
                result_queue = ctx.Queue(maxsize=1)
                process = ctx.Process(
                    target=_execute_impl_once,
                    args=(impl_path, x1, x2, d1, d2, result_queue),
                )
                process.start()
                process.join(self.timeout_per_test)

                if process.is_alive():
                    process.terminate()
                    process.join()
                    metrics = {
                        'error': f'Timeout after {self.timeout_per_test:.2f}s',
                        'pass': False,
                        'execution_time': self.timeout_per_test,
                        'timed_out': True
                    }
                    return metrics

                if result_queue.empty():
                    metrics = {
                        'error': 'Implementation process exited without result',
                        'pass': False,
                        'execution_time': 0
                    }
                    return metrics

                worker_result = result_queue.get()
                if not worker_result.get("ok", False):
                    metrics = {
                        'error': worker_result.get("error", "Unknown error"),
                        'pass': False,
                        'execution_time': worker_result.get("execution_time", 0)
                    }
                    return metrics

                output = worker_result["output"]
                exec_time = worker_result.get("execution_time", 0.0)
            else:
                start_time = time.time()
                output = impl_func(x1, x2, d1, d2)
                exec_time = time.time() - start_time
            
            reference = ref_solve_shift_and_scale(x1, x2, d1, d2)
            
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
            
            result = self.test_solve_shift_and_scale(impl_func, test_case, impl_path=impl_path)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (L1={result.get('l1_error', 0):.2e}, L2={result.get('l2_error', 0):.2e}, "
                          f"max={result.get('max_error', 0):.2e}, time={result.get('execution_time', 0):.4f}s)")
                    if 'matched_solutions' in result:
                        print(f"    Solutions: {result.get('matched_solutions', 0)} matched, "
                              f"ref={result.get('reference_solutions', 0)}, "
                              f"out={result.get('output_solutions', 0)}")
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
        l1_errors = []
        l2_errors = []
        max_errors = []
        exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                l1_errors.append(result.get('l1_error', 0))
                l2_errors.append(result.get('l2_error', 0))
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
                    
                    # Write overall statistics
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    overall_rate = summary.get('overall_pass_rate', 0.0)
                    f.write(f"Overall Average Pass Rate: {overall_rate:.1f}% ({pass_count}/{test_count} tests passed)\n")
                    
                    if 'avg_l1' in summary:
                        f.write(f"Average L1 error: {summary['avg_l1']:.2e}\n")
                        f.write(f"Average L2 error: {summary['avg_l2']:.2e}\n")
                        f.write(f"Average max error: {summary['avg_max']:.2e}\n")
                        f.write(f"Average execution time: {summary['avg_time']:.4f}s\n")
                    
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<20} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg Max':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<30} {'ERROR':<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        f.write("-" * 100 + "\n")
                        continue
                    
                    pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
                    avg_l1 = f"{summary.get('avg_l1', 0):.2e}" if 'avg_l1' in summary else "N/A"
                    avg_l2 = f"{summary.get('avg_l2', 0):.2e}" if 'avg_l2' in summary else "N/A"
                    avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
                    avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
                    
                    f.write(f"{name:<30} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_max:<12} {avg_time:<12}\n")
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

        Schema is defined in repository root `schema.json`.
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

            # Prefer the runner's explicit counters.
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
    
    parser = argparse.ArgumentParser(description='Test runner for solve_shift_and_scale()')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                       help='Error tolerance for pass/fail (default: 1e-5)')
    parser.add_argument('--timeout-per-test', type=float, default=3.0,
                       help='Timeout in seconds for each single test call (default: 3.0)')
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
        tolerance=args.tolerance,
        timeout_per_test=args.timeout_per_test
    )
    
    # Run tests
    results = runner.batch_test(str(impl_dir))
    
    return results


if __name__ == '__main__':
    main()
