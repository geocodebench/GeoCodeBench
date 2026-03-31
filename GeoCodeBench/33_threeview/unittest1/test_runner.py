"""
Test Runner for run_test_dE23dr12, run_test_dE23dt12, run_test_dE23dr13 functions
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import numpy as np
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

# Import reference implementation and test generator
from reference_implementation import (
    run_test_dE23dr12 as ref_run_test_dE23dr12,
    run_test_dE23dt12 as ref_run_test_dE23dt12,
    run_test_dE23dr13 as ref_run_test_dE23dr13,
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
            
            # Check for all three functions
            functions = {}
            for func_name in ['run_test_dE23dr12', 'run_test_dE23dt12', 'run_test_dE23dr13']:
                if not hasattr(module, func_name):
                    raise AttributeError(f"No {func_name} function found in {filepath}")
                functions[func_name] = getattr(module, func_name)
            
            return functions
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference (both are scalars)."""
        metrics = {}
        
        # Check if both are scalars
        if not isinstance(output, (int, float, np.number)):
            metrics['error'] = f"Output is not a scalar (got {type(output)})"
            return metrics
        
        if not isinstance(reference, (int, float, np.number)):
            metrics['error'] = f"Reference is not a scalar (got {type(reference)})"
            return metrics
        
        # Convert to float
        output = float(output)
        reference = float(reference)
        
        # Absolute error
        abs_error = abs(output - reference)
        metrics['abs_error'] = abs_error
        
        # Relative error (percentage)
        if abs(reference) > 1e-10:
            relative_error = (abs_error / abs(reference)) * 100
        else:
            relative_error = 100.0 if abs_error > self.tolerance else 0.0
        metrics['relative_error'] = relative_error
        
        # Check if pass
        metrics['pass'] = abs_error < self.tolerance
        
        return metrics
    
    def test_function(self, impl_func, test_case, ref_func):
        """Test a single function call."""
        func_name = test_case['function']
        i = test_case['i']
        eps = test_case['eps']
        
        try:
            # Set random seed for reproducibility (use test index as seed)
            test_seed = self.test_generator.seed + test_case.get('test_idx', 0)
            np.random.seed(test_seed)
            
            start_time = time.time()
            output = impl_func(i=i, eps=eps)
            exec_time = time.time() - start_time
            
            # Reset seed to get same random numbers for reference
            np.random.seed(test_seed)
            reference = ref_func(i=i, eps=eps)
            
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
        impl_functions = self.load_llm_implementation(impl_path)
        
        if impl_functions is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': 0
            }
        
        # Map function names to reference functions
        ref_functions = {
            'run_test_dE23dr12': ref_run_test_dE23dr12,
            'run_test_dE23dt12': ref_run_test_dE23dt12,
            'run_test_dE23dr13': ref_run_test_dE23dr13
        }
        
        all_results = []
        
        # Run all test cases
        for i, test_case in enumerate(self.test_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(self.test_cases)}: {test_case['description']}")
            
            func_name = test_case['function']
            impl_func = impl_functions[func_name]
            ref_func = ref_functions[func_name]
            
            # Add test index to test_case for seed generation
            test_case['test_idx'] = i
            result = self.test_function(impl_func, test_case, ref_func)
            
            test_result = {
                'test_idx': i,
                'function': func_name,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (abs_err={result.get('abs_error', 0):.2e}, "
                          f"rel_err={result.get('relative_error', 0):.2f}%, "
                          f"time={result.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
                    if 'abs_error' in result:
                        print(f"    Abs error: {result['abs_error']:.2e} (tolerance: {self.tolerance:.2e})")
            
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
        
        # Group by function
        func_results = {
            'run_test_dE23dr12': [],
            'run_test_dE23dt12': [],
            'run_test_dE23dr13': []
        }
        
        for test_result in all_results:
            func_name = test_result['function']
            if func_name in func_results:
                func_results[func_name].append(test_result)
        
        # Compute per-function statistics
        for func_name, results in func_results.items():
            if not results:
                continue
            
            passes = []
            abs_errors = []
            exec_times = []
            
            for test_result in results:
                result = test_result['result']
                if result.get('pass', False):
                    passes.append(True)
                    abs_errors.append(result.get('abs_error', 0))
                    exec_times.append(result.get('execution_time', 0))
                else:
                    passes.append(False)
            
            if passes:
                pass_rate = sum(passes) / len(passes) * 100
                summary[f'{func_name}_pass_rate'] = pass_rate
                summary[f'{func_name}_pass_count'] = sum(passes)
                summary[f'{func_name}_test_count'] = len(passes)
                
                if abs_errors:
                    summary[f'{func_name}_avg_abs'] = sum(abs_errors) / len(abs_errors)
                    summary[f'{func_name}_avg_time'] = sum(exec_times) / len(exec_times)
        
        # Overall statistics
        all_passes = []
        all_abs_errors = []
        all_exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                all_passes.append(True)
                all_abs_errors.append(result.get('abs_error', 0))
                all_exec_times.append(result.get('execution_time', 0))
            else:
                all_passes.append(False)
        
        if all_passes:
            summary['pass_rate'] = sum(all_passes) / len(all_passes) * 100
            summary['total_pass_count'] = sum(all_passes)
            summary['total_test_count'] = len(all_passes)
            
            if all_abs_errors:
                summary['avg_abs'] = sum(all_abs_errors) / len(all_abs_errors)
                summary['avg_time'] = sum(all_exec_times) / len(all_exec_times)
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
        
        if 'avg_abs' in summary:
            print(f"  Avg absolute error: {summary['avg_abs']:.2e}")
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
        # Save structured results for cross-run aggregation
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
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg Abs Err':<12} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_abs = f"{summary.get('avg_abs', 0):.2e}" if 'avg_abs' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_abs:<12} {avg_time:<12}")
        
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
                    
                    # Write per-function statistics
                    func_names = ['run_test_dE23dr12', 'run_test_dE23dt12', 'run_test_dE23dr13']
                    for func_name in func_names:
                        pass_rate_key = f'{func_name}_pass_rate'
                        if pass_rate_key in summary:
                            f.write(f"{func_name}:\n")
                            f.write(f"  Pass rate: {summary[pass_rate_key]:.1f}%\n")
                            if f'{func_name}_avg_abs' in summary:
                                f.write(f"  Avg absolute error: {summary[f'{func_name}_avg_abs']:.2e}\n")
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
                f.write(f"{'Implementation':<20} {'Function':<20} {'Pass Rate':<12} {'Avg Abs Err':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<30} {'ERROR':<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12}\n")
                        f.write(f"{'  → AVERAGE':<20} {'(0/0)':<20} {'0.0%':<12} {'':<12} {'':<12}\n")
                        f.write("-" * 100 + "\n")
                        continue
                    
                    # Write per-function rows
                    func_names = ['run_test_dE23dr12', 'run_test_dE23dt12', 'run_test_dE23dr13']
                    for func_name in func_names:
                        pass_rate_key = f'{func_name}_pass_rate'
                        if pass_rate_key in summary:
                            pass_rate = f"{summary[pass_rate_key]:.1f}%"
                            
                            if f'{func_name}_avg_abs' in summary:
                                avg_abs = f"{summary[f'{func_name}_avg_abs']:.2e}"
                                avg_time = f"{summary[f'{func_name}_avg_time']:.4f}s"
                            else:
                                avg_abs = "N/A"
                                avg_time = "N/A"
                            
                            f.write(f"{name:<30} {func_name:<30} {pass_rate:<12} {avg_abs:<12} {avg_time:<12}\n")
                            name = ""
                    
                    overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    count_info = f"({pass_count}/{test_count})"
                    f.write(f"{'  → AVERAGE':<20} {count_info:<20} {overall_rate:<12} {'':<12} {'':<12}\n")
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
        """Save structured test summary JSON aligned to schema.json."""
        if not all_summaries:
            return None

        script_dir = Path(__file__).parent
        project_id = script_dir.parent.name
        unittest_id = script_dir.name.replace("unittest", "")
        suite_path = f"{project_id}/{script_dir.name}"

        if output_path is None:
            output_path = script_dir / "test_summary.json"
        else:
            output_path = Path(output_path)

        data = {
            "suite": {
                "project_id": project_id,
                "unittest_id": unittest_id,
                "suite_path": suite_path,
                "num_tests_requested": int(self.num_tests),
            },
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": [],
        }

        for summary in all_summaries:
            data["implementations"].append(
                {
                    "name": summary.get("implementation", ""),
                    "test_total": int(summary.get("total_test_count", 0)),
                    "test_pass": int(summary.get("total_pass_count", 0)),
                }
            )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"\n✗ Error saving test summary JSON: {e}")
            return None

        return str(output_path)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for run_test_dE23dr12, run_test_dE23dt12, run_test_dE23dr13')
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
