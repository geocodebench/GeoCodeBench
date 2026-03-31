"""
Test Runner for get_res_scale method
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import os
import sys
import importlib.util
import time
import math
from pathlib import Path
from datetime import datetime, timezone
import json

from reference_implementation import TrainingScheduler as RefTrainingScheduler
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=0):
        """
        Initialize test runner.
        
        Args:
            num_tests: Number of test cases to run
            verbose: Whether to print detailed output
            tolerance: Error tolerance (for integer outputs, should be 0)
        """
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
            
            if not hasattr(module, 'TrainingScheduler'):
                raise AttributeError(f"No TrainingScheduler class found in {filepath}")
            
            return module.TrainingScheduler
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check type
        if not isinstance(output, int):
            metrics['error'] = f"Output is not an integer (got {type(output)})"
            return metrics
        
        if not isinstance(reference, int):
            metrics['error'] = f"Reference is not an integer (got {type(reference)})"
            return metrics
        
        # Absolute difference
        abs_diff = abs(output - reference)
        metrics['abs_diff'] = abs_diff
        
        # Relative error (percentage)
        if reference != 0:
            relative_error = abs(output - reference) / abs(reference) * 100
        else:
            relative_error = 0.0 if output == reference else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass (for integers, should be exact match)
        metrics['pass'] = abs_diff <= self.tolerance
        
        return metrics
    
    def test_get_res_scale(self, impl_class, test_case):
        """Test get_res_scale method."""
        resolution_mode = test_case['resolution_mode']
        reso_scales = test_case['reso_scales']
        reso_level_begin = test_case['reso_level_begin']
        increase_reso_until = test_case['increase_reso_until']
        test_iterations = test_case['test_iterations']
        
        try:
            # Create scheduler instance
            scheduler = impl_class(
                resolution_mode=resolution_mode,
                reso_scales=reso_scales,
                reso_level_begin=reso_level_begin,
                increase_reso_until=increase_reso_until
            )
            
            # Create reference scheduler
            ref_scheduler = RefTrainingScheduler(
                resolution_mode=resolution_mode,
                reso_scales=reso_scales,
                reso_level_begin=reso_level_begin,
                increase_reso_until=increase_reso_until
            )
            
            all_metrics = []
            total_time = 0
            
            # Test each iteration
            for iteration in test_iterations:
                # Reset next_i for each new test to handle stateful behavior
                if resolution_mode == 'freq':
                    ref_scheduler.next_i = 2
                    scheduler.next_i = 2
                
                start_time = time.time()
                output = scheduler.get_res_scale(iteration)
                exec_time = time.time() - start_time
                total_time += exec_time
                
                reference = ref_scheduler.get_res_scale(iteration)
                metrics = self.compute_error(output, reference)
                metrics['execution_time'] = exec_time
                metrics['iteration'] = iteration
                metrics['output'] = output
                metrics['reference'] = reference
                
                all_metrics.append(metrics)
            
            # Aggregate results
            passes = [m.get('pass', False) for m in all_metrics]
            avg_metrics = {
                'all_pass': all(passes),
                'pass_rate': sum(passes) / len(passes) * 100 if passes else 0.0,
                'total_iterations': len(test_iterations),
                'passed_iterations': sum(passes),
                'total_time': total_time,
                'details': all_metrics
            }
            
            return avg_metrics
            
        except Exception as e:
            return {
                'error': str(e),
                'all_pass': False,
                'pass_rate': 0.0,
                'total_iterations': len(test_iterations),
                'passed_iterations': 0,
                'total_time': 0
            }
    
    def test_single_implementation(self, impl_path):
        """Test a single LLM implementation."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # Load implementation
        impl_class = self.load_llm_implementation(impl_path)
        
        if impl_class is None:
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
            
            result = self.test_get_res_scale(impl_class, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('all_pass', False):
                    pass_rate = result.get('pass_rate', 0.0)
                    passed = result.get('passed_iterations', 0)
                    total = result.get('total_iterations', 0)
                    total_time = result.get('total_time', 0)
                    print(f"  ✓ Pass ({passed}/{total} iterations, time={total_time:.6f}s)")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Some iterations failed')}")
                    if 'details' in result:
                        for detail in result['details']:
                            if not detail.get('pass', False):
                                iter_num = detail.get('iteration', '?')
                                output = detail.get('output', '?')
                                reference = detail.get('reference', '?')
                                print(f"    Iteration {iter_num}: output={output}, expected={reference}")
            
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
        
        test_passes = []
        total_iterations = 0
        passed_iterations = 0
        total_times = []
        
        for test_result in all_results:
            result = test_result['result']
            test_passes.append(result.get('all_pass', False))
            total_iterations += result.get('total_iterations', 0)
            passed_iterations += result.get('passed_iterations', 0)
            if 'total_time' in result:
                total_times.append(result['total_time'])
        
        # Calculate metrics
        if test_passes:
            test_pass_rate = sum(test_passes) / len(test_passes) * 100
            summary['test_pass_rate'] = test_pass_rate
            summary['tests_passed'] = sum(test_passes)
            
            if total_times:
                summary['avg_test_time'] = sum(total_times) / len(total_times)
                summary['total_time'] = sum(total_times)
        else:
            test_pass_rate = 0.0
            summary['test_pass_rate'] = 0.0
            summary['tests_passed'] = 0
        
        # Overall pass rate based on iterations
        if total_iterations > 0:
            iteration_pass_rate = passed_iterations / total_iterations * 100
        else:
            iteration_pass_rate = 0.0
        
        summary['iteration_pass_rate'] = iteration_pass_rate
        summary['total_iterations'] = total_iterations
        summary['passed_iterations'] = passed_iterations
        summary['overall_pass_rate'] = test_pass_rate
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Summary for {summary['implementation']}:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Tests passed: {summary.get('tests_passed', 0)}/{summary['total_tests']} ({summary.get('test_pass_rate', 0.0):.1f}%)")
        print(f"  Total iterations: {summary.get('total_iterations', 0)}")
        print(f"  Iterations passed: {summary.get('passed_iterations', 0)}/{summary.get('total_iterations', 0)} ({summary.get('iteration_pass_rate', 0.0):.1f}%)")
        
        if 'avg_test_time' in summary:
            print(f"  Avg test time: {summary['avg_test_time']:.6f}s")
            print(f"  Total time: {summary.get('total_time', 0):.6f}s")
        
        print(f"  Overall: {summary.get('overall_pass_rate', 0.0):.1f}%")
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
        self.save_summary_to_file(all_summaries)
        
        return all_summaries

    def save_summary_to_file(self, all_summaries, output_path=None):
        """
        Save structured per-implementation pass/total summary.

        Output is aligned with repository-level `schema.json`.
        """
        if not all_summaries:
            return None

        script_dir = Path(__file__).parent
        project_id = script_dir.parent.name
        unittest_id = script_dir.name.replace("unittest", "")
        suite_path = f"{project_id}/{script_dir.name}"

        payload = {
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
            impl_name = summary.get("implementation", "")
            test_total = summary.get("total_test_count", summary.get("total_tests", 0))
            test_pass = summary.get("total_pass_count", summary.get("tests_passed", 0))

            payload["implementations"].append(
                {
                    "name": impl_name,
                    "test_total": int(test_total),
                    "test_pass": int(test_pass),
                }
            )

        if output_path is None:
            output_path = script_dir / "test_summary.json"
        else:
            output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

        return str(output_path)
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Implementation':<25} {'Test Pass':<15} {'Iter Pass':<15} {'Time (avg)':<15}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<15} {'0.0%':<15} {'N/A':<15}")
                continue
            
            test_pass = f"{summary.get('test_pass_rate', 0.0):.1f}%"
            iter_pass_count = f"{summary.get('passed_iterations', 0)}/{summary.get('total_iterations', 0)}"
            avg_time = f"{summary.get('avg_test_time', 0):.6f}s" if 'avg_test_time' in summary else "N/A"
            
            print(f"{name:<25} {test_pass:<15} {iter_pass_count:<15} {avg_time:<15}")
        
        print("-" * 100)
        
        # Print ranking
        print(f"\n{'OVERALL RANKING':<25} {'Pass Rate':<15} {'Tests Passed':<15}")
        print("-" * 57)
        sorted_summaries = sorted(all_summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
        for i, summary in enumerate(sorted_summaries, 1):
            name = summary['implementation'][:23]
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            tests_passed = summary.get('tests_passed', 0)
            total_tests = summary.get('total_tests', 0)
            count_str = f"{tests_passed}/{total_tests}"
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



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for get_res_scale')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=int, default=0,
                       help='Error tolerance for pass/fail (default: 0)')
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

