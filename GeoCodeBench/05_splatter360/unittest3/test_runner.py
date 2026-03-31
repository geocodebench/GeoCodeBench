"""
Test Runner for Rotation Matrix Functions
Supports batch testing of multiple LLM implementations.
"""

import torch
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

from reference_implementation import get_rotation_x as ref_rotation_x
from reference_implementation import get_rotation_y as ref_rotation_y
from reference_implementation import get_rotation_z as ref_rotation_z
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-6):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.test_generator = TestDataGenerator()
        self.test_cases = self.test_generator.generate_test_suite(num_tests)
        self.function_names = ['get_rotation_x', 'get_rotation_y', 'get_rotation_z']
        self.ref_functions = {
            'get_rotation_x': ref_rotation_x,
            'get_rotation_y': ref_rotation_y,
            'get_rotation_z': ref_rotation_z,
        }
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if all required functions exist
            functions = {}
            for func_name in self.function_names:
                if not hasattr(module, func_name):
                    raise AttributeError(f"No {func_name} function found in {filepath}")
                functions[func_name] = getattr(module, func_name)
            
            return functions
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
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
        
        # Check if shape is [3, 3]
        if output.shape != (3, 3):
            metrics['error'] = f"Expected shape [3, 3], got {output.shape}"
            return metrics
        
        # Convert to float for comparison
        output_f = output.float().cpu()
        reference_f = reference.float().cpu()
        
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
    
    def test_rotation_function(self, impl_func, ref_func, test_case, func_name):
        """Test a single rotation function with all angles in test case."""
        angles = test_case['angles']
        results = []
        
        for angle in angles:
            try:
                start_time = time.time()
                output = impl_func(angle, device='cpu')
                exec_time = time.time() - start_time
                
                reference = ref_func(angle, device='cpu')
                metrics = self.compute_error(output, reference)
                metrics['execution_time'] = exec_time
                metrics['angle'] = angle
            except Exception as e:
                metrics = {
                    'error': str(e),
                    'pass': False,
                    'execution_time': 0,
                    'angle': angle,
                }
            
            results.append(metrics)
        
        return results
    
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
        
        all_results = []
        
        # Run all test cases for all functions
        for func_name in self.function_names:
            impl_func = impl_functions[func_name]
            ref_func = self.ref_functions[func_name]
            
            if self.verbose:
                print(f"\n--- Testing {func_name} ---")
            
            for test_idx, test_case in enumerate(self.test_cases):
                if self.verbose:
                    print(f"\nTest {test_idx+1}/{len(self.test_cases)}: {test_case['description']}")
                
                results = self.test_rotation_function(impl_func, ref_func, test_case, func_name)
                
                # Count passes and failures
                passes = sum(1 for r in results if r.get('pass', False))
                total = len(results)
                
                if self.verbose:
                    if passes == total:
                        avg_error = sum(r.get('max_error', 0) for r in results) / total if total > 0 else 0
                        avg_time = sum(r.get('execution_time', 0) for r in results) / total if total > 0 else 0
                        print(f"  ✓ All angles passed ({passes}/{total})")
                        print(f"    Avg max error: {avg_error:.2e}, Avg time: {avg_time:.6f}s")
                    else:
                        print(f"  ✗ Failed: {total - passes}/{total} angles failed")
                        # Show first few failures
                        failures = [r for r in results if not r.get('pass', False)]
                        for i, failure in enumerate(failures[:3]):
                            angle = failure.get('angle', 'N/A')
                            error_msg = failure.get('error', 'Error exceeds tolerance')
                            print(f"    - Angle {angle}°: {error_msg}")
                            if 'max_error' in failure:
                                print(f"      Max error: {failure['max_error']:.2e} (tolerance: {self.tolerance:.2e})")
                
                # Store results
                for result in results:
                    all_results.append({
                        'function': func_name,
                        'test_idx': test_idx,
                        'description': test_case['description'],
                        'result': result
                    })
        
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
        
        # Overall statistics
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
        
        # Per-function statistics
        function_stats = {}
        for func_name in self.function_names:
            func_results = [r for r in all_results if r['function'] == func_name]
            func_passes = [r['result'].get('pass', False) for r in func_results]
            
            function_stats[func_name] = {
                'total': len(func_passes),
                'passed': sum(func_passes),
                'pass_rate': sum(func_passes) / len(func_passes) * 100 if func_passes else 0.0,
            }
        
        summary['function_stats'] = function_stats
        
        # Calculate overall metrics
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
        print(f"  Total tests: {summary['total_tests']} (across all functions)")
        print(f"  Pass rate: {summary.get('pass_rate', 0.0):.1f}%")
        
        # Per-function stats
        if 'function_stats' in summary:
            print(f"\n  Per-function results:")
            for func_name, stats in summary['function_stats'].items():
                print(f"    {func_name}: {stats['passed']}/{stats['total']} passed ({stats['pass_rate']:.1f}%)")
        
        if 'avg_l1' in summary:
            print(f"\n  Avg L1 error: {summary['avg_l1']:.2e}")
            print(f"  Avg L2 error: {summary['avg_l2']:.2e}")
            print(f"  Avg max error: {summary['avg_max']:.2e}")
            print(f"  Avg time: {summary['avg_time']:.6f}s")
        
        pass_count = summary.get('total_pass_count', 0)
        test_count = summary.get('total_test_count', 0)
        print(f"\n  Overall: {summary.get('overall_pass_rate', 0.0):.1f}% ({pass_count}/{test_count} tests passed)")
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
        print(f"Running {self.num_tests} test cases per function ({len(self.function_names)} functions)\n")
        
        # Test each implementation
        all_summaries = []
        for impl_file in impl_files:
            summary = self.test_single_implementation(str(impl_file))
            all_summaries.append(summary)
        
        # Print comparison
        self.print_comparison(all_summaries)
        
        # Save results to file
        self.save_results_to_file(all_summaries)
        
        # Save structured results to JSON
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
            avg_time = f"{summary.get('avg_time', 0):.6f}s" if 'avg_time' in summary else "N/A"
            
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
                f.write("="*100 + "\n\n")
                
                # Write detailed results for each implementation
                for summary in all_summaries:
                    f.write("="*80 + "\n")
                    f.write(f"Implementation: {summary['implementation']}\n")
                    f.write("="*80 + "\n")
                    
                    # Check if there was an error loading
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"ERROR: {summary['error']}\n")
                        f.write("Pass Rate: 0.0%\n")
                        f.write("\n")
                        continue
                    
                    # Write test summary
                    f.write(f"Total tests: {summary.get('total_tests', 0)}\n")
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    f.write(f"Passed tests: {pass_count}\n")
                    f.write(f"Failed tests: {max(test_count - pass_count, 0)}\n")
                    f.write(f"Pass rate: {summary.get('pass_rate', 0.0):.1f}%\n\n")
                    
                    # Write detailed test results if available
                    if 'results' in summary:
                        f.write("Detailed Test Results:\n")
                        f.write("-" * 80 + "\n")
                        for i, test_result in enumerate(summary['results'], 1):
                            test_desc = test_result.get('description', f'Test {i}')
                            func_name = test_result.get('function', 'unknown')
                            result = test_result.get('result', {})
                            f.write(f"\nTest {i}: {test_desc}\n")
                            f.write(f"  Function: {func_name}\n")
                            
                            if result.get('pass', False):
                                f.write("  ✓ Success\n")
                                f.write(f"    Angle: {result.get('angle', 'N/A')}\n")
                                f.write(f"    Execution time: {result.get('execution_time', 0):.4f}s\n")
                                f.write(f"    L1 error: {result.get('l1_error', 0):.2e}\n")
                                f.write(f"    L2 error: {result.get('l2_error', 0):.2e}\n")
                                f.write(f"    Max error: {result.get('max_error', 0):.2e}\n")
                            else:
                                f.write(f"  ✗ Failed: {result.get('error', 'Error exceeds tolerance')}\n")
                                f.write(f"    Angle: {result.get('angle', 'N/A')}\n")
                                if result.get('execution_time', 0) > 0:
                                    f.write(f"    Execution time: {result['execution_time']:.4f}s\n")
                        
                        f.write("\n" + "-" * 80 + "\n\n")
                    
                    # Write statistics summary if available
                    if summary.get('total_pass_count', 0) > 0:
                        f.write("Summary Statistics:\n")
                        if 'avg_time' in summary:
                            f.write(f"  Average execution time: {summary['avg_time']:.6f}s\n")
                        if 'avg_l1' in summary:
                            f.write(f"  Average L1 error: {summary['avg_l1']:.2e}\n")
                        if 'avg_l2' in summary:
                            f.write(f"  Average L2 error: {summary['avg_l2']:.2e}\n")
                        if 'avg_max' in summary:
                            f.write(f"  Average max error: {summary['avg_max']:.2e}\n")
                    if 'function_stats' in summary:
                        f.write("  Per-function pass rates:\n")
                        for func_name, stats in summary['function_stats'].items():
                            f.write(f"    {func_name}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1f}%)\n")
                    
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg Max':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation'][:23]
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        continue
                    
                    pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
                    avg_l1 = f"{summary.get('avg_l1', 0):.2e}" if 'avg_l1' in summary else "N/A"
                    avg_l2 = f"{summary.get('avg_l2', 0):.2e}" if 'avg_l2' in summary else "N/A"
                    avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
                    avg_time = f"{summary.get('avg_time', 0):.6f}s" if 'avg_time' in summary else "N/A"

                    f.write(f"{name:<25} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_max:<12} {avg_time:<12}\n")
                
                f.write("-" * 100 + "\n")
                
                # Write ranking
                f.write(f"\n{'OVERALL RANKING':<25} {'Pass Rate':<15} {'Pass Count':<15}\n")
                f.write("-" * 57 + "\n")
                sorted_summaries = sorted(all_summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
                for i, summary in enumerate(sorted_summaries, 1):
                    name = summary['implementation'][:23]
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
        """Save structured per-implementation test counts to `test_summary.json`."""
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

        payload = {
            "suite": {
                "project_id": project_id,
                "unittest_id": unittest_id,
                "suite_path": suite_path,
                "num_tests_requested": self.num_tests,
            },
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": [],
        }

        for summary in all_summaries:
            payload["implementations"].append(
                {
                    "name": summary.get("implementation", "unknown"),
                    "test_total": int(summary.get("total_test_count", 0)),
                    "test_pass": int(summary.get("total_pass_count", 0)),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(output_path)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for rotation matrix functions')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Error tolerance for pass/fail (default: 1e-6)')
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
