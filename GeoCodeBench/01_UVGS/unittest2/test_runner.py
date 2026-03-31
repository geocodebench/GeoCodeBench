"""
Test Runner for equirectangular_unwrap_topK_opacity function
Supports batch testing of multiple LLM implementations.
"""

import numpy as np
import os
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from reference_implementation import equirectangular_unwrap_topK_opacity as reference_impl
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True):
        """
        Initialize the test runner.
        
        Args:
            num_tests (int): Number of test cases to run.
            verbose (bool): Whether to print detailed output.
        """
        self.num_tests = num_tests
        self.verbose = verbose
        self.test_generator = TestDataGenerator()
        self.test_cases = self.test_generator.generate_test_suite(num_tests)
    
    def load_llm_implementation(self, filepath):
        """
        Dynamically load an LLM implementation from a file.
        
        Args:
            filepath (str): Path to the implementation file.
            
        Returns:
            function: The loaded function.
        """
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'equirectangular_unwrap_topK_opacity'):
                return module.equirectangular_unwrap_topK_opacity
            else:
                raise AttributeError(f"No 'equirectangular_unwrap_topK_opacity' function found in {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_metrics(self, output, reference):
        """
        Compute comparison metrics between output and reference.
        
        Args:
            output (np.array): Output from LLM implementation (height, width, K).
            reference (np.array): Reference output (height, width, K).
            
        Returns:
            dict: Dictionary of metrics.
        """
        metrics = {}
        
        # Check shape
        metrics['shape_match'] = output.shape == reference.shape
        
        if not metrics['shape_match']:
            metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
            return metrics
        
        # Exact match for all K channels
        metrics['exact_match'] = np.allclose(output, reference, rtol=1e-9, atol=1e-9)
        
        # Pixel-wise exact match (all K channels must match)
        pixel_exact_match = np.all(output == reference, axis=2)
        metrics['exact_match_rate'] = np.mean(pixel_exact_match) * 100
        
        # Top-K accuracy: at least one of the K channels matches
        # For each pixel, check if any channel in output matches any channel in reference
        height, width, K = output.shape
        topk_matches = np.zeros((height, width), dtype=bool)
        
        for i in range(height):
            for j in range(width):
                # Check if there's any overlap between output[i,j,:] and reference[i,j,:]
                output_set = set(output[i, j, :])
                reference_set = set(reference[i, j, :])
                if len(output_set & reference_set) > 0:
                    topk_matches[i, j] = True
        
        metrics['topk_accuracy'] = np.mean(topk_matches) * 100
        
        # Mean absolute error
        metrics['mae'] = np.mean(np.abs(output.astype(float) - reference.astype(float)))
        
        # Root mean square error
        metrics['rmse'] = np.sqrt(np.mean((output.astype(float) - reference.astype(float)) ** 2))
        
        # Max absolute error
        metrics['max_error'] = np.max(np.abs(output.astype(float) - reference.astype(float)))
        
        # Non-zero pixels comparison
        non_zero_ref = np.any(reference != 0, axis=2)
        non_zero_out = np.any(output != 0, axis=2)
        
        metrics['non_zero_ref_count'] = np.sum(non_zero_ref)
        metrics['non_zero_out_count'] = np.sum(non_zero_out)
        
        if np.sum(non_zero_ref) > 0:
            non_zero_matches = pixel_exact_match[non_zero_ref]
            metrics['non_zero_exact_match'] = np.mean(non_zero_matches) * 100
        else:
            metrics['non_zero_exact_match'] = 100.0
        
        return metrics
    
    def run_single_test(self, impl_func, test_case, test_idx):
        """
        Run a single test case.
        
        Args:
            impl_func (function): Implementation to test.
            test_case (dict): Test case data.
            test_idx (int): Test case index.
            
        Returns:
            dict: Test results.
        """
        try:
            start_time = time.time()
            
            # Run the implementation
            output = impl_func(
                test_case['points'],
                test_case['opacity'],
                test_case['height'],
                test_case['width'],
                test_case['K']
            )
            
            execution_time = time.time() - start_time
            
            # Run reference implementation
            reference = reference_impl(
                test_case['points'],
                test_case['opacity'],
                test_case['height'],
                test_case['width'],
                test_case['K']
            )
            
            # Compute metrics
            metrics = self.compute_metrics(output, reference)
            metrics['execution_time'] = execution_time
            
            # Mark as success if no errors
            if 'error' in metrics:
                metrics['success'] = False
            else:
                metrics['success'] = True
            
        except Exception as e:
            metrics = {
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
        
        return metrics
    
    def test_implementation(self, impl_path):
        """
        Test a single LLM implementation against all test cases.
        
        Args:
            impl_path (str): Path to the implementation file.
            
        Returns:
            dict: Test results for all test cases.
        """
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
                'test_results': []
            }
        
        # Run all test cases
        test_results = []
        for i, test_case in enumerate(self.test_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(self.test_cases)}: {test_case['description']}")
            
            result = self.run_single_test(impl_func, test_case, i)
            result['test_description'] = test_case['description']
            test_results.append(result)
            
            if self.verbose:
                if result['success']:
                    print(f"  ✓ Success")
                    print(f"    Execution time: {result['execution_time']:.4f}s")
                    print(f"    Exact match: {result.get('exact_match_rate', 0):.2f}%")
                    print(f"    Top-K accuracy: {result.get('topk_accuracy', 0):.2f}%")
                    print(f"    MAE: {result.get('mae', 0):.6f}")
                    print(f"    RMSE: {result.get('rmse', 0):.6f}")
                else:
                    print(f"  ✗ Failed: {result['error']}")
                    if result.get('execution_time', 0) > 0:
                        print(f"    Execution time: {result['execution_time']:.4f}s")
        
        # Compute summary statistics
        successful_tests = [r for r in test_results if r['success']]
        
        summary = {
            'implementation': impl_name,
            'total_tests': len(test_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(test_results) - len(successful_tests),
            'test_results': test_results
        }
        
        if successful_tests:
            summary['avg_execution_time'] = np.mean([r['execution_time'] for r in successful_tests])
            summary['avg_exact_match'] = np.mean([r.get('exact_match_rate', 0) for r in successful_tests])
            summary['avg_topk_accuracy'] = np.mean([r.get('topk_accuracy', 0) for r in successful_tests])
            summary['avg_mae'] = np.mean([r.get('mae', 0) for r in successful_tests])
            summary['avg_rmse'] = np.mean([r.get('rmse', 0) for r in successful_tests])
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Summary for {impl_name}:")
            print(f"  Tests passed: {summary['successful_tests']}/{summary['total_tests']}")
            if successful_tests:
                print(f"  Average execution time: {summary['avg_execution_time']:.4f}s")
                print(f"  Average exact match: {summary['avg_exact_match']:.2f}%")
                print(f"  Average top-K accuracy: {summary['avg_topk_accuracy']:.2f}%")
                print(f"  Average MAE: {summary['avg_mae']:.6f}")
                print(f"  Average RMSE: {summary['avg_rmse']:.6f}")
            print(f"{'='*80}")
        
        return summary
    
    def batch_test(self, implementations_dir):
        """
        Test all implementations in a directory.
        
        Args:
            implementations_dir (str): Directory containing LLM implementations.
            
        Returns:
            list: List of test results for all implementations.
        """
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
        all_results = []
        for impl_file in impl_files:
            result = self.test_implementation(str(impl_file))
            all_results.append(result)
        
        # Print comparison summary
        self.print_comparison_summary(all_results)
        
        
        # Save results to file
        self.save_results_to_file(all_results)

        # Save structured summary for cross-suite aggregation
        self.save_summary_to_file(all_results)
        
        return all_results

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured summary JSON following schema.json."""
        script_dir = Path(__file__).parent
        if output_path is None:
            output_path = script_dir / "test_summary.json"
        else:
            output_path = Path(output_path)

        project_id = script_dir.parent.name
        unittest_dir = script_dir.name
        unittest_id = unittest_dir.replace("unittest", "")

        summary = {
            "suite": {
                "project_id": project_id,
                "unittest_id": unittest_id,
                "suite_path": f"{project_id}/{unittest_dir}",
                "num_tests_requested": self.num_tests,
            },
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": [],
        }

        for result in all_results:
            test_total = int(result.get("total_tests", self.num_tests))
            test_pass = int(result.get("successful_tests", 0))
            summary["implementations"].append(
                {
                    "name": result.get("implementation", "unknown"),
                    "test_total": test_total,
                    "test_pass": test_pass,
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"✓ Structured summary saved to: {output_path}")
        return str(output_path)
    
    def print_comparison_summary(self, all_results):
        """
        Print a comparison summary of all implementations.
        
        Args:
            all_results (list): List of test results for all implementations.
        """
        if not all_results:
            return
        
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        # Create comparison table
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg Time':<12} {'Exact Match':<12} {'Top-K Acc':<12} {'MAE':<12}")
        print("-" * 95)
        
        for result in all_results:
            name = result['implementation'][:23]
            # Check if there was an error loading the implementation
            if 'error' in result and 'successful_tests' not in result:
                print(f"{name:<30} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{result['successful_tests']}/{result['total_tests']}"
            
            if result['successful_tests'] > 0:
                avg_time = f"{result['avg_execution_time']:.4f}s"
                exact_match = f"{result['avg_exact_match']:.2f}%"
                topk_acc = f"{result['avg_topk_accuracy']:.2f}%"
                mae = f"{result['avg_mae']:.6f}"
            else:
                avg_time = "N/A"
                exact_match = "N/A"
                topk_acc = "N/A"
                mae = "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_time:<12} {exact_match:<12} {topk_acc:<12} {mae:<12}")
        
        print("-" * 95)

    




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
                    if 'error' in summary and 'successful_tests' not in summary:
                        f.write(f"ERROR: {summary['error']}\n")
                        f.write(f"Pass Rate: 0/0 (0.0%)\n")
                        f.write("\n")
                        continue
                    
                    # Write test summary
                    f.write(f"Total tests: {summary.get('total_tests', 0)}\n")
                    f.write(f"Successful tests: {summary.get('successful_tests', 0)}\n")
                    f.write(f"Failed tests: {summary.get('failed_tests', 0)}\n\n")
                    
                    # Write detailed test results if available
                    if 'test_results' in summary:
                        f.write("Detailed Test Results:\n")
                        f.write("-" * 80 + "\n")
                        for i, test_result in enumerate(summary['test_results'], 1):
                            test_desc = test_result.get('test_description', f'Test {i}')
                            f.write(f"\nTest {i}: {test_desc}\n")
                            
                            if test_result.get('success', False):
                                f.write("  ✓ Success\n")
                                f.write(f"    Execution time: {test_result.get('execution_time', 0):.4f}s\n")
                                if 'pixel_accuracy' in test_result:
                                    f.write(f"    Pixel accuracy: {test_result['pixel_accuracy']:.2f}%\n")
                                if 'non_zero_pixel_accuracy' in test_result:
                                    f.write(f"    Non-zero pixel accuracy: {test_result['non_zero_pixel_accuracy']:.2f}%\n")
                                if 'mae' in test_result:
                                    f.write(f"    MAE: {test_result['mae']:.6f}\n")
                                if 'rmse' in test_result:
                                    f.write(f"    RMSE: {test_result['rmse']:.6f}\n")
                            else:
                                f.write(f"  ✗ Failed: {test_result.get('error', 'Unknown error')}\n")
                                if test_result.get('execution_time', 0) > 0:
                                    f.write(f"    Execution time: {test_result['execution_time']:.4f}s\n")
                        
                        f.write("\n" + "-" * 80 + "\n\n")
                    
                    # Write statistics summary if available
                    if summary.get('successful_tests', 0) > 0:
                        f.write("Summary Statistics:\n")
                        if 'avg_execution_time' in summary:
                            f.write(f"  Average execution time: {summary['avg_execution_time']:.4f}s\n")
                        if 'avg_exact_match' in summary:
                            f.write(f"  Average exact match: {summary['avg_exact_match']:.2f}%\n")
                        if 'avg_topk_accuracy' in summary:
                            f.write(f"  Average top-K accuracy: {summary['avg_topk_accuracy']:.2f}%\n")
                        if 'avg_pixel_accuracy' in summary:
                            f.write(f"  Average pixel accuracy: {summary['avg_pixel_accuracy']:.2f}%\n")
                        if 'avg_non_zero_accuracy' in summary:
                            f.write(f"  Average non-zero accuracy: {summary['avg_non_zero_accuracy']:.2f}%\n")
                        if 'avg_mae' in summary:
                            f.write(f"  Average MAE: {summary['avg_mae']:.6f}\n")
                        if 'avg_rmse' in summary:
                            f.write(f"  Average RMSE: {summary['avg_rmse']:.6f}\n")
                    
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<30} {'Pass Rate':<12} {'Avg Time':<12} {'Pixel Acc':<12} {'MAE':<12}\n")
                f.write("-" * 80 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'successful_tests' not in summary:
                        f.write(f"{name:<30} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        continue
                    
                    pass_rate = f"{summary.get('successful_tests', 0)}/{summary.get('total_tests', 0)}"
                    
                    if summary.get('successful_tests', 0) > 0:
                        avg_time = f"{summary.get('avg_execution_time', 0):.4f}s" if 'avg_execution_time' in summary else "N/A"
                        pixel_acc = f"{summary.get('avg_pixel_accuracy', 0):.2f}%" if 'avg_pixel_accuracy' in summary else "N/A"
                        mae = f"{summary.get('avg_mae', 0):.6f}" if 'avg_mae' in summary else "N/A"
                    else:
                        avg_time = "N/A"
                        pixel_acc = "N/A"
                        mae = "N/A"
                    
                    f.write(f"{name:<30} {pass_rate:<12} {avg_time:<12} {pixel_acc:<12} {mae:<12}\n")
                
                f.write("-" * 80 + "\n")
                
                # Write ranking
                f.write(f"\n{'OVERALL RANKING':<20} {'Avg Pass Rate':<20} {'Pass Count':<15}\n")
                f.write("-" * 57 + "\n")
                sorted_summaries = sorted(all_summaries, key=lambda x: (x.get('successful_tests', 0) / max(x.get('total_tests', 1), 1)) * 100, reverse=True)
                for i, summary in enumerate(sorted_summaries, 1):
                    name = summary['implementation']
                    successful = summary.get('successful_tests', 0)
                    total = summary.get('total_tests', 0)
                    overall_rate = f"{(successful / max(total, 1)) * 100:.1f}%" if total > 0 else "0.0%"
                    count_str = f"{successful}/{total}"
                    f.write(f"{i}. {name:<30} {overall_rate:<20} {count_str:<15}\n")
                f.write("-" * 57 + "\n")
            
            print(f"\n✓ Results saved to: {output_file}")
            return str(output_file)
        
        except Exception as e:
            print(f"\n✗ Error saving results to file: {e}")
            return None



def main():
    """Main entry point for the test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for equirectangular_unwrap_topK_opacity implementations')
    parser.add_argument('--num-tests', type=int, default=5, 
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, 
                       default='llm_implementations',
                       help='Directory containing LLM implementations (default: llm_implementations)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    impl_dir = script_dir / args.impl_dir
    
    # Create test runner
    runner = TestRunner(num_tests=args.num_tests, verbose=not args.quiet)
    
    # Run tests
    results = runner.batch_test(str(impl_dir))
    
    return results


if __name__ == '__main__':
    main()

