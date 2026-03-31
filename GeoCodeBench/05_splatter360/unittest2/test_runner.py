"""
Test Runner for correlation_softmax_depth function
Supports batch testing of multiple LLM implementations.
"""

import torch
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

from reference_implementation import correlation_softmax_depth as ref_correlation_softmax_depth
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
            
            if not hasattr(module, 'correlation_softmax_depth'):
                raise AttributeError(f"No correlation_softmax_depth function found in {filepath}")
            
            return module.correlation_softmax_depth
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check if output is tuple
        if isinstance(output, tuple):
            if len(output) != 2:
                metrics['error'] = f"Output tuple length mismatch: {len(output)} vs 2"
                return metrics
            depth_out, prob_out = output
            depth_ref, prob_ref = reference
        else:
            metrics['error'] = f"Output is not a tuple (got {type(output)})"
            return metrics
        
        # Check depth
        if not isinstance(depth_out, torch.Tensor):
            metrics['error'] = f"Depth is not a torch.Tensor (got {type(depth_out)})"
            return metrics
        
        if depth_out.shape != depth_ref.shape:
            metrics['error'] = f"Depth shape mismatch: {depth_out.shape} vs {depth_ref.shape}"
            return metrics
        
        # Check prob
        if not isinstance(prob_out, torch.Tensor):
            metrics['error'] = f"Prob is not a torch.Tensor (got {type(prob_out)})"
            return metrics
        
        if prob_out.shape != prob_ref.shape:
            metrics['error'] = f"Prob shape mismatch: {prob_out.shape} vs {prob_ref.shape}"
            return metrics
        
        # Convert to float for comparison
        depth_out_f = depth_out.float()
        depth_ref_f = depth_ref.float()
        prob_out_f = prob_out.float()
        prob_ref_f = prob_ref.float()
        
        # Compute errors for depth
        depth_l1 = torch.mean(torch.abs(depth_out_f - depth_ref_f)).item()
        depth_l2 = torch.sqrt(torch.mean((depth_out_f - depth_ref_f) ** 2)).item()
        depth_max = torch.max(torch.abs(depth_out_f - depth_ref_f)).item()
        
        # Compute errors for prob
        prob_l1 = torch.mean(torch.abs(prob_out_f - prob_ref_f)).item()
        prob_l2 = torch.sqrt(torch.mean((prob_out_f - prob_ref_f) ** 2)).item()
        prob_max = torch.max(torch.abs(prob_out_f - prob_ref_f)).item()
        
        # Overall metrics (max of depth and prob)
        metrics['l1_error'] = max(depth_l1, prob_l1)
        metrics['l2_error'] = max(depth_l2, prob_l2)
        metrics['max_error'] = max(depth_max, prob_max)
        
        # Detailed metrics
        metrics['depth_l1'] = depth_l1
        metrics['depth_l2'] = depth_l2
        metrics['depth_max'] = depth_max
        metrics['prob_l1'] = prob_l1
        metrics['prob_l2'] = prob_l2
        metrics['prob_max'] = prob_max
        
        # Relative error
        depth_ref_norm = torch.norm(depth_ref_f)
        prob_ref_norm = torch.norm(prob_ref_f)
        
        if depth_ref_norm > 1e-10:
            depth_rel_error = (torch.norm(depth_out_f - depth_ref_f) / depth_ref_norm).item() * 100
        else:
            depth_rel_error = 0.0 if depth_max < self.tolerance else 100.0
        
        if prob_ref_norm > 1e-10:
            prob_rel_error = (torch.norm(prob_out_f - prob_ref_f) / prob_ref_norm).item() * 100
        else:
            prob_rel_error = 0.0 if prob_max < self.tolerance else 100.0
        
        metrics['relative_error'] = max(depth_rel_error, prob_rel_error)
        
        # Check if pass (within tolerance)
        metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_correlation_softmax_depth(self, impl_func, test_case):
        """Test correlation_softmax_depth function."""
        feature0 = test_case['feature0']
        feature1 = test_case['feature1']
        intrinsics = test_case['intrinsics']
        pose = test_case['pose']
        depth_candidates = test_case['depth_candidates']
        depth_from_argmax = test_case['depth_from_argmax']
        pred_bidir_depth = test_case['pred_bidir_depth']
        
        try:
            start_time = time.time()
            output = impl_func(feature0, feature1, intrinsics, pose, depth_candidates,
                             depth_from_argmax=depth_from_argmax, pred_bidir_depth=pred_bidir_depth)
            exec_time = time.time() - start_time
            
            reference = ref_correlation_softmax_depth(feature0, feature1, intrinsics, pose, depth_candidates,
                                                     depth_from_argmax=depth_from_argmax, pred_bidir_depth=pred_bidir_depth)
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
            
            result = self.test_correlation_softmax_depth(impl_func, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (L1={result.get('l1_error', 0):.2e}, L2={result.get('l2_error', 0):.2e}, "
                          f"max={result.get('max_error', 0):.2e}, time={result.get('execution_time', 0):.4f}s)")
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
                            result = test_result.get('result', {})
                            f.write(f"\nTest {i}: {test_desc}\n")
                            
                            if result.get('pass', False):
                                f.write("  ✓ Success\n")
                                f.write(f"    Execution time: {result.get('execution_time', 0):.4f}s\n")
                                f.write(f"    L1 error: {result.get('l1_error', 0):.2e}\n")
                                f.write(f"    L2 error: {result.get('l2_error', 0):.2e}\n")
                                f.write(f"    Max error: {result.get('max_error', 0):.2e}\n")
                            else:
                                f.write(f"  ✗ Failed: {result.get('error', 'Error exceeds tolerance')}\n")
                                if result.get('execution_time', 0) > 0:
                                    f.write(f"    Execution time: {result['execution_time']:.4f}s\n")
                        
                        f.write("\n" + "-" * 80 + "\n\n")
                    
                    # Write statistics summary if available
                    if summary.get('total_pass_count', 0) > 0:
                        f.write("Summary Statistics:\n")
                        if 'avg_time' in summary:
                            f.write(f"  Average execution time: {summary['avg_time']:.4f}s\n")
                        if 'avg_l1' in summary:
                            f.write(f"  Average L1 error: {summary['avg_l1']:.2e}\n")
                        if 'avg_l2' in summary:
                            f.write(f"  Average L2 error: {summary['avg_l2']:.2e}\n")
                        if 'avg_max' in summary:
                            f.write(f"  Average max error: {summary['avg_max']:.2e}\n")
                    
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
                    avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"

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
    
    parser = argparse.ArgumentParser(description='Test runner for correlation_softmax_depth')
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
