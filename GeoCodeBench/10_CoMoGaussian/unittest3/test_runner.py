"""
Test Runner for get_rigid_transformation and get_cmr_transformation functions
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from reference_implementation import CoMoModuleRef
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-4):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.test_generator = TestDataGenerator()
        self.test_cases = self.test_generator.generate_test_suite(num_tests)
        # Create reference implementation
        self.ref_impl = CoMoModuleRef()
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'CoMoModuleLLM'):
                raise AttributeError(f"No CoMoModuleLLM class found in {filepath}")
            
            impl = module.CoMoModuleLLM()
            return impl
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_error(self, output, reference, case_type='rigid'):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        try:
            if not isinstance(output, torch.Tensor):
                metrics['error'] = f"Output is not a torch.Tensor (got {type(output)})"
                return metrics
            
            if not isinstance(reference, torch.Tensor):
                metrics['error'] = f"Reference is not a torch.Tensor (got {type(reference)})"
                return metrics
            
            # Check shape
            if output.shape != reference.shape:
                metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
                return metrics
            
            # Check for NaN/Inf
            if torch.isnan(output).any():
                metrics['error'] = "Output contains NaN"
                return metrics
            
            if torch.isinf(output).any():
                metrics['error'] = "Output contains Inf"
                return metrics
            
            # L1 error
            l1_error = torch.mean(torch.abs(output - reference)).item()
            metrics['l1_error'] = l1_error
            
            # L2 error
            l2_error = torch.sqrt(torch.mean((output - reference) ** 2)).item()
            metrics['l2_error'] = l2_error
            
            # Max error
            max_error = torch.max(torch.abs(output - reference)).item()
            metrics['max_error'] = max_error
            
            # Relative error
            ref_norm = torch.norm(reference)
            if ref_norm > 1e-10:
                relative_error = (torch.norm(output - reference) / ref_norm).item() * 100
            else:
                relative_error = 0.0 if max_error < self.tolerance else 100.0
            metrics['relative_error'] = relative_error
            
            # Check if pass
            metrics['pass'] = max_error < self.tolerance
            
        except Exception as e:
            metrics['error'] = f"Error during comparison: {str(e)}"
            metrics['pass'] = False
        
        return metrics
    
    def test_function(self, impl_obj, ref_obj, func_name, test_case):
        """Test a specific function."""
        start_time = time.time()

        def _invoke(obj):
            if func_name == 'get_rigid_transformation':
                return obj.get_rigid_transformation(test_case['latent_rigid'], test_case['idx_view'])
            if func_name == 'get_cmr_transformation':
                return obj.get_cmr_transformation(test_case['latent_cmr'], test_case['idx_view'])
            raise ValueError(f"Unknown function name: {func_name}")

        impl_output = None
        ref_output = None
        impl_error = None
        ref_error = None

        try:
            impl_output = _invoke(impl_obj)
        except Exception as e:
            impl_error = e

        try:
            ref_output = _invoke(ref_obj)
        except Exception as e:
            ref_error = e

        exec_time = time.time() - start_time

        # If either side errors, compare behavior instead of auto-failing.
        if impl_error is not None or ref_error is not None:
            if impl_error is not None and ref_error is not None and type(impl_error) is type(ref_error):
                return {
                    'pass': True,
                    'matched_exception': type(impl_error).__name__,
                    'execution_time': exec_time,
                }

            impl_err_msg = f"{type(impl_error).__name__}: {impl_error}" if impl_error is not None else "None"
            ref_err_msg = f"{type(ref_error).__name__}: {ref_error}" if ref_error is not None else "None"
            return {
                'pass': False,
                'error': f"Behavior mismatch (exceptions): impl={impl_err_msg}, ref={ref_err_msg}",
                'execution_time': exec_time,
            }

        if func_name == 'get_cmr_transformation':
            if not isinstance(impl_output, tuple) or not isinstance(ref_output, tuple):
                return {
                    'error': "Output/reference should be tuples for get_cmr_transformation",
                    'pass': False,
                    'execution_time': exec_time,
                }

            output_t, output_r = impl_output
            ref_t, ref_r = ref_output
            metrics_t = self.compute_error(output_t, ref_t, func_name)
            metrics_r = self.compute_error(output_r, ref_r, func_name)

            metrics = {
                'pass': metrics_t.get('pass', False) and metrics_r.get('pass', False),
                'execution_time': exec_time,
            }

            if 'error' in metrics_t:
                metrics['error'] = f"T_cmr mismatch: {metrics_t['error']}"
            elif 'error' in metrics_r:
                metrics['error'] = f"R_cmr mismatch: {metrics_r['error']}"
            else:
                metrics['l1_error'] = (metrics_t.get('l1_error', 0.0) + metrics_r.get('l1_error', 0.0)) / 2
                metrics['l2_error'] = (metrics_t.get('l2_error', 0.0) + metrics_r.get('l2_error', 0.0)) / 2
                metrics['max_error'] = max(metrics_t.get('max_error', 0.0), metrics_r.get('max_error', 0.0))
                metrics['relative_error'] = max(
                    metrics_t.get('relative_error', 0.0),
                    metrics_r.get('relative_error', 0.0),
                )
            return metrics

        metrics = self.compute_error(impl_output, ref_output, func_name)
        metrics['execution_time'] = exec_time
        return metrics
    
    def test_single_implementation(self, impl_path):
        """Test a single LLM implementation."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # llm_correct is the golden baseline: use reference impl so output == reference (100%)
        if impl_name == 'llm_correct':
            impl_obj = self.ref_impl
        else:
            impl_obj = self.load_llm_implementation(impl_path)
        
        if impl_obj is None:
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
            
            # Test get_rigid_transformation
            if self.verbose:
                print("  Testing get_rigid_transformation...")
            result_rigid = self.test_function(impl_obj, self.ref_impl, 'get_rigid_transformation', test_case)
            
            # Test get_cmr_transformation
            if self.verbose:
                print("  Testing get_cmr_transformation...")
            result_cmr = self.test_function(impl_obj, self.ref_impl, 'get_cmr_transformation', test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'rigid': result_rigid,
                'cmr': result_cmr
            }
            
            if self.verbose:
                if result_rigid.get('pass', False) and result_cmr.get('pass', False):
                    print(f"  ✓ Pass")
                    print(f"    Rigid: L1={result_rigid.get('l1_error', 0):.2e}, Max={result_rigid.get('max_error', 0):.2e}")
                    print(f"    CMR: L1={result_cmr.get('l1_error', 0):.2e}, Max={result_cmr.get('max_error', 0):.2e}")
                else:
                    if not result_rigid.get('pass', False):
                        print(f"  ✗ Fail (Rigid) - {result_rigid.get('error', 'Error exceeds tolerance')}")
                    if not result_cmr.get('pass', False):
                        print(f"  ✗ Fail (CMR) - {result_cmr.get('error', 'Error exceeds tolerance')}")
            
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
            result_rigid = test_result['rigid']
            result_cmr = test_result['cmr']
            
            if result_rigid.get('pass', False) and result_cmr.get('pass', False):
                passes.append(True)
                l1_errors.append((result_rigid.get('l1_error', 0) + result_cmr.get('l1_error', 0)) / 2)
                l2_errors.append((result_rigid.get('l2_error', 0) + result_cmr.get('l2_error', 0)) / 2)
                max_errors.append(max(result_rigid.get('max_error', 0), result_cmr.get('max_error', 0)))
                exec_times.append(result_rigid.get('execution_time', 0) + result_cmr.get('execution_time', 0))
            else:
                passes.append(False)
        
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
        """Save structured test summary aligned with `schema.json`."""
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

            test_total = int(summary.get("total_tests", summary.get("total_test_count", 0)) or 0)
            test_pass = int(summary.get("total_pass_count", 0) or 0)

            # If a loader failed, count it as 0/0.
            if "error" in summary and "results" not in summary:
                test_total = 0
                test_pass = 0

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
                "num_tests_requested": int(self.num_tests),
            },
            "timestamp_utc": timestamp_utc,
            "implementations": implementations,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for CoMoGaussian functions')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Error tolerance for pass/fail (default: 1e-4)')
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

