"""
Test Runner for skew_symmetric, transform_SE3, and rodrigues_formula functions
Supports batch testing of multiple LLM implementations.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from reference_implementation import (
    rodrigues_formula as ref_rodrigues_formula,
    skew_symmetric as ref_skew_symmetric,
    transform_SE3 as ref_transform_SE3,
)
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-5):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.test_generator = TestDataGenerator()
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            required_functions = ['skew_symmetric', 'transform_SE3', 'rodrigues_formula']
            for func_name in required_functions:
                if not hasattr(module, func_name):
                    raise AttributeError(f"No {func_name} function found in {filepath}")
            
            return {
                'skew_symmetric': module.skew_symmetric,
                'transform_SE3': module.transform_SE3,
                'rodrigues_formula': module.rodrigues_formula,
            }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check type
        if not isinstance(output, torch.Tensor):
            metrics['error'] = f"Output is not a Tensor (got {type(output)})"
            return metrics
        
        # Check shape
        if output.shape != reference.shape:
            metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
            return metrics
        
        # L1 error (Mean Absolute Error)
        l1_error = torch.mean(torch.abs(output - reference)).item()
        metrics['l1_error'] = l1_error
        
        # L2 error (Root Mean Square Error)
        l2_error = torch.sqrt(torch.mean((output - reference) ** 2)).item()
        metrics['l2_error'] = l2_error
        
        # Max error
        max_error = torch.max(torch.abs(output - reference)).item()
        metrics['max_error'] = max_error
        
        # Relative error (avoid division by zero)
        ref_norm = torch.norm(reference)
        if ref_norm > 1e-10:
            relative_error = (torch.norm(output - reference) / ref_norm).item() * 100
        else:
            relative_error = 0.0 if max_error < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass (within tolerance)
        metrics['pass'] = max_error < self.tolerance
        
        return metrics
    
    @staticmethod
    def _clone_test_case(test_case):
        """Clone test case so impl/ref cannot alter shared tensors. Only clone Tensor values."""
        out = {}
        for k, v in test_case.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.clone()
            else:
                out[k] = v
        return out

    def test_function(self, impl_func, ref_func, test_case, function_name):
        """Test a single function. Use cloned inputs so neither impl nor ref can alter the other."""
        try:
            start_time = time.time()
            ref_case = self._clone_test_case(test_case)
            reference = ref_func(**ref_case)
            impl_case = self._clone_test_case(test_case)
            output = impl_func(**impl_case)
            exec_time = time.time() - start_time

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
        """Test a single LLM implementation for all three functions."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # Load implementation
        impl_funcs = self.load_llm_implementation(impl_path)
        
        if impl_funcs is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': 0
            }
        
        # Reset RNG so every implementation is tested on the same set of test cases
        torch.manual_seed(self.test_generator.seed)
        
        # Generate test cases for each function
        skew_cases = self.test_generator.generate_skew_symmetric_tests(self.num_tests)
        se3_cases = self.test_generator.generate_transform_SE3_tests(self.num_tests)
        rodrigues_cases = self.test_generator.generate_rodrigues_formula_tests(self.num_tests)
        
        all_results = []
        
        # Test skew_symmetric
        if self.verbose:
            print(f"\n{'-'*80}")
            print("Testing skew_symmetric:")
            print(f"{'-'*80}")
        
        for i, test_case in enumerate(skew_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(skew_cases)}: {test_case['description']}")
            
            result = self.test_function(
                impl_funcs['skew_symmetric'],
                ref_skew_symmetric,
                {'w': test_case['w']},
                'skew_symmetric'
            )
            
            test_result = {
                'function': 'skew_symmetric',
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (L1={result.get('l1_error', 0):.2e}, L2={result.get('l2_error', 0):.2e}, "
                          f"max={result.get('max_error', 0):.2e})")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
            
            all_results.append(test_result)
        
        # Test transform_SE3
        if self.verbose:
            print(f"\n{'-'*80}")
            print("Testing transform_SE3:")
            print(f"{'-'*80}")
        
        for i, test_case in enumerate(se3_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(se3_cases)}: {test_case['description']}")
            
            result = self.test_function(
                impl_funcs['transform_SE3'],
                ref_transform_SE3,
                {'exp_w_skew': test_case['exp_w_skew'], 'p': test_case['p']},
                'transform_SE3'
            )
            
            test_result = {
                'function': 'transform_SE3',
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (L1={result.get('l1_error', 0):.2e}, L2={result.get('l2_error', 0):.2e}, "
                          f"max={result.get('max_error', 0):.2e})")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
            
            all_results.append(test_result)
        
        # Test rodrigues_formula
        if self.verbose:
            print(f"\n{'-'*80}")
            print("Testing rodrigues_formula:")
            print(f"{'-'*80}")
        
        for i, test_case in enumerate(rodrigues_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(rodrigues_cases)}: {test_case['description']}")
            
            result = self.test_function(
                impl_funcs['rodrigues_formula'],
                ref_rodrigues_formula,
                {'w': test_case['w'], 'theta': test_case['theta']},
                'rodrigues_formula'
            )
            
            test_result = {
                'function': 'rodrigues_formula',
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (L1={result.get('l1_error', 0):.2e}, L2={result.get('l2_error', 0):.2e}, "
                          f"max={result.get('max_error', 0):.2e})")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
            
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
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                l1_errors.append(result.get('l1_error', 0))
                l2_errors.append(result.get('l2_error', 0))
                max_errors.append(result.get('max_error', 0))
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
        
        pass_count = summary.get('total_pass_count', 0)
        test_count = summary.get('total_test_count', 0)
        print(f"  Overall: {summary.get('overall_pass_rate', 0.0):.1f}% ({pass_count}/{test_count} tests passed)")
        # Print failed tests for debugging (especially for llm_correct / reference)
        failed = [r for r in summary.get('results', []) if not r['result'].get('pass', True)]
        if failed and self.verbose:
            print(f"  Failed tests:")
            for r in failed:
                err = r['result'].get('error') or f"max_error={r['result'].get('max_error', 'N/A')}"
                print(f"    - {r['function']} #{r['test_idx']}: {r.get('description', '')} → {err}")
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
        print(f"Running {self.num_tests} test cases per function\n")
        
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
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg Max':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_l1 = f"{summary.get('avg_l1', 0):.2e}" if 'avg_l1' in summary else "N/A"
            avg_l2 = f"{summary.get('avg_l2', 0):.2e}" if 'avg_l2' in summary else "N/A"
            avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_max:<12}")
        
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
                    # Write failed test details for debugging
                    failed = [r for r in summary.get('results', []) if not r['result'].get('pass', True)]
                    if failed:
                        f.write("Failed tests:\n")
                        for r in failed:
                            err = r['result'].get('error') or f"max_error={r['result'].get('max_error', 'N/A')}"
                            f.write(f"  - {r['function']} #{r['test_idx']}: {r.get('description', '')} -> {err}\n")
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
    
    parser = argparse.ArgumentParser(description='Test runner for skew_symmetric, transform_SE3, rodrigues_formula')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run per function (default: 5)')
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

