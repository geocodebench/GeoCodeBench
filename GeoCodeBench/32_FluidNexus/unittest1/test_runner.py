"""
Test Runner for GaussianModel.project_gas_constraints() function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import torch
import os
import sys
import importlib.util
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Add script directory to path for imports
_script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _script_dir)

# Import reference implementation and test generator
try:
    from reference_implementation import GaussianModel as RefGaussianModel
except ImportError as e:
    print(f"Error: could not import reference_implementation: {e}")
    sys.exit(1)
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
            
            if not hasattr(module, 'GaussianModel'):
                raise AttributeError(f"No GaussianModel class found in {filepath}")
            
            return module.GaussianModel
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Both outputs should be dictionaries
        if not isinstance(output, dict) or not isinstance(reference, dict):
            metrics['error'] = f"Type mismatch: {type(output)} vs {type(reference)}"
            return metrics
        
        # Check all keys match
        if set(output.keys()) != set(reference.keys()):
            missing_keys = set(reference.keys()) - set(output.keys())
            extra_keys = set(output.keys()) - set(reference.keys())
            metrics['error'] = f"Key mismatch: missing {missing_keys}, extra {extra_keys}"
            return metrics
        
        # Compute errors for each value
        total_l1 = 0
        total_l2 = 0
        total_max = 0
        valid_keys = 0
        
        for key in reference.keys():
            if key == 'elapsed_time':
                # Skip timing comparison
                continue
                
            out_val = output[key]
            ref_val = reference[key]
            
            # Both should be scalars (floats)
            if not isinstance(out_val, (int, float)) or not isinstance(ref_val, (int, float)):
                metrics['error'] = f"Value type mismatch for key '{key}': {type(out_val)} vs {type(ref_val)}"
                return metrics
            
            # Compute errors
            diff = abs(out_val - ref_val)
            total_l1 += diff
            total_l2 += diff ** 2
            total_max = max(total_max, diff)
            valid_keys += 1
        
        if valid_keys > 0:
            metrics['l1_error'] = total_l1 / valid_keys
            metrics['l2_error'] = (total_l2 / valid_keys) ** 0.5
            metrics['max_error'] = total_max
        else:
            metrics['l1_error'] = 0.0
            metrics['l2_error'] = 0.0
            metrics['max_error'] = 0.0
        
        # Relative error
        ref_norm = sum(abs(v) for k, v in reference.items() if k != 'elapsed_time')
        if ref_norm > 1e-10:
            out_diff_norm = sum(abs(output.get(k, 0) - v) for k, v in reference.items() if k != 'elapsed_time')
            relative_error = (out_diff_norm / ref_norm) * 100
        else:
            relative_error = 0.0 if metrics['max_error'] < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass
        metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_project_gas_constraints(self, impl_class, test_case, ref_model):
        """Test project_gas_constraints function."""
        try:
            # Create model instance
            model = impl_class()
            model.setup_constants(
                H=test_case['H'],
                p0=test_case['p0'],
                k=test_case['k'],
                KNN_K=test_case['KNN_K'],
                record_time=False
            )
            
            # Set up test data
            model._estimate_xyz = test_case['estimate_xyz'].clone()
            model._xyz = test_case['xyz'].clone()
            model._velocity = test_case['velocity'].clone()
            model._force = test_case['force'].clone()
            model._imass = test_case['imass'].clone()
            model._counts = test_case['counts'].clone()
            
            # Create reference model with same data
            ref_model.setup_constants(
                H=test_case['H'],
                p0=test_case['p0'],
                k=test_case['k'],
                KNN_K=test_case['KNN_K'],
                record_time=False
            )
            ref_model._estimate_xyz = test_case['estimate_xyz'].clone()
            ref_model._xyz = test_case['xyz'].clone()
            ref_model._velocity = test_case['velocity'].clone()
            ref_model._force = test_case['force'].clone()
            ref_model._imass = test_case['imass'].clone()
            ref_model._counts = test_case['counts'].clone()
            
            start_time = time.time()
            with torch.no_grad():
                output = model.project_gas_constraints()
            exec_time = time.time() - start_time
            
            with torch.no_grad():
                reference = ref_model.project_gas_constraints()
            
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
        impl_class = self.load_llm_implementation(impl_path)
        
        if impl_class is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': 0
            }
        
        # Create reference model
        ref_model = RefGaussianModel()
        
        all_results = []
        
        # Run all test cases
        for i, test_case in enumerate(self.test_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(self.test_cases)}: {test_case['description']}")
            
            result = self.test_project_gas_constraints(impl_class, test_case, ref_model)
            
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
                        f.write(f"Avg L1 error: {summary['avg_l1']:.2e}\n")
                        f.write(f"Avg L2 error: {summary['avg_l2']:.2e}\n")
                        f.write(f"Avg max error: {summary['avg_max']:.2e}\n")
                        f.write(f"Avg time: {summary['avg_time']:.4f}s\n")
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
        """Save structured test summary aligned with schema.json."""
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

        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        implementations = []
        for summary in all_summaries:
            name = summary.get("implementation", "unknown")
            test_total = int(summary.get("total_test_count", summary.get("total_tests", 0)) or 0)
            test_pass = int(summary.get("total_pass_count", 0) or 0)

            # If a runner failed to load/execute, keep counts at 0.
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

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving test_summary.json: {e}")
            return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for GaussianModel.project_gas_constraints()')
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
