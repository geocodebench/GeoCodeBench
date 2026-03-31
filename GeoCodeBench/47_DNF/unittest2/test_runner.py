"""
Test Runner for Dict_exp.forward() method
Supports batch testing of multiple LLM implementations.

Function: Dict_exp.forward(input) -> torch.Tensor
Description: Forward pass of Dict_exp network with positional encoding and Sigma scaling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import os
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementation and test generator
try:
    from reference_implementation import Dict_exp as RefDictExp
except ImportError:
    print("Error: reference_implementation.py not found!")
    sys.exit(1)
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-5, device='cpu'):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.device = device
        self.test_generator = TestDataGenerator(device=device)
        self.test_cases = self.test_generator.generate_test_suite(num_tests)
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'Dict'):
                raise AttributeError(f"No Dict class found in {filepath}")
            
            return module.Dict
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference using pure numerical computation."""
        metrics = {}
        
        if not isinstance(output, torch.Tensor):
            metrics['error'] = f"Output is not a tensor (got {type(output)})"
            return metrics
        
        if output.shape != reference.shape:
            metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
            return metrics
        
        # Compute errors (pure numerical, no extra libraries)
        diff = output - reference
        abs_diff = torch.abs(diff)
        
        # L1 error (Mean Absolute Error)
        metrics['l1_error'] = torch.mean(abs_diff).item()
        
        # L2 error (Root Mean Square Error)
        metrics['l2_error'] = torch.sqrt(torch.mean(diff ** 2)).item()
        
        # Max error
        metrics['max_error'] = torch.max(abs_diff).item()
        
        # MSE (Mean Squared Error)
        metrics['mse'] = torch.mean(diff ** 2).item()
        
        # Relative error
        ref_norm = torch.norm(reference).item()
        if ref_norm > 1e-10:
            out_diff_norm = torch.norm(diff).item()
            relative_error = (out_diff_norm / ref_norm) * 100
        else:
            relative_error = 0.0 if metrics['max_error'] < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass
        metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_forward(self, impl_class, test_case):
        """Test forward method."""
        model = test_case['model']
        input_tensor = test_case['input'].clone()
        
        try:
            # Create LLM implementation instance with same parameters as reference
            ref_model = model
            
            # Extract parameters from reference model
            bias = [layer.bias.data.clone() for layer in ref_model.U]
            U = [layer.weight.data.clone() for layer in ref_model.U]
            Sigma = [param.data.clone() for param in ref_model.Sigma]
            Vt = [layer.weight.data.clone() for layer in ref_model.Vt]
            
            llm_model = impl_class(
                bias=bias,
                U=U,
                Sigma=Sigma,
                Vt=Vt,
                latent_in=ref_model.latent_in,
                dropout_prob=ref_model.dropout_prob,
                dropout=ref_model.dropout,
                positional_enc=hasattr(ref_model, 'pos_embedder'),
                n_positional_freqs=ref_model.n_positional_freqs if hasattr(ref_model, 'n_positional_freqs') else 8
            )
            llm_model.eval()
            
            start_time = time.time()
            with torch.no_grad():
                output = llm_model(input_tensor.clone())
            exec_time = time.time() - start_time
            
            with torch.no_grad():
                reference = ref_model(input_tensor.clone())
            
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
            
            result = self.test_forward(impl_class, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (L1={result.get('l1_error', 0):.2e}, L2={result.get('l2_error', 0):.2e}, "
                          f"MSE={result.get('mse', 0):.2e}, max={result.get('max_error', 0):.2e}, "
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
        l1_errors = []
        l2_errors = []
        max_errors = []
        mse_errors = []
        relative_errors = []
        exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                l1_errors.append(result.get('l1_error', 0))
                l2_errors.append(result.get('l2_error', 0))
                max_errors.append(result.get('max_error', 0))
                mse_errors.append(result.get('mse', 0))
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
            
            if l1_errors:
                summary['avg_l1'] = sum(l1_errors) / len(l1_errors)
                summary['avg_l2'] = sum(l2_errors) / len(l2_errors)
                summary['avg_max'] = sum(max_errors) / len(max_errors)
                summary['avg_mse'] = sum(mse_errors) / len(mse_errors)
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
        
        if 'avg_l1' in summary:
            print(f"  Avg L1 error: {summary['avg_l1']:.2e}")
            print(f"  Avg L2 error: {summary['avg_l2']:.2e}")
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
        
        # Find all Python files (exclude template and __init__)
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
        
        # Save structured JSON summary (schema.json-compatible)
        self.save_summary_to_file(all_summaries)
        
        return all_summaries
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*120}")
        print("COMPARISON SUMMARY")
        print(f"{'='*120}\n")
        
        # Header
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg MSE':<12} {'Avg Max':<12} {'Avg Time':<12}")
        print("-" * 120)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_l1 = f"{summary.get('avg_l1', 0):.2e}" if 'avg_l1' in summary else "N/A"
            avg_l2 = f"{summary.get('avg_l2', 0):.2e}" if 'avg_l2' in summary else "N/A"
            avg_mse = f"{summary.get('avg_mse', 0):.2e}" if 'avg_mse' in summary else "N/A"
            avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_mse:<12} {avg_max:<12} {avg_time:<12}")
        
        print("-" * 120)
        
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
                f.write("="*120 + "\n")
                f.write("TEST RESULTS SUMMARY - Dict_exp.forward() method\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of implementations tested: {len(all_summaries)}\n")
                f.write(f"Number of test cases per implementation: {self.num_tests}\n")
                f.write(f"Error tolerance: {self.tolerance}\n")
                f.write(f"Device: {self.device}\n")
                f.write("="*120 + "\n\n")
                
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
                    
                    f.write(f"Total tests: {summary.get('total_tests', 0)}\n")
                    f.write(f"Pass rate: {summary.get('pass_rate', 0.0):.1f}%\n")
                    
                    if 'avg_l1' in summary:
                        f.write(f"Avg L1 error: {summary['avg_l1']:.2e}\n")
                        f.write(f"Avg L2 error: {summary['avg_l2']:.2e}\n")
                        f.write(f"Avg MSE: {summary['avg_mse']:.2e}\n")
                        f.write(f"Avg max error: {summary['avg_max']:.2e}\n")
                        f.write(f"Avg relative error: {summary['avg_relative']:.2f}%\n")
                        f.write(f"Avg time: {summary['avg_time']:.4f}s\n")
                    
                    # Write per-test details
                    f.write("\nPer-test details:\n")
                    for test_result in summary.get('results', []):
                        result = test_result['result']
                        desc = test_result['description']
                        if result.get('pass', False):
                            f.write(f"  ✓ {desc}: L1={result.get('l1_error', 0):.2e}, "
                                   f"L2={result.get('l2_error', 0):.2e}, MSE={result.get('mse', 0):.2e}\n")
                        else:
                            f.write(f"  ✗ {desc}: {result.get('error', 'Failed')}\n")
                    
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    overall_rate = summary.get('overall_pass_rate', 0.0)
                    f.write(f"\nOverall: {overall_rate:.1f}% ({pass_count}/{test_count} tests passed)\n")
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*120 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*120 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg MSE':<12} {'Avg Max':<12} {'Avg Time':<12}\n")
                f.write("-" * 120 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation'][:23]
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        continue
                    
                    pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
                    avg_l1 = f"{summary.get('avg_l1', 0):.2e}" if 'avg_l1' in summary else "N/A"
                    avg_l2 = f"{summary.get('avg_l2', 0):.2e}" if 'avg_l2' in summary else "N/A"
                    avg_mse = f"{summary.get('avg_mse', 0):.2e}" if 'avg_mse' in summary else "N/A"
                    avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
                    avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
                    
                    f.write(f"{name:<25} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_mse:<12} {avg_max:<12} {avg_time:<12}\n")
                
                f.write("-" * 120 + "\n")
                
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
                    f.write(f"{i}. {name:<30} {overall_rate:<15} {count_str:<15}\n")
                f.write("-" * 57 + "\n")
            
            print(f"\n✓ Results saved to: {output_file}")
            return str(output_file)
        
        except Exception as e:
            print(f"\n✗ Error saving results to file: {e}")
            return None
    
    def save_summary_to_file(self, all_summaries, output_path=None):
        """Save structured test summary JSON aligned with schema.json."""
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
                "num_tests_requested": self.num_tests,
            },
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": [],
        }
        
        for summary in all_summaries:
            if not isinstance(summary, dict):
                continue
            
            name = summary.get("implementation", "unknown")
            test_total_raw = summary.get("total_test_count", summary.get("total_tests", 0))
            test_pass_raw = summary.get("total_pass_count", 0)
            
            payload["implementations"].append(
                {
                    "name": name,
                    "test_total": int(test_total_raw or 0),
                    "test_pass": int(test_pass_raw or 0),
                }
            )
        
        if output_path is None:
            output_path = script_dir / "test_summary.json"
        else:
            output_path = Path(output_path)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if self.verbose:
                print(f"\n✓ Structured test summary saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving test_summary.json: {e}")
            return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for Dict_exp.forward() method')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                       help='Error tolerance for pass/fail (default: 1e-5)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run tests on (default: cpu)')
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    impl_dir = script_dir / args.impl_dir
    
    # Create test runner
    runner = TestRunner(
        num_tests=args.num_tests,
        verbose=not args.quiet,
        tolerance=args.tolerance,
        device=args.device
    )
    
    # Run tests
    results = runner.batch_test(str(impl_dir))
    
    return results


if __name__ == '__main__':
    main()
