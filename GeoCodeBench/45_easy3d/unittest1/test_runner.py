"""
Test Runner for TwoWayAttentionBlock.forward() function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import os
import sys
import importlib.util
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Type
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementation and test generator
try:
    from reference_implementation import TwoWayAttentionBlock as RefTwoWayAttentionBlock, MLPBlock
except ImportError:
    print("Error: reference_implementation.py not found!")
    sys.exit(1)
try:
    from test_generator import TestDataGenerator
except ImportError:
    print("Error: test_generator.py not found!")
    sys.exit(1)


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
            
            if not hasattr(module, 'TwoWayAttentionBlock'):
                raise AttributeError(f"No TwoWayAttentionBlock class found in {filepath}")
            
            return module.TwoWayAttentionBlock
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check if output is a tuple (should be for this function)
        if isinstance(output, tuple) and isinstance(reference, tuple):
            if len(output) != len(reference):
                metrics['error'] = f"Tuple length mismatch: {len(output)} vs {len(reference)}"
                return metrics
            
            if len(output) != 2:
                metrics['error'] = f"Expected tuple of length 2, got {len(output)}"
                return metrics
            
            queries_out, keys_out = output
            queries_ref, keys_ref = reference
            
            # Check queries
            if not isinstance(queries_out, torch.Tensor):
                metrics['error'] = f"Output queries is not a tensor (got {type(queries_out)})"
                return metrics
            
            if queries_out.shape != queries_ref.shape:
                metrics['error'] = f"Queries shape mismatch: {queries_out.shape} vs {queries_ref.shape}"
                return metrics
            
            # Check keys
            if not isinstance(keys_out, torch.Tensor):
                metrics['error'] = f"Output keys is not a tensor (got {type(keys_out)})"
                return metrics
            
            if keys_out.shape != keys_ref.shape:
                metrics['error'] = f"Keys shape mismatch: {keys_out.shape} vs {keys_ref.shape}"
                return metrics
            
            # Compute errors for queries
            queries_l1 = torch.mean(torch.abs(queries_out - queries_ref)).item()
            queries_l2 = torch.sqrt(torch.mean((queries_out - queries_ref) ** 2)).item()
            queries_max = torch.max(torch.abs(queries_out - queries_ref)).item()
            
            # Compute errors for keys
            keys_l1 = torch.mean(torch.abs(keys_out - keys_ref)).item()
            keys_l2 = torch.sqrt(torch.mean((keys_out - keys_ref) ** 2)).item()
            keys_max = torch.max(torch.abs(keys_out - keys_ref)).item()
            
            # Average across both outputs
            metrics['l1_error'] = (queries_l1 + keys_l1) / 2.0
            metrics['l2_error'] = (queries_l2 + keys_l2) / 2.0
            metrics['max_error'] = max(queries_max, keys_max)
            metrics['queries_l1'] = queries_l1
            metrics['queries_l2'] = queries_l2
            metrics['queries_max'] = queries_max
            metrics['keys_l1'] = keys_l1
            metrics['keys_l2'] = keys_l2
            metrics['keys_max'] = keys_max
            
        else:
            metrics['error'] = f"Type mismatch: expected tuple, got {type(output)} vs {type(reference)}"
            return metrics
        
        # Relative error
        queries_ref_norm = torch.norm(queries_ref).item()
        keys_ref_norm = torch.norm(keys_ref).item()
        ref_norm = queries_ref_norm + keys_ref_norm
        
        if ref_norm > 1e-10:
            queries_diff_norm = torch.norm(queries_out - queries_ref).item()
            keys_diff_norm = torch.norm(keys_out - keys_ref).item()
            diff_norm = queries_diff_norm + keys_diff_norm
            relative_error = (diff_norm / ref_norm) * 100
        else:
            relative_error = 0.0 if metrics['max_error'] < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass
        metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_forward(self, impl_class, test_case, ref_model):
        """Test forward function."""
        queries = test_case['queries']
        keys = test_case['keys']
        query_pe = test_case['query_pe']
        key_pe = test_case['key_pe']
        
        try:
            # Create model instance with same config as reference
            embedding_dim = test_case['embedding_dim']
            num_heads = test_case['num_heads']
            mlp_dim = test_case['mlp_dim']
            skip_first_layer_pe = test_case['skip_first_layer_pe']
            
            model = impl_class(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                skip_first_layer_pe=skip_first_layer_pe
            )
            model.eval()
            
            # Copy weights from reference model
            model.load_state_dict(ref_model.state_dict())
            
            start_time = time.time()
            with torch.no_grad():
                output = model.forward(queries=queries, keys=keys, query_pe=query_pe, key_pe=key_pe)
            exec_time = time.time() - start_time
            
            with torch.no_grad():
                reference = ref_model.forward(queries=queries, keys=keys, query_pe=query_pe, key_pe=key_pe)
            
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
            
            # Create reference model for this test case
            ref_model = RefTwoWayAttentionBlock(
                embedding_dim=test_case['embedding_dim'],
                num_heads=test_case['num_heads'],
                mlp_dim=test_case['mlp_dim'],
                skip_first_layer_pe=test_case['skip_first_layer_pe']
            )
            ref_model.eval()
            
            result = self.test_forward(impl_class, test_case, ref_model)
            
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
        # Save structured JSON summary (schema-aligned)
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

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured test summary to JSON (schema-aligned)."""
        try:
            script_dir = Path(__file__).parent
            project_id = script_dir.parent.name
            unittest_id = script_dir.name.replace("unittest", "")
            suite_path = f"{project_id}/{script_dir.name}"

            if output_path is None:
                output_path = script_dir / "test_summary.json"
            else:
                output_path = Path(output_path)

            implementations = []
            for summary in all_results or []:
                impl_name = summary.get("implementation")
                # Runners in this repo sometimes use different counter names
                test_total = summary.get("total_tests", summary.get("total_test_count", 0))
                test_pass = summary.get("total_pass_count", 0)

                implementations.append(
                    {
                        "name": impl_name,
                        "test_total": int(test_total),
                        "test_pass": int(test_pass),
                    }
                )

            payload = {
                "suite": {
                    "project_id": project_id,
                    "unittest_id": unittest_id,
                    "suite_path": suite_path,
                    "num_tests_requested": int(self.num_tests),
                },
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "implementations": implementations,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving test_summary.json: {e}")
            return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for TwoWayAttentionBlock.forward()')
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
