"""
Test Runner for high_frequency_strength() and patchify_and_get_fdomain() functions
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import numpy as np
import os
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path for direct imports
sys.path.insert(0, os.path.dirname(__file__))

from reference_implementation import (
    high_frequency_strength as ref_high_frequency_strength,
    patchify_and_get_fdomain as ref_patchify_and_get_fdomain,
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
            
            if not hasattr(module, 'high_frequency_strength'):
                raise AttributeError(f"No high_frequency_strength function found in {filepath}")
            if not hasattr(module, 'patchify_and_get_fdomain'):
                raise AttributeError(f"No patchify_and_get_fdomain function found in {filepath}")
            
            return {
                'high_frequency_strength': module.high_frequency_strength,
                'patchify_and_get_fdomain': module.patchify_and_get_fdomain
            }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference, function_name):
        """Compute error metrics between output and reference using pure numpy."""
        metrics = {}
        
        try:
            if function_name == 'high_frequency_strength':
                # Output should be a scalar (float)
                if not isinstance(output, (float, np.floating, np.ndarray)):
                    metrics['error'] = f"Output type mismatch: expected scalar, got {type(output)}"
                    return metrics
                
                if isinstance(output, np.ndarray):
                    if output.size != 1:
                        metrics['error'] = f"Output size mismatch: expected scalar, got array of size {output.size}"
                        return metrics
                    output = float(output.item())
                
                if not isinstance(reference, (float, np.floating, np.ndarray)):
                    metrics['error'] = f"Reference type mismatch: expected scalar, got {type(reference)}"
                    return metrics
                
                if isinstance(reference, np.ndarray):
                    reference = float(reference.item())
                
                # Absolute error
                abs_error = abs(output - reference)
                metrics['abs_error'] = abs_error
                
                # Relative error
                if abs(reference) > 1e-10:
                    relative_error = (abs_error / abs(reference)) * 100
                else:
                    relative_error = 100.0 if abs_error > self.tolerance else 0.0
                metrics['relative_error'] = relative_error
                
                # Check if pass
                metrics['pass'] = abs_error < self.tolerance
                
            elif function_name == 'patchify_and_get_fdomain':
                # Output should be a tuple of (frequency_patches, high_frequency_score_list)
                if not isinstance(output, tuple) or len(output) != 2:
                    metrics['error'] = f"Output type mismatch: expected tuple of length 2, got {type(output)}"
                    return metrics
                
                if not isinstance(reference, tuple) or len(reference) != 2:
                    metrics['error'] = f"Reference type mismatch: expected tuple of length 2, got {type(reference)}"
                    return metrics
                
                freq_patches_out, scores_out = output
                freq_patches_ref, scores_ref = reference
                
                # Check frequency_patches
                if len(freq_patches_out) != len(freq_patches_ref):
                    metrics['error'] = f"Frequency patches length mismatch: {len(freq_patches_out)} vs {len(freq_patches_ref)}"
                    return metrics
                
                # Compute MSE for frequency patches
                patch_errors = []
                for i, (patch_out, patch_ref) in enumerate(zip(freq_patches_out, freq_patches_ref)):
                    if not isinstance(patch_out, np.ndarray) or not isinstance(patch_ref, np.ndarray):
                        metrics['error'] = f"Frequency patch {i} type mismatch: {type(patch_out)} vs {type(patch_ref)}"
                        return metrics
                    
                    if patch_out.shape != patch_ref.shape:
                        metrics['error'] = f"Frequency patch {i} shape mismatch: {patch_out.shape} vs {patch_ref.shape}"
                        return metrics
                    
                    # MSE for complex arrays (consider both real and imaginary parts)
                    mse = np.mean(np.abs(patch_out - patch_ref) ** 2)
                    patch_errors.append(mse)
                
                metrics['freq_patches_mse'] = np.mean(patch_errors)
                metrics['freq_patches_max_error'] = np.max(patch_errors)
                
                # Check high_frequency_score_list
                if len(scores_out) != len(scores_ref):
                    metrics['error'] = f"Score list length mismatch: {len(scores_out)} vs {len(scores_ref)}"
                    return metrics
                
                scores_out_arr = np.array(scores_out)
                scores_ref_arr = np.array(scores_ref)
                
                # MSE for scores
                scores_mse = np.mean((scores_out_arr - scores_ref_arr) ** 2)
                metrics['scores_mse'] = scores_mse
                
                # L1 error for scores
                scores_l1 = np.mean(np.abs(scores_out_arr - scores_ref_arr))
                metrics['scores_l1'] = scores_l1
                
                # Max error for scores
                scores_max = np.max(np.abs(scores_out_arr - scores_ref_arr))
                metrics['scores_max_error'] = scores_max
                
                # Relative error for scores
                if np.linalg.norm(scores_ref_arr) > 1e-10:
                    scores_relative = (np.linalg.norm(scores_out_arr - scores_ref_arr) / np.linalg.norm(scores_ref_arr)) * 100
                else:
                    scores_relative = 100.0 if scores_max > self.tolerance else 0.0
                metrics['scores_relative_error'] = scores_relative
                
                # Overall error (weighted combination)
                overall_error = max(metrics['freq_patches_max_error'], metrics['scores_max_error'])
                metrics['overall_max_error'] = overall_error
                
                # Check if pass
                metrics['pass'] = overall_error < self.tolerance
                
            else:
                metrics['error'] = f"Unknown function: {function_name}"
                return metrics
                
        except Exception as e:
            metrics['error'] = f"Error computing metrics: {str(e)}"
            return metrics
        
        return metrics
    
    def test_function(self, impl_funcs, test_case, ref_func):
        """Test a single function."""
        func_name = test_case['function']
        args = test_case['args']
        
        try:
            start_time = time.time()
            
            if func_name == 'high_frequency_strength':
                output = impl_funcs['high_frequency_strength'](args['patch'])
                reference = ref_func(args['patch'])
            elif func_name == 'patchify_and_get_fdomain':
                output = impl_funcs['patchify_and_get_fdomain'](args['image'], args['patch_size'])
                reference = ref_func(args['image'], args['patch_size'])
            else:
                return {
                    'error': f'Unknown function: {func_name}',
                    'pass': False,
                    'execution_time': 0
                }
            
            exec_time = time.time() - start_time
            
            metrics = self.compute_error(output, reference, func_name)
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
        impl_funcs = self.load_llm_implementation(impl_path)
        
        if impl_funcs is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': 0
            }
        
        # Get reference functions
        ref_funcs = {
            'high_frequency_strength': ref_high_frequency_strength,
            'patchify_and_get_fdomain': ref_patchify_and_get_fdomain
        }
        
        all_results = []
        
        # Run all test cases
        for i, test_case in enumerate(self.test_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(self.test_cases)}: {test_case['description']}")
            
            func_name = test_case['function']
            ref_func = ref_funcs[func_name]
            
            result = self.test_function(impl_funcs, test_case, ref_func)
            
            test_result = {
                'test_idx': i,
                'function': func_name,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    if func_name == 'high_frequency_strength':
                        print(f"  ✓ Pass (abs_error={result.get('abs_error', 0):.2e}, "
                              f"rel_error={result.get('relative_error', 0):.2f}%, "
                              f"time={result.get('execution_time', 0):.4f}s)")
                    else:
                        print(f"  ✓ Pass (scores_mse={result.get('scores_mse', 0):.2e}, "
                              f"scores_max={result.get('scores_max_error', 0):.2e}, "
                              f"time={result.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
                    if 'overall_max_error' in result:
                        print(f"    Max error: {result['overall_max_error']:.2e} (tolerance: {self.tolerance:.2e})")
                    elif 'abs_error' in result:
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
        
        # Separate by function
        hfs_results = [r for r in all_results if r['function'] == 'high_frequency_strength']
        pgf_results = [r for r in all_results if r['function'] == 'patchify_and_get_fdomain']
        
        # Compute metrics for high_frequency_strength
        if hfs_results:
            hfs_passes = [r['result'].get('pass', False) for r in hfs_results]
            hfs_abs_errors = [r['result'].get('abs_error', 0) for r in hfs_results if r['result'].get('pass', False)]
            hfs_rel_errors = [r['result'].get('relative_error', 0) for r in hfs_results if r['result'].get('pass', False)]
            hfs_times = [r['result'].get('execution_time', 0) for r in hfs_results if r['result'].get('pass', False)]
            
            summary['high_frequency_strength_pass_rate'] = (sum(hfs_passes) / len(hfs_passes) * 100) if hfs_passes else 0.0
            summary['high_frequency_strength_total_pass'] = sum(hfs_passes)
            summary['high_frequency_strength_total_count'] = len(hfs_passes)
            
            if hfs_abs_errors:
                summary['high_frequency_strength_avg_abs'] = np.mean(hfs_abs_errors)
                summary['high_frequency_strength_avg_rel'] = np.mean(hfs_rel_errors)
                summary['high_frequency_strength_avg_time'] = np.mean(hfs_times)
        
        # Compute metrics for patchify_and_get_fdomain
        if pgf_results:
            pgf_passes = [r['result'].get('pass', False) for r in pgf_results]
            pgf_scores_mse = [r['result'].get('scores_mse', 0) for r in pgf_results if r['result'].get('pass', False)]
            pgf_scores_max = [r['result'].get('scores_max_error', 0) for r in pgf_results if r['result'].get('pass', False)]
            pgf_times = [r['result'].get('execution_time', 0) for r in pgf_results if r['result'].get('pass', False)]
            
            summary['patchify_and_get_fdomain_pass_rate'] = (sum(pgf_passes) / len(pgf_passes) * 100) if pgf_passes else 0.0
            summary['patchify_and_get_fdomain_total_pass'] = sum(pgf_passes)
            summary['patchify_and_get_fdomain_total_count'] = len(pgf_passes)
            
            if pgf_scores_mse:
                summary['patchify_and_get_fdomain_avg_mse'] = np.mean(pgf_scores_mse)
                summary['patchify_and_get_fdomain_avg_max'] = np.mean(pgf_scores_max)
                summary['patchify_and_get_fdomain_avg_time'] = np.mean(pgf_times)
        
        # Overall metrics
        all_passes = [r['result'].get('pass', False) for r in all_results]
        if all_passes:
            summary['overall_pass_rate'] = (sum(all_passes) / len(all_passes) * 100)
            summary['total_pass_count'] = sum(all_passes)
            summary['total_test_count'] = len(all_passes)
        else:
            summary['overall_pass_rate'] = 0.0
            summary['total_pass_count'] = 0
            summary['total_test_count'] = 0
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Summary for {summary['implementation']}:")
        print(f"  Total tests: {summary['total_tests']}")
        
        if 'high_frequency_strength_pass_rate' in summary:
            print(f"\n  high_frequency_strength:")
            print(f"    Pass rate: {summary['high_frequency_strength_pass_rate']:.1f}% "
                  f"({summary['high_frequency_strength_total_pass']}/{summary['high_frequency_strength_total_count']})")
            if 'high_frequency_strength_avg_abs' in summary:
                print(f"    Avg abs error: {summary['high_frequency_strength_avg_abs']:.2e}")
                print(f"    Avg rel error: {summary['high_frequency_strength_avg_rel']:.2f}%")
                print(f"    Avg time: {summary['high_frequency_strength_avg_time']:.4f}s")
        
        if 'patchify_and_get_fdomain_pass_rate' in summary:
            print(f"\n  patchify_and_get_fdomain:")
            print(f"    Pass rate: {summary['patchify_and_get_fdomain_pass_rate']:.1f}% "
                  f"({summary['patchify_and_get_fdomain_total_pass']}/{summary['patchify_and_get_fdomain_total_count']})")
            if 'patchify_and_get_fdomain_avg_mse' in summary:
                print(f"    Avg MSE: {summary['patchify_and_get_fdomain_avg_mse']:.2e}")
                print(f"    Avg max error: {summary['patchify_and_get_fdomain_avg_max']:.2e}")
                print(f"    Avg time: {summary['patchify_and_get_fdomain_avg_time']:.4f}s")
        
        print(f"\n  Overall: {summary.get('overall_pass_rate', 0.0):.1f}% "
              f"({summary.get('total_pass_count', 0)}/{summary.get('total_test_count', 0)} tests passed)")
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
        
        # Save structured test summary for aggregation
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
        print(f"{'Implementation':<25} {'Function':<30} {'Pass Rate':<12} {'Avg Error':<15} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'ERROR':<30} {'0.0%':<12} {'N/A':<15} {'N/A':<12}")
                continue
            
            # high_frequency_strength
            if 'high_frequency_strength_pass_rate' in summary:
                pass_rate = f"{summary['high_frequency_strength_pass_rate']:.1f}%"
                avg_error = f"{summary.get('high_frequency_strength_avg_abs', 0):.2e}" if 'high_frequency_strength_avg_abs' in summary else "N/A"
                avg_time = f"{summary.get('high_frequency_strength_avg_time', 0):.4f}s" if 'high_frequency_strength_avg_time' in summary else "N/A"
                print(f"{name:<25} {'high_frequency_strength':<30} {pass_rate:<12} {avg_error:<15} {avg_time:<12}")
                name = ""
            
            # patchify_and_get_fdomain
            if 'patchify_and_get_fdomain_pass_rate' in summary:
                pass_rate = f"{summary['patchify_and_get_fdomain_pass_rate']:.1f}%"
                avg_error = f"{summary.get('patchify_and_get_fdomain_avg_max', 0):.2e}" if 'patchify_and_get_fdomain_avg_max' in summary else "N/A"
                avg_time = f"{summary.get('patchify_and_get_fdomain_avg_time', 0):.4f}s" if 'patchify_and_get_fdomain_avg_time' in summary else "N/A"
                print(f"{name:<25} {'patchify_and_get_fdomain':<30} {pass_rate:<12} {avg_error:<15} {avg_time:<12}")
                name = ""
            
            # Overall
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            pass_count = summary.get('total_pass_count', 0)
            test_count = summary.get('total_test_count', 0)
            count_str = f"({pass_count}/{test_count})"
            print(f"{'  → OVERALL':<25} {count_str:<30} {overall_rate:<12} {'':<15} {'':<12}")
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
                    if 'high_frequency_strength_pass_rate' in summary:
                        f.write("high_frequency_strength:\n")
                        f.write(f"  Pass rate: {summary['high_frequency_strength_pass_rate']:.1f}% "
                                f"({summary['high_frequency_strength_total_pass']}/{summary['high_frequency_strength_total_count']})\n")
                        if 'high_frequency_strength_avg_abs' in summary:
                            f.write(f"  Avg abs error: {summary['high_frequency_strength_avg_abs']:.2e}\n")
                            f.write(f"  Avg rel error: {summary['high_frequency_strength_avg_rel']:.2f}%\n")
                            f.write(f"  Avg time: {summary['high_frequency_strength_avg_time']:.4f}s\n")
                        f.write("\n")
                    
                    if 'patchify_and_get_fdomain_pass_rate' in summary:
                        f.write("patchify_and_get_fdomain:\n")
                        f.write(f"  Pass rate: {summary['patchify_and_get_fdomain_pass_rate']:.1f}% "
                                f"({summary['patchify_and_get_fdomain_total_pass']}/{summary['patchify_and_get_fdomain_total_count']})\n")
                        if 'patchify_and_get_fdomain_avg_mse' in summary:
                            f.write(f"  Avg MSE: {summary['patchify_and_get_fdomain_avg_mse']:.2e}\n")
                            f.write(f"  Avg max error: {summary['patchify_and_get_fdomain_avg_max']:.2e}\n")
                            f.write(f"  Avg time: {summary['patchify_and_get_fdomain_avg_time']:.4f}s\n")
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
                f.write(f"{'Implementation':<20} {'Function':<30} {'Pass Rate':<12} {'Avg Error':<15} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<30} {'ERROR':<30} {'0.0%':<12} {'N/A':<15} {'N/A':<12}\n")
                        f.write(f"{'  → OVERALL':<20} {'(0/0)':<30} {'0.0%':<12} {'':<15} {'':<12}\n")
                        f.write("-" * 100 + "\n")
                        continue
                    
                    # high_frequency_strength
                    if 'high_frequency_strength_pass_rate' in summary:
                        pass_rate = f"{summary['high_frequency_strength_pass_rate']:.1f}%"
                        avg_error = f"{summary.get('high_frequency_strength_avg_abs', 0):.2e}" if 'high_frequency_strength_avg_abs' in summary else "N/A"
                        avg_time = f"{summary.get('high_frequency_strength_avg_time', 0):.4f}s" if 'high_frequency_strength_avg_time' in summary else "N/A"
                        f.write(f"{name:<30} {'high_frequency_strength':<30} {pass_rate:<12} {avg_error:<15} {avg_time:<12}\n")
                        name = ""
                    
                    # patchify_and_get_fdomain
                    if 'patchify_and_get_fdomain_pass_rate' in summary:
                        pass_rate = f"{summary['patchify_and_get_fdomain_pass_rate']:.1f}%"
                        avg_error = f"{summary.get('patchify_and_get_fdomain_avg_max', 0):.2e}" if 'patchify_and_get_fdomain_avg_max' in summary else "N/A"
                        avg_time = f"{summary.get('patchify_and_get_fdomain_avg_time', 0):.4f}s" if 'patchify_and_get_fdomain_avg_time' in summary else "N/A"
                        f.write(f"{name:<30} {'patchify_and_get_fdomain':<30} {pass_rate:<12} {avg_error:<15} {avg_time:<12}\n")
                        name = ""
                    
                    overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    count_info = f"({pass_count}/{test_count})"
                    f.write(f"{'  → OVERALL':<20} {count_info:<30} {overall_rate:<12} {'':<15} {'':<12}\n")
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

        summary_payload = {
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
            # Existing runner already computes total_pass_count and total_test_count.
            summary_payload["implementations"].append(
                {
                    "name": summary.get("implementation", ""),
                    "test_total": int(summary.get("total_test_count", 0)),
                    "test_pass": int(summary.get("total_pass_count", 0)),
                }
            )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2, ensure_ascii=True)
            if not self.verbose:
                return str(output_path)
            print(f"\n✓ Structured summary saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving structured summary to file: {e}")
            return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for high_frequency_strength() and patchify_and_get_fdomain()')
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
