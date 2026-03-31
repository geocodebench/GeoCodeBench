"""
Test Runner for HOGS Gaussian Model Functions
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from reference_implementation import GaussianModel as RefGaussianModel
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
    
    def compute_error(self, output, reference, output_type='tensor'):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Convert to tensors for comparison
        if output_type == 'tensor':
            if isinstance(output, torch.Tensor) and isinstance(reference, torch.Tensor):
                output_tensor = output
                reference_tensor = reference
            else:
                metrics['error'] = f"Output or reference is not a Tensor"
                return metrics
        elif output_type == 'numpy':
            if isinstance(output, np.ndarray) and isinstance(reference, np.ndarray):
                output_tensor = torch.from_numpy(output)
                reference_tensor = torch.from_numpy(reference)
            else:
                metrics['error'] = f"Output or reference is not a NumPy array"
                return metrics
        else:
            metrics['error'] = f"Unknown output type: {output_type}"
            return metrics
        
        # Check shape
        if output_tensor.shape != reference_tensor.shape:
            metrics['error'] = f"Shape mismatch: {output_tensor.shape} vs {reference_tensor.shape}"
            return metrics
        
        # L1 error (Mean Absolute Error)
        l1_error = torch.mean(torch.abs(output_tensor - reference_tensor)).item()
        metrics['l1_error'] = l1_error
        
        # L2 error (Root Mean Square Error)
        l2_error = torch.sqrt(torch.mean((output_tensor - reference_tensor) ** 2)).item()
        metrics['l2_error'] = l2_error
        
        # Max error
        max_error = torch.max(torch.abs(output_tensor - reference_tensor)).item()
        metrics['max_error'] = max_error
        
        # Relative error (avoid division by zero)
        ref_norm = torch.norm(reference_tensor)
        if ref_norm > 1e-10:
            relative_error = (torch.norm(output_tensor - reference_tensor) / ref_norm).item() * 100
        else:
            relative_error = 0.0 if max_error < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass (within tolerance)
        metrics['pass'] = max_error < self.tolerance
        
        return metrics
    
    def test_get_w(self, model_class, test_case):
        """Test get_w property."""
        try:
            model = model_class()
            model._xyz = test_case['_xyz']
            model._w = test_case['_w']
            
            ref_model = RefGaussianModel()
            ref_model._xyz = test_case['_xyz']
            ref_model._w = test_case['_w']
            
            start_time = time.time()
            output = model.get_w
            exec_time = time.time() - start_time
            
            reference = ref_model.get_w
            metrics = self.compute_error(output, reference, output_type='tensor')
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_get_w_inv(self, model_class, test_case):
        """Test get_w_inv property."""
        try:
            model = model_class()
            model._xyz = test_case['_xyz']
            model._w = test_case['_w']
            
            ref_model = RefGaussianModel()
            ref_model._xyz = test_case['_xyz']
            ref_model._w = test_case['_w']
            
            start_time = time.time()
            output = model.get_w_inv
            exec_time = time.time() - start_time
            
            reference = ref_model.get_w_inv
            metrics = self.compute_error(output, reference, output_type='tensor')
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_get_means3D(self, model_class, test_case):
        """Test get_means3D property."""
        try:
            model = model_class()
            model._xyz = test_case['_xyz']
            model._w = test_case['_w']
            
            ref_model = RefGaussianModel()
            ref_model._xyz = test_case['_xyz']
            ref_model._w = test_case['_w']
            
            start_time = time.time()
            output = model.get_means3D
            exec_time = time.time() - start_time
            
            reference = ref_model.get_means3D
            metrics = self.compute_error(output, reference, output_type='tensor')
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_get_points_hom(self, model_class, test_case):
        """Test get_points_hom property."""
        try:
            model = model_class()
            model._xyz = test_case['_xyz']
            model._w = test_case['_w']
            
            ref_model = RefGaussianModel()
            ref_model._xyz = test_case['_xyz']
            ref_model._w = test_case['_w']
            
            start_time = time.time()
            output = model.get_points_hom
            exec_time = time.time() - start_time
            
            reference = ref_model.get_points_hom
            metrics = self.compute_error(output, reference, output_type='tensor')
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_xyz_to_polar(self, model_class, test_case):
        """Test xyz_to_polar method."""
        try:
            model = model_class()
            ref_model = RefGaussianModel()
            
            xyz = test_case['_xyz']
            
            start_time = time.time()
            output = model.xyz_to_polar(xyz)
            exec_time = time.time() - start_time
            
            reference = ref_model.xyz_to_polar(xyz)
            
            # Check all three outputs
            metrics_polar = self.compute_error(output[0], reference[0], output_type='tensor')
            metrics_inv_r = self.compute_error(output[1], reference[1], output_type='tensor')
            metrics_r = self.compute_error(output[2], reference[2], output_type='tensor')
            
            # Aggregate metrics
            metrics = {
                'l1_error': max(metrics_polar.get('l1_error', float('inf')),
                               metrics_inv_r.get('l1_error', float('inf')),
                               metrics_r.get('l1_error', float('inf'))),
                'l2_error': max(metrics_polar.get('l2_error', float('inf')),
                               metrics_inv_r.get('l2_error', float('inf')),
                               metrics_r.get('l2_error', float('inf'))),
                'max_error': max(metrics_polar.get('max_error', float('inf')),
                                metrics_inv_r.get('max_error', float('inf')),
                                metrics_r.get('max_error', float('inf'))),
                'pass': (metrics_polar.get('pass', False) and 
                        metrics_inv_r.get('pass', False) and 
                        metrics_r.get('pass', False)),
                'execution_time': exec_time
            }
            
            if 'error' in metrics_polar:
                metrics['error'] = metrics_polar['error']
            elif 'error' in metrics_inv_r:
                metrics['error'] = metrics_inv_r['error']
            elif 'error' in metrics_r:
                metrics['error'] = metrics_r['error']
                
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_xyz_to_polar_np(self, model_class, test_case):
        """Test xyz_to_polar_np method."""
        try:
            model = model_class()
            ref_model = RefGaussianModel()
            
            xyz = test_case['_xyz'].numpy()
            
            start_time = time.time()
            output = model.xyz_to_polar_np(xyz)
            exec_time = time.time() - start_time
            
            reference = ref_model.xyz_to_polar_np(xyz)
            
            # Check all three outputs
            metrics_polar = self.compute_error(output[0], reference[0], output_type='numpy')
            metrics_inv_r = self.compute_error(output[1], reference[1], output_type='numpy')
            metrics_r = self.compute_error(output[2], reference[2], output_type='numpy')
            
            # Aggregate metrics
            metrics = {
                'l1_error': max(metrics_polar.get('l1_error', float('inf')),
                               metrics_inv_r.get('l1_error', float('inf')),
                               metrics_r.get('l1_error', float('inf'))),
                'l2_error': max(metrics_polar.get('l2_error', float('inf')),
                               metrics_inv_r.get('l2_error', float('inf')),
                               metrics_r.get('l2_error', float('inf'))),
                'max_error': max(metrics_polar.get('max_error', float('inf')),
                                metrics_inv_r.get('max_error', float('inf')),
                                metrics_r.get('max_error', float('inf'))),
                'pass': (metrics_polar.get('pass', False) and 
                        metrics_inv_r.get('pass', False) and 
                        metrics_r.get('pass', False)),
                'execution_time': exec_time
            }
            
            if 'error' in metrics_polar:
                metrics['error'] = metrics_polar['error']
            elif 'error' in metrics_inv_r:
                metrics['error'] = metrics_inv_r['error']
            elif 'error' in metrics_r:
                metrics['error'] = metrics_r['error']
                
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
        model_class = self.load_llm_implementation(impl_path)
        
        if model_class is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': 0
            }
        
        all_results = []
        
        # Test each function on all test cases
        function_tests = [
            ('get_w', self.test_get_w),
            ('get_w_inv', self.test_get_w_inv),
            ('get_means3D', self.test_get_means3D),
            ('get_points_hom', self.test_get_points_hom),
            ('xyz_to_polar', self.test_xyz_to_polar),
            ('xyz_to_polar_np', self.test_xyz_to_polar_np),
        ]
        
        for func_name, test_func in function_tests:
            if self.verbose:
                print(f"\n--- Testing function: {func_name} ---")
            
            for i, test_case in enumerate(self.test_cases):
                if self.verbose:
                    print(f"\nTest {i+1}/{len(self.test_cases)}: {test_case['description']}")
                
                result = test_func(model_class, test_case)
                
                test_result = {
                    'test_idx': i,
                    'function': func_name,
                    'description': test_case['description'],
                    'result': result
                }
                
                if self.verbose:
                    if result.get('pass', False):
                        print(f"  ✓ Pass (L1={result.get('l1_error', 0):.2e}, L2={result.get('l2_error', 0):.2e}, "
                              f"max={result.get('max_error', 0):.2e}, time={result.get('execution_time', 0):.6f}s)")
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
        
        # Per-function statistics
        function_stats = {}
        
        for test_result in all_results:
            result = test_result['result']
            func_name = test_result['function']
            
            if func_name not in function_stats:
                function_stats[func_name] = {
                    'passes': 0,
                    'total': 0,
                    'errors': []
                }
            
            function_stats[func_name]['total'] += 1
            
            if result.get('pass', False):
                passes.append(True)
                function_stats[func_name]['passes'] += 1
                l1_errors.append(result.get('l1_error', 0))
                l2_errors.append(result.get('l2_error', 0))
                max_errors.append(result.get('max_error', 0))
                exec_times.append(result.get('execution_time', 0))
            else:
                passes.append(False)
                function_stats[func_name]['errors'].append(result.get('error', 'Unknown error'))
        
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
        summary['function_stats'] = function_stats
        
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
            print(f"  Avg time: {summary['avg_time']:.6f}s")
        
        # Print per-function statistics
        if 'function_stats' in summary:
            print(f"\n  Per-function results:")
            for func_name, stats in summary['function_stats'].items():
                pass_rate = (stats['passes'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"    {func_name}: {pass_rate:.1f}% ({stats['passes']}/{stats['total']} passed)")
        
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
        print(f"Running {self.num_tests} test cases per implementation")
        print(f"Testing 6 functions: get_w, get_w_inv, get_means3D, get_points_hom, xyz_to_polar, xyz_to_polar_np\n")
        
        # Test each implementation
        all_summaries = []
        for impl_file in impl_files:
            summary = self.test_single_implementation(str(impl_file))
            all_summaries.append(summary)
        
        # Print comparison
        self.print_comparison(all_summaries)
        
        # Save results to file
        self.save_results_to_file(all_summaries)
        
        # Save structured JSON summary (schema.json-aligned)
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
        """Save structured test summary aligned with schema.json."""
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
            impl_name = summary.get("implementation", "unknown")

            # Match schema keys with existing runner counters.
            test_total = summary.get("total_test_count", summary.get("total_tests", 0))
            test_pass = summary.get("total_pass_count", 0)

            implementations.append(
                {
                    "name": impl_name,
                    "test_total": int(test_total or 0),
                    "test_pass": int(test_pass or 0),
                }
            )

        payload = {
            "suite": {
                "project_id": project_id,
                "unittest_id": unittest_id,
                "suite_path": suite_path,
                "num_tests_requested": self.num_tests,
            },
            "timestamp_utc": timestamp_utc,
            "implementations": implementations,
        }

        try:
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            if self.verbose:
                print(f"\n✓ test_summary.json saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving test_summary.json: {e}")
            return None



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for HOGS Gaussian Model functions')
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

