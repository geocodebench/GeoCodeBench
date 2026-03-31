"""
Test Runner for chol.py functions
Supports batch testing of multiple LLM implementations.
Tests: CholeskySolver.apply, block_solve, schur_solve
"""

from __future__ import annotations

import torch
import os
import sys
import importlib.util
import time
from pathlib import Path
from datetime import datetime, timezone
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementation and test generator
try:
    from reference_implementation import CholeskySolver, block_solve, schur_solve
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
        self.device = torch.device('cpu')  # No CUDA
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required functions/classes
            required = ['CholeskySolver', 'block_solve', 'schur_solve']
            missing = [r for r in required if not hasattr(module, r)]
            if missing:
                raise AttributeError(f"Missing: {missing} in {filepath}")
            
            return module
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check if output is a tuple (for schur_solve)
        if isinstance(output, tuple) and isinstance(reference, tuple):
            if len(output) != len(reference):
                metrics['error'] = f"Tuple length mismatch: {len(output)} vs {len(reference)}"
                return metrics
            
            # Compute errors for each element
            total_l1 = 0
            total_l2 = 0
            total_max = 0
            
            for i, (out_tensor, ref_tensor) in enumerate(zip(output, reference)):
                if not isinstance(out_tensor, torch.Tensor):
                    metrics['error'] = f"Output element {i} is not a tensor (got {type(out_tensor)})"
                    return metrics
                
                if out_tensor.shape != ref_tensor.shape:
                    metrics['error'] = f"Shape mismatch in element {i}: {out_tensor.shape} vs {ref_tensor.shape}"
                    return metrics
                
                # L1 error
                l1 = torch.mean(torch.abs(out_tensor - ref_tensor)).item()
                total_l1 += l1
                
                # L2 error (MSE)
                l2 = torch.sqrt(torch.mean((out_tensor - ref_tensor) ** 2)).item()
                total_l2 += l2
                
                # Max error
                max_err = torch.max(torch.abs(out_tensor - ref_tensor)).item()
                total_max = max(total_max, max_err)
            
            # Average across elements
            metrics['l1_error'] = total_l1 / len(output)
            metrics['l2_error'] = total_l2 / len(output)
            metrics['max_error'] = total_max
            
        elif isinstance(output, torch.Tensor) and isinstance(reference, torch.Tensor):
            # Single tensor output
            if output.shape != reference.shape:
                metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
                return metrics
            
            # L1 error
            metrics['l1_error'] = torch.mean(torch.abs(output - reference)).item()
            
            # L2 error (MSE)
            metrics['l2_error'] = torch.sqrt(torch.mean((output - reference) ** 2)).item()
            
            # Max error
            metrics['max_error'] = torch.max(torch.abs(output - reference)).item()
        else:
            metrics['error'] = f"Type mismatch: {type(output)} vs {type(reference)}"
            return metrics
        
        # Relative error
        if isinstance(reference, tuple):
            ref_norm = sum(torch.norm(t).item() for t in reference)
        else:
            ref_norm = torch.norm(reference).item()
        
        if ref_norm > 1e-10:
            if isinstance(output, tuple):
                out_diff_norm = sum(torch.norm(o - r).item() for o, r in zip(output, reference))
            else:
                out_diff_norm = torch.norm(output - reference).item()
            relative_error = (out_diff_norm / ref_norm) * 100
        else:
            relative_error = 0.0 if metrics['max_error'] < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass
        metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_cholesky_apply(self, module, test_case):
        """Test CholeskySolver.apply function."""
        H = test_case['H']
        b = test_case['b']
        
        try:
            start_time = time.time()
            with torch.no_grad():
                output = module.CholeskySolver.apply(H, b)
            exec_time = time.time() - start_time
            
            with torch.no_grad():
                reference = CholeskySolver.apply(H, b)
            
            metrics = self.compute_error(output, reference)
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_block_solve(self, module, test_case):
        """Test block_solve function."""
        H = test_case['H']
        b = test_case['b']
        ep = test_case['ep']
        lm = test_case['lm']
        
        try:
            start_time = time.time()
            with torch.no_grad():
                output = module.block_solve(H, b, ep, lm)
            exec_time = time.time() - start_time
            
            with torch.no_grad():
                reference = block_solve(H, b, ep, lm)
            
            metrics = self.compute_error(output, reference)
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return metrics
    
    def test_schur_solve(self, module, test_case):
        """Test schur_solve function."""
        H = test_case['H']
        E = test_case['E']
        C = test_case['C']
        v = test_case['v']
        w = test_case['w']
        ep = test_case['ep']
        lm = test_case['lm']
        sless = test_case['sless']
        
        try:
            start_time = time.time()
            with torch.no_grad():
                output = module.schur_solve(H, E, C, v, w, ep, lm, sless)
            exec_time = time.time() - start_time
            
            with torch.no_grad():
                reference = schur_solve(H, E, C, v, w, ep, lm, sless)
            
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
        module = self.load_llm_implementation(impl_path)
        
        if module is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': 0
            }
        
        all_results = {}
        
        # Test CholeskySolver.apply
        cholesky_cases = self.test_generator.generate_cholesky_test_cases(self.num_tests)
        cholesky_results = []
        for i, test_case in enumerate(cholesky_cases):
            if self.verbose:
                print(f"\nCholeskySolver.apply Test {i+1}/{len(cholesky_cases)}: {test_case['description']}")
            
            result = self.test_cholesky_apply(module, test_case)
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
            
            cholesky_results.append(test_result)
        all_results['cholesky_apply'] = cholesky_results
        
        # Test block_solve
        block_cases = self.test_generator.generate_block_solve_test_cases(self.num_tests)
        block_results = []
        for i, test_case in enumerate(block_cases):
            if self.verbose:
                print(f"\nblock_solve Test {i+1}/{len(block_cases)}: {test_case['description']}")
            
            result = self.test_block_solve(module, test_case)
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
            
            block_results.append(test_result)
        all_results['block_solve'] = block_results
        
        # Test schur_solve
        schur_cases = self.test_generator.generate_schur_solve_test_cases(self.num_tests)
        schur_results = []
        for i, test_case in enumerate(schur_cases):
            if self.verbose:
                print(f"\nschur_solve Test {i+1}/{len(schur_cases)}: {test_case['description']}")
            
            result = self.test_schur_solve(module, test_case)
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
            
            schur_results.append(test_result)
        all_results['schur_solve'] = schur_results
        
        # Compute summary
        summary = self.compute_summary(impl_name, all_results)
        
        if self.verbose:
            self.print_summary(summary)
        
        return summary
    
    def compute_summary(self, impl_name, all_results):
        """Compute summary statistics."""
        summary = {
            'implementation': impl_name,
            'results': all_results
        }
        
        all_passes = []
        all_l1_errors = []
        all_l2_errors = []
        all_max_errors = []
        all_exec_times = []
        
        # Process each function
        for func_name, func_results in all_results.items():
            passes = []
            l1_errors = []
            l2_errors = []
            max_errors = []
            exec_times = []
            
            for test_result in func_results:
                result = test_result['result']
                if result.get('pass', False):
                    passes.append(True)
                    l1_errors.append(result.get('l1_error', 0))
                    l2_errors.append(result.get('l2_error', 0))
                    max_errors.append(result.get('max_error', 0))
                    exec_times.append(result.get('execution_time', 0))
                else:
                    passes.append(False)
            
            # Function-level statistics
            if passes:
                pass_rate = sum(passes) / len(passes) * 100
                summary[f'{func_name}_pass_rate'] = pass_rate
                summary[f'{func_name}_total_tests'] = len(passes)
                summary[f'{func_name}_pass_count'] = sum(passes)
                
                if l1_errors:
                    summary[f'{func_name}_avg_l1'] = sum(l1_errors) / len(l1_errors)
                    summary[f'{func_name}_avg_l2'] = sum(l2_errors) / len(l2_errors)
                    summary[f'{func_name}_avg_max'] = sum(max_errors) / len(max_errors)
                    summary[f'{func_name}_avg_time'] = sum(exec_times) / len(exec_times)
                
                all_passes.extend(passes)
                all_l1_errors.extend(l1_errors)
                all_l2_errors.extend(l2_errors)
                all_max_errors.extend(max_errors)
                all_exec_times.extend(exec_times)
        
        # Overall statistics
        if all_passes:
            summary['overall_pass_rate'] = sum(all_passes) / len(all_passes) * 100
            summary['total_pass_count'] = sum(all_passes)
            summary['total_test_count'] = len(all_passes)
            
            if all_l1_errors:
                summary['avg_l1'] = sum(all_l1_errors) / len(all_l1_errors)
                summary['avg_l2'] = sum(all_l2_errors) / len(all_l2_errors)
                summary['avg_max'] = sum(all_max_errors) / len(all_max_errors)
                summary['avg_time'] = sum(all_exec_times) / len(all_exec_times)
        else:
            summary['overall_pass_rate'] = 0.0
            summary['total_pass_count'] = 0
            summary['total_test_count'] = 0
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Summary for {summary['implementation']}:")
        
        func_names = ['cholesky_apply', 'block_solve', 'schur_solve']
        for func_name in func_names:
            if f'{func_name}_pass_rate' in summary:
                pass_rate = summary[f'{func_name}_pass_rate']
                pass_count = summary.get(f'{func_name}_pass_count', 0)
                total_tests = summary.get(f'{func_name}_total_tests', 0)
                print(f"  {func_name}:")
                print(f"    Pass rate: {pass_rate:.1f}% ({pass_count}/{total_tests})")
                if f'{func_name}_avg_l1' in summary:
                    print(f"    Avg L1 error: {summary[f'{func_name}_avg_l1']:.2e}")
                    print(f"    Avg L2 error: {summary[f'{func_name}_avg_l2']:.2e}")
                    print(f"    Avg max error: {summary[f'{func_name}_avg_max']:.2e}")
                    print(f"    Avg time: {summary[f'{func_name}_avg_time']:.4f}s")
        
        pass_count = summary.get('total_pass_count', 0)
        test_count = summary.get('total_test_count', 0)
        overall_rate = summary.get('overall_pass_rate', 0.0)
        print(f"  Overall: {overall_rate:.1f}% ({pass_count}/{test_count} tests passed)")
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
        print(f"Running {self.num_tests} test cases per function per implementation\n")
        
        # Test each implementation
        all_summaries = []
        for impl_file in impl_files:
            summary = self.test_single_implementation(str(impl_file))
            all_summaries.append(summary)
        
        # Print comparison
        self.print_comparison(all_summaries)
        
        # Save results to file
        self.save_results_to_file(all_summaries)
        
        # Save structured summary aligned with schema.json
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
        print(f"{'Implementation':<25} {'Function':<20} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg Max':<12} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'ERROR':<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            func_names = ['cholesky_apply', 'block_solve', 'schur_solve']
            for func_name in func_names:
                if f'{func_name}_pass_rate' in summary:
                    pass_rate = f"{summary[f'{func_name}_pass_rate']:.1f}%"
                    avg_l1 = f"{summary.get(f'{func_name}_avg_l1', 0):.2e}" if f'{func_name}_avg_l1' in summary else "N/A"
                    avg_l2 = f"{summary.get(f'{func_name}_avg_l2', 0):.2e}" if f'{func_name}_avg_l2' in summary else "N/A"
                    avg_max = f"{summary.get(f'{func_name}_avg_max', 0):.2e}" if f'{func_name}_avg_max' in summary else "N/A"
                    avg_time = f"{summary.get(f'{func_name}_avg_time', 0):.4f}s" if f'{func_name}_avg_time' in summary else "N/A"
                    
                    print(f"{name:<25} {func_name:<20} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_max:<12} {avg_time:<12}")
                    name = ""  # Only print name once
            
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            pass_count = summary.get('total_pass_count', 0)
            test_count = summary.get('total_test_count', 0)
            count_info = f"({pass_count}/{test_count})"
            print(f"{'  → AVERAGE':<25} {count_info:<20} {overall_rate:<12} {'':<12} {'':<12} {'':<12} {'':<12}")
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
                f.write(f"Number of test cases per function per implementation: {self.num_tests}\n")
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
                    
                    func_names = ['cholesky_apply', 'block_solve', 'schur_solve']
                    for func_name in func_names:
                        if f'{func_name}_pass_rate' in summary:
                            f.write(f"\n{func_name}:\n")
                            f.write(f"  Pass rate: {summary[f'{func_name}_pass_rate']:.1f}%\n")
                            f.write(f"  Pass count: {summary.get(f'{func_name}_pass_count', 0)}/{summary.get(f'{func_name}_total_tests', 0)}\n")
                            if f'{func_name}_avg_l1' in summary:
                                f.write(f"  Avg L1 error: {summary[f'{func_name}_avg_l1']:.2e}\n")
                                f.write(f"  Avg L2 error: {summary[f'{func_name}_avg_l2']:.2e}\n")
                                f.write(f"  Avg max error: {summary[f'{func_name}_avg_max']:.2e}\n")
                                f.write(f"  Avg time: {summary[f'{func_name}_avg_time']:.4f}s\n")
                    
                    # Write overall statistics
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    overall_rate = summary.get('overall_pass_rate', 0.0)
                    f.write(f"\nOverall Average Pass Rate: {overall_rate:.1f}% ({pass_count}/{test_count} tests passed)\n")
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
                    
                    func_names = ['cholesky_apply', 'block_solve', 'schur_solve']
                    for func_name in func_names:
                        if f'{func_name}_pass_rate' in summary:
                            pass_rate = f"{summary[f'{func_name}_pass_rate']:.1f}%"
                            
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

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured summary aligned with schema.json."""
        if not all_results:
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
        for summary in all_results:
            name = summary.get("implementation", "")
            test_total = summary.get("total_test_count", summary.get("total_tests", 0)) or 0
            test_pass = summary.get("total_pass_count", 0) or 0

            implementations.append(
                {
                    "name": name,
                    "test_total": int(test_total),
                    "test_pass": int(test_pass),
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

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return str(output_path)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for chol.py functions')
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
