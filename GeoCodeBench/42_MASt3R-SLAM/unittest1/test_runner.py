"""
Test Runner for project_calib() function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

import torch

from reference_implementation import project_calib as ref_project_calib
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
            
            if not hasattr(module, 'project_calib'):
                raise AttributeError(f"No project_calib function found in {filepath}")
            
            return module.project_calib
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check if output is a tuple (for jacobian=True cases)
        if isinstance(output, tuple) and isinstance(reference, tuple):
            if len(output) != len(reference):
                metrics['error'] = f"Tuple length mismatch: {len(output)} vs {len(reference)}"
                return metrics
            
            # Compute errors for each element (skip bool tensors like valid)
            total_l1 = 0
            total_l2 = 0
            total_max = 0
            total_mse = 0
            num_numeric = 0
            
            for i, (out_tensor, ref_tensor) in enumerate(zip(output, reference)):
                if not isinstance(out_tensor, torch.Tensor):
                    metrics['error'] = f"Output element {i} is not a tensor (got {type(out_tensor)})"
                    return metrics
                
                if out_tensor.shape != ref_tensor.shape:
                    metrics['error'] = f"Shape mismatch in element {i}: {out_tensor.shape} vs {ref_tensor.shape}"
                    return metrics
                
                # Skip bool tensors (like valid mask) - handle them separately
                if out_tensor.dtype == torch.bool or ref_tensor.dtype == torch.bool:
                    # Check if valid masks match
                    if not (out_tensor == ref_tensor).all().item():
                        metrics['valid_mismatch'] = True
                        metrics['valid_error_rate'] = (out_tensor != ref_tensor).float().mean().item()
                    continue
                
                # L1 error
                l1 = torch.mean(torch.abs(out_tensor - ref_tensor)).item()
                total_l1 += l1
                
                # L2 error
                l2 = torch.sqrt(torch.mean((out_tensor - ref_tensor) ** 2)).item()
                total_l2 += l2
                
                # MSE
                mse = torch.mean((out_tensor - ref_tensor) ** 2).item()
                total_mse += mse
                
                # Max error
                max_err = torch.max(torch.abs(out_tensor - ref_tensor)).item()
                total_max = max(total_max, max_err)
                num_numeric += 1
            
            # Average across numeric elements only
            if num_numeric > 0:
                metrics['l1_error'] = total_l1 / num_numeric
                metrics['l2_error'] = total_l2 / num_numeric
                metrics['mse'] = total_mse / num_numeric
                metrics['max_error'] = total_max
            else:
                metrics['l1_error'] = 0.0
                metrics['l2_error'] = 0.0
                metrics['mse'] = 0.0
                metrics['max_error'] = 0.0
            
        elif isinstance(output, tuple) and len(output) == 2:
            # Two-element tuple: (pz, valid) or (pz, dpz_dP, valid)
            # Handle valid mask separately (boolean comparison)
            if len(output) == 2 and len(reference) == 2:
                pz_out, valid_out = output
                pz_ref, valid_ref = reference
                
                # Check valid mask
                if valid_out.shape != valid_ref.shape:
                    metrics['error'] = f"Valid mask shape mismatch: {valid_out.shape} vs {valid_ref.shape}"
                    return metrics
                
                valid_match = (valid_out == valid_ref).all().item()
                if not valid_match:
                    metrics['valid_mismatch'] = True
                    metrics['valid_error_rate'] = (valid_out != valid_ref).float().mean().item()
                
                # Compute errors for pz
                if pz_out.shape != pz_ref.shape:
                    metrics['error'] = f"Shape mismatch: {pz_out.shape} vs {pz_ref.shape}"
                    return metrics
                
                metrics['l1_error'] = torch.mean(torch.abs(pz_out - pz_ref)).item()
                metrics['l2_error'] = torch.sqrt(torch.mean((pz_out - pz_ref) ** 2)).item()
                metrics['mse'] = torch.mean((pz_out - pz_ref) ** 2).item()
                metrics['max_error'] = torch.max(torch.abs(pz_out - pz_ref)).item()
                
            elif len(output) == 3 and len(reference) == 3:
                pz_out, dpz_dP_out, valid_out = output
                pz_ref, dpz_dP_ref, valid_ref = reference
                
                # Check valid mask
                if valid_out.shape != valid_ref.shape:
                    metrics['error'] = f"Valid mask shape mismatch: {valid_out.shape} vs {valid_ref.shape}"
                    return metrics
                
                valid_match = (valid_out == valid_ref).all().item()
                if not valid_match:
                    metrics['valid_mismatch'] = True
                    metrics['valid_error_rate'] = (valid_out != valid_ref).float().mean().item()
                
                # Compute errors for pz
                if pz_out.shape != pz_ref.shape:
                    metrics['error'] = f"pz shape mismatch: {pz_out.shape} vs {pz_ref.shape}"
                    return metrics
                
                pz_l1 = torch.mean(torch.abs(pz_out - pz_ref)).item()
                pz_l2 = torch.sqrt(torch.mean((pz_out - pz_ref) ** 2)).item()
                pz_mse = torch.mean((pz_out - pz_ref) ** 2).item()
                pz_max = torch.max(torch.abs(pz_out - pz_ref)).item()
                
                # Compute errors for dpz_dP
                if dpz_dP_out.shape != dpz_dP_ref.shape:
                    metrics['error'] = f"dpz_dP shape mismatch: {dpz_dP_out.shape} vs {dpz_dP_ref.shape}"
                    return metrics
                
                dpz_l1 = torch.mean(torch.abs(dpz_dP_out - dpz_dP_ref)).item()
                dpz_l2 = torch.sqrt(torch.mean((dpz_dP_out - dpz_dP_ref) ** 2)).item()
                dpz_mse = torch.mean((dpz_dP_out - dpz_dP_ref) ** 2).item()
                dpz_max = torch.max(torch.abs(dpz_dP_out - dpz_dP_ref)).item()
                
                # Average
                metrics['l1_error'] = (pz_l1 + dpz_l1) / 2
                metrics['l2_error'] = (pz_l2 + dpz_l2) / 2
                metrics['mse'] = (pz_mse + dpz_mse) / 2
                metrics['max_error'] = max(pz_max, dpz_max)
            else:
                metrics['error'] = f"Tuple length mismatch: {len(output)} vs {len(reference)}"
                return metrics
        else:
            metrics['error'] = f"Type mismatch: {type(output)} vs {type(reference)}"
            return metrics
        
        # Relative error (skip bool tensors)
        if isinstance(reference, tuple):
            ref_norm = sum(torch.norm(t).item() for t in reference 
                          if isinstance(t, torch.Tensor) and t.dtype != torch.bool)
        else:
            ref_norm = torch.norm(reference).item() if (isinstance(reference, torch.Tensor) and reference.dtype != torch.bool) else 0
        
        if ref_norm > 1e-10:
            if isinstance(output, tuple):
                out_diff_norm = sum(torch.norm(o - r).item() for o, r in zip(output, reference) 
                                  if isinstance(o, torch.Tensor) and isinstance(r, torch.Tensor)
                                  and o.dtype != torch.bool and r.dtype != torch.bool)
            else:
                if isinstance(output, torch.Tensor) and output.dtype != torch.bool:
                    out_diff_norm = torch.norm(output - reference).item()
                else:
                    out_diff_norm = 0.0
            relative_error = (out_diff_norm / ref_norm) * 100 if ref_norm > 0 else 0.0
        else:
            relative_error = 0.0 if metrics['max_error'] < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass (consider valid mask if present)
        if 'valid_mismatch' in metrics and metrics['valid_mismatch']:
            metrics['pass'] = False
        else:
            metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def test_project_calib(self, impl_func, test_case, ref_func):
        """Test project_calib function."""
        P = test_case['P']
        K = test_case['K']
        img_size = test_case['img_size']
        jacobian = test_case['jacobian']
        border = test_case['border']
        z_eps = test_case['z_eps']
        
        try:
            start_time = time.time()
            with torch.no_grad():
                output = impl_func(P, K, img_size, jacobian=jacobian, border=border, z_eps=z_eps)
            exec_time = time.time() - start_time
            
            with torch.no_grad():
                reference = ref_func(P, K, img_size, jacobian=jacobian, border=border, z_eps=z_eps)
            
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
            
            result = self.test_project_calib(impl_func, test_case, ref_project_calib)
            
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
                    if 'valid_mismatch' in result:
                        print(f"    Warning: Valid mask mismatch rate: {result.get('valid_error_rate', 0):.2%}")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
                    if 'max_error' in result:
                        print(f"    Max error: {result['max_error']:.2e} (tolerance: {self.tolerance:.2e})")
                    if 'valid_mismatch' in result:
                        print(f"    Valid mask mismatch rate: {result.get('valid_error_rate', 0):.2%}")
            
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
        mse_errors = []
        max_errors = []
        exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                l1_errors.append(result.get('l1_error', 0))
                l2_errors.append(result.get('l2_error', 0))
                mse_errors.append(result.get('mse', 0))
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
                summary['avg_mse'] = sum(mse_errors) / len(mse_errors)
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
            print(f"  Avg MSE: {summary['avg_mse']:.2e}")
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
        
        # Save structured summary in schema.json format
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
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg MSE':<12} {'Avg Max':<12} {'Avg Time':<12}")
        print("-" * 100)
        
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
                f.write(f"Device: {self.device}\n")
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
                    
                    # Write per-test details
                    for test_result in summary.get('results', []):
                        f.write(f"Test {test_result['test_idx']+1}: {test_result['description']}\n")
                        result = test_result['result']
                        if result.get('pass', False):
                            f.write(f"  ✓ Pass\n")
                            f.write(f"    L1 error: {result.get('l1_error', 0):.2e}\n")
                            f.write(f"    L2 error: {result.get('l2_error', 0):.2e}\n")
                            f.write(f"    MSE: {result.get('mse', 0):.2e}\n")
                            f.write(f"    Max error: {result.get('max_error', 0):.2e}\n")
                            f.write(f"    Time: {result.get('execution_time', 0):.4f}s\n")
                        else:
                            f.write(f"  ✗ Fail\n")
                            f.write(f"    Error: {result.get('error', 'Error exceeds tolerance')}\n")
                            if 'max_error' in result:
                                f.write(f"    Max error: {result['max_error']:.2e}\n")
                        f.write("\n")
                    
                    # Write overall statistics
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    overall_rate = summary.get('overall_pass_rate', 0.0)
                    f.write(f"Overall Average Pass Rate: {overall_rate:.1f}% ({pass_count}/{test_count} tests passed)\n")
                    if 'avg_l1' in summary:
                        f.write(f"Average L1 error: {summary['avg_l1']:.2e}\n")
                        f.write(f"Average L2 error: {summary['avg_l2']:.2e}\n")
                        f.write(f"Average MSE: {summary['avg_mse']:.2e}\n")
                        f.write(f"Average max error: {summary['avg_max']:.2e}\n")
                        f.write(f"Average time: {summary['avg_time']:.4f}s\n")
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg MSE':<12} {'Avg Max':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
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
                
                f.write("-" * 100 + "\n")
                
                # Write ranking
                f.write(f"\n{'OVERALL RANKING':<25} {'Pass Rate':<15} {'Pass Count':<15}\n")
                f.write("-" * 57 + "\n")
                sorted_summaries = sorted(all_summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
                for i, summary in enumerate(sorted_summaries, 1):
                    name = summary['implementation']
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

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured test summary aligned with schema.json."""
        if not all_results:
            return None

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
            for summary in all_results:
                name = summary.get("implementation", "unknown")

                test_total = summary.get("total_test_count")
                if test_total is None:
                    test_total = summary.get("total_tests", 0)
                test_pass = summary.get("total_pass_count", 0)

                implementations.append(
                    {
                        "name": name,
                        "test_total": int(test_total or 0),
                        "test_pass": int(test_pass or 0),
                    }
                )

            payload = {
                "suite": {
                    "project_id": project_id,
                    "unittest_id": str(unittest_id),
                    "suite_path": suite_path,
                    "num_tests_requested": int(self.num_tests),
                },
                "timestamp_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "implementations": implementations,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving test_summary.json: {e}")
            return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for project_calib()')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                       help='Error tolerance for pass/fail (default: 1e-5)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')
    
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
