"""
Test Runner for compute_hom function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import torch
import json
import os
import sys
import importlib.util
import time
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime, timezone

from reference_implementation import compute_hom as ref_compute_hom
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-4):
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
            
            if not hasattr(module, 'compute_hom'):
                raise AttributeError(f"No compute_hom function found in {filepath}")
            
            return module.compute_hom
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        try:
            # Unpack outputs
            ncc_out, mask_out, patches_out = output
            ncc_ref, mask_ref, patches_ref = reference
            
            # Check types
            if not isinstance(ncc_out, torch.Tensor):
                metrics['error'] = f"ncc output is not a Tensor (got {type(ncc_out)})"
                return metrics
            
            if not isinstance(mask_out, torch.Tensor):
                metrics['error'] = f"mask output is not a Tensor (got {type(mask_out)})"
                return metrics
            
            if not isinstance(patches_out, torch.Tensor):
                metrics['error'] = f"patches output is not a Tensor (got {type(patches_out)})"
                return metrics
            
            # Check shapes
            if ncc_out.shape != ncc_ref.shape:
                metrics['error'] = f"ncc shape mismatch: {ncc_out.shape} vs {ncc_ref.shape}"
                return metrics
            
            if mask_out.shape != mask_ref.shape:
                metrics['error'] = f"mask shape mismatch: {mask_out.shape} vs {mask_ref.shape}"
                return metrics
            
            if patches_out.shape != patches_ref.shape:
                metrics['error'] = f"patches shape mismatch: {patches_out.shape} vs {patches_ref.shape}"
                return metrics
            
            # Compute errors for ncc
            ncc_l1_error = torch.mean(torch.abs(ncc_out - ncc_ref)).item()
            ncc_l2_error = torch.sqrt(torch.mean((ncc_out - ncc_ref) ** 2)).item()
            ncc_max_error = torch.max(torch.abs(ncc_out - ncc_ref)).item()
            
            # Compute errors for mask (boolean comparison)
            mask_accuracy = (mask_out == mask_ref).float().mean().item() * 100
            
            # Compute errors for patches
            patches_l1_error = torch.mean(torch.abs(patches_out - patches_ref)).item()
            patches_l2_error = torch.sqrt(torch.mean((patches_out - patches_ref) ** 2)).item()
            patches_max_error = torch.max(torch.abs(patches_out - patches_ref)).item()
            
            # Overall metrics
            metrics['ncc_l1_error'] = ncc_l1_error
            metrics['ncc_l2_error'] = ncc_l2_error
            metrics['ncc_max_error'] = ncc_max_error
            metrics['mask_accuracy'] = mask_accuracy
            metrics['patches_l1_error'] = patches_l1_error
            metrics['patches_l2_error'] = patches_l2_error
            metrics['patches_max_error'] = patches_max_error
            
            # Compute relative error
            ncc_ref_norm = torch.norm(ncc_ref)
            if ncc_ref_norm > 1e-10:
                ncc_relative_error = (torch.norm(ncc_out - ncc_ref) / ncc_ref_norm).item() * 100
            else:
                ncc_relative_error = 0.0 if ncc_max_error < self.tolerance else 100.0
            metrics['ncc_relative_error'] = ncc_relative_error
            
            patches_ref_norm = torch.norm(patches_ref)
            if patches_ref_norm > 1e-10:
                patches_relative_error = (torch.norm(patches_out - patches_ref) / patches_ref_norm).item() * 100
            else:
                patches_relative_error = 0.0 if patches_max_error < self.tolerance else 100.0
            metrics['patches_relative_error'] = patches_relative_error
            
            # Check if pass (within tolerance)
            # Consider passing if all major errors are within tolerance
            metrics['pass'] = (ncc_max_error < self.tolerance and 
                             patches_max_error < self.tolerance and 
                             mask_accuracy > 95.0)
            
        except Exception as e:
            metrics['error'] = f"Error computing metrics: {str(e)}"
            metrics['pass'] = False
        
        return metrics
    
    def test_compute_hom(self, impl_func, test_case):
        """Test compute_hom function."""
        depth = test_case['depth']
        normal = test_case['normal']
        points = test_case['points']
        view_ref = test_case['view_ref']
        view_src = test_case['view_src']
        patch_size = test_case['patch_size']
        patch_offset = test_case['patch_offset']
        
        try:
            start_time = time.time()
            output = impl_func(depth, normal, points, view_ref, view_src, 
                             patch_size=patch_size, patch_offset=patch_offset)
            exec_time = time.time() - start_time
            
            reference = ref_compute_hom(depth, normal, points, view_ref, view_src, 
                                       patch_size=patch_size, patch_offset=patch_offset)
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
            
            result = self.test_compute_hom(impl_func, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (ncc_L1={result.get('ncc_l1_error', 0):.2e}, "
                          f"patches_L1={result.get('patches_l1_error', 0):.2e}, "
                          f"mask_acc={result.get('mask_accuracy', 0):.1f}%, "
                          f"time={result.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
                    if 'ncc_max_error' in result:
                        print(f"    ncc max error: {result['ncc_max_error']:.2e}")
                        print(f"    patches max error: {result['patches_max_error']:.2e}")
                        print(f"    mask accuracy: {result.get('mask_accuracy', 0):.1f}%")
            
            all_results.append(test_result)
        
        # Compute summary
        summary = self.compute_summary(impl_name, all_results)
        
        if self.verbose:
            self.print_summary(summary)
        
        return summary

    def test_single_implementation_in_subprocess(self, impl_path):
        """Run one implementation in subprocess to isolate native crashes."""
        impl_name = Path(impl_path).stem
        script_path = Path(__file__).resolve()

        cmd = [
            sys.executable,
            str(script_path),
            "--num-tests",
            str(self.num_tests),
            "--tolerance",
            str(self.tolerance),
            "--impl-dir",
            str(Path(impl_path).parent),
            "--single-impl",
            str(impl_path),
            "--quiet",
        ]

        # Keep a generous timeout to avoid hanging implementations.
        timeout_seconds = max(60, int(self.num_tests * 30))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "implementation": impl_name,
                "error": f"Subprocess timeout after {timeout_seconds}s",
                "overall_pass_rate": 0.0,
                "total_pass_count": 0,
                "total_test_count": 0,
            }
        except Exception as e:
            return {
                "implementation": impl_name,
                "error": f"Subprocess launch failed: {type(e).__name__}: {e}",
                "overall_pass_rate": 0.0,
                "total_pass_count": 0,
                "total_test_count": 0,
            }

        stdout_lines = (proc.stdout or "").splitlines()
        marker = "__SINGLE_IMPL_RESULT__"
        marker_idx = None
        for i, line in enumerate(stdout_lines):
            if line.strip() == marker:
                marker_idx = i
                break

        if marker_idx is not None and marker_idx + 1 < len(stdout_lines):
            payload_line = stdout_lines[marker_idx + 1].strip()
            try:
                return json.loads(payload_line)
            except json.JSONDecodeError:
                pass

        # If child crashed (e.g. segfault), surface a stable error record and continue.
        if proc.returncode < 0:
            signal_num = -proc.returncode
            return {
                "implementation": impl_name,
                "error": f"Subprocess terminated by signal {signal_num}",
                "overall_pass_rate": 0.0,
                "total_pass_count": 0,
                "total_test_count": 0,
            }

        stderr_tail = (proc.stderr or "").strip()
        if stderr_tail:
            stderr_tail = stderr_tail.splitlines()[-1]
        else:
            stderr_tail = "Unknown subprocess failure"

        return {
            "implementation": impl_name,
            "error": f"Subprocess failed (code {proc.returncode}): {stderr_tail}",
            "overall_pass_rate": 0.0,
            "total_pass_count": 0,
            "total_test_count": 0,
        }
    
    def compute_summary(self, impl_name, all_results):
        """Compute summary statistics."""
        summary = {
            'implementation': impl_name,
            'total_tests': len(all_results),
            'results': all_results
        }
        
        passes = []
        ncc_l1_errors = []
        ncc_l2_errors = []
        ncc_max_errors = []
        patches_l1_errors = []
        patches_l2_errors = []
        patches_max_errors = []
        mask_accuracies = []
        exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                ncc_l1_errors.append(result.get('ncc_l1_error', 0))
                ncc_l2_errors.append(result.get('ncc_l2_error', 0))
                ncc_max_errors.append(result.get('ncc_max_error', 0))
                patches_l1_errors.append(result.get('patches_l1_error', 0))
                patches_l2_errors.append(result.get('patches_l2_error', 0))
                patches_max_errors.append(result.get('patches_max_error', 0))
                mask_accuracies.append(result.get('mask_accuracy', 0))
                exec_times.append(result.get('execution_time', 0))
            else:
                passes.append(False)
        
        # Calculate metrics
        if passes:
            pass_rate = sum(passes) / len(passes) * 100
            summary['pass_rate'] = pass_rate
            summary['total_pass_count'] = sum(passes)
            summary['total_test_count'] = len(passes)
            
            if ncc_l1_errors:
                summary['avg_ncc_l1'] = sum(ncc_l1_errors) / len(ncc_l1_errors)
                summary['avg_ncc_l2'] = sum(ncc_l2_errors) / len(ncc_l2_errors)
                summary['avg_ncc_max'] = sum(ncc_max_errors) / len(ncc_max_errors)
                summary['avg_patches_l1'] = sum(patches_l1_errors) / len(patches_l1_errors)
                summary['avg_patches_l2'] = sum(patches_l2_errors) / len(patches_l2_errors)
                summary['avg_patches_max'] = sum(patches_max_errors) / len(patches_max_errors)
                summary['avg_mask_acc'] = sum(mask_accuracies) / len(mask_accuracies)
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
        
        if 'avg_ncc_l1' in summary:
            print(f"  Avg ncc L1 error: {summary['avg_ncc_l1']:.2e}")
            print(f"  Avg ncc L2 error: {summary['avg_ncc_l2']:.2e}")
            print(f"  Avg ncc max error: {summary['avg_ncc_max']:.2e}")
            print(f"  Avg patches L1 error: {summary['avg_patches_l1']:.2e}")
            print(f"  Avg patches max error: {summary['avg_patches_max']:.2e}")
            print(f"  Avg mask accuracy: {summary['avg_mask_acc']:.1f}%")
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
            summary = self.test_single_implementation_in_subprocess(str(impl_file))
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
        
        print(f"\n{'='*120}")
        print("COMPARISON SUMMARY")
        print(f"{'='*120}\n")
        
        # Header
        print(f"{'Implementation':<20} {'Pass Rate':<12} {'NCC L1':<12} {'NCC Max':<12} {'Patches L1':<12} {'Mask Acc':<12} {'Avg Time':<12}")
        print("-" * 120)
        
        for summary in all_summaries:
            name = summary['implementation']
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_ncc_l1 = f"{summary.get('avg_ncc_l1', 0):.2e}" if 'avg_ncc_l1' in summary else "N/A"
            avg_ncc_max = f"{summary.get('avg_ncc_max', 0):.2e}" if 'avg_ncc_max' in summary else "N/A"
            avg_patches_l1 = f"{summary.get('avg_patches_l1', 0):.2e}" if 'avg_patches_l1' in summary else "N/A"
            avg_mask_acc = f"{summary.get('avg_mask_acc', 0):.1f}%" if 'avg_mask_acc' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<20} {pass_rate:<12} {avg_ncc_l1:<12} {avg_ncc_max:<12} {avg_patches_l1:<12} {avg_mask_acc:<12} {avg_time:<12}")
        
        print("-" * 120)
        
        # Print ranking
        print(f"\n{'OVERALL RANKING':<20} {'Pass Rate':<15} {'Pass Count':<15}")
        print("-" * 52)
        sorted_summaries = sorted(all_summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
        for i, summary in enumerate(sorted_summaries, 1):
            name = summary['implementation']
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            pass_count = summary.get('total_pass_count', 0)
            test_count = summary.get('total_test_count', 0)
            count_str = f"{pass_count}/{test_count}"
            print(f"{i}. {name:<30} {overall_rate:<15} {count_str:<15}")
        print("-" * 52)

    


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
        """Save structured per-implementation test summary."""
        if output_path is None:
            output_path = Path(__file__).parent / "test_summary.json"
        else:
            output_path = Path(output_path)

        script_dir = Path(__file__).parent
        project_id = script_dir.parent.name
        unittest_id = script_dir.name.replace("unittest", "")
        suite_path = f"{project_id}/{script_dir.name}"

        implementations = []
        for summary in all_summaries or []:
            name = summary.get("implementation")
            test_total = int(summary.get("total_test_count", summary.get("total_tests", 0)) or 0)
            test_pass = int(summary.get("total_pass_count", 0) or 0)
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
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": implementations,
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"\n✗ Error saving summary to file: {e}")
            return None

        return str(output_path)



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for compute_hom')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Error tolerance for pass/fail (default: 1e-4)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--single-impl', type=str, default='',
                       help='Internal use: run only one implementation and emit JSON result')
    
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
    
    if args.single_impl:
        result = runner.test_single_implementation(args.single_impl)
        print("__SINGLE_IMPL_RESULT__")
        print(json.dumps(result, ensure_ascii=False))
        return result

    # Run tests
    results = runner.batch_test(str(impl_dir))
    
    return results


if __name__ == '__main__':
    main()

