"""
Test Runner for apply_distortion function
Supports batch testing of multiple LLM implementations.
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

from reference_implementation import apply_distortion as ref_apply_distortion
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
            
            if not hasattr(module, 'apply_distortion'):
                raise AttributeError(f"No apply_distortion function found in {filepath}")
            
            return module.apply_distortion
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Unpack outputs
        try:
            out_image, out_mask, out_flow = output
            ref_image, ref_mask, ref_flow = reference
        except Exception as e:
            metrics['error'] = f"Failed to unpack output: {e}"
            return metrics
        
        # Check types
        if not isinstance(out_image, torch.Tensor):
            metrics['error'] = f"Output image is not a Tensor (got {type(out_image)})"
            return metrics
        
        if not isinstance(out_mask, torch.Tensor):
            metrics['error'] = f"Output mask is not a Tensor (got {type(out_mask)})"
            return metrics
        
        if not isinstance(out_flow, torch.Tensor):
            metrics['error'] = f"Output flow is not a Tensor (got {type(out_flow)})"
            return metrics
        
        # Check shapes
        if out_image.shape != ref_image.shape:
            metrics['error'] = f"Image shape mismatch: {out_image.shape} vs {ref_image.shape}"
            return metrics
        
        if out_mask.shape != ref_mask.shape:
            metrics['error'] = f"Mask shape mismatch: {out_mask.shape} vs {ref_mask.shape}"
            return metrics
        
        if out_flow.shape != ref_flow.shape:
            metrics['error'] = f"Flow shape mismatch: {out_flow.shape} vs {ref_flow.shape}"
            return metrics
        
        # Compute errors for image
        image_l1 = torch.mean(torch.abs(out_image - ref_image)).item()
        image_l2 = torch.sqrt(torch.mean((out_image - ref_image) ** 2)).item()
        image_max = torch.max(torch.abs(out_image - ref_image)).item()
        
        # Compute errors for mask
        mask_l1 = torch.mean(torch.abs(out_mask - ref_mask)).item()
        mask_max = torch.max(torch.abs(out_mask - ref_mask)).item()
        
        # Compute errors for flow
        flow_l1 = torch.mean(torch.abs(out_flow - ref_flow)).item()
        flow_l2 = torch.sqrt(torch.mean((out_flow - ref_flow) ** 2)).item()
        flow_max = torch.max(torch.abs(out_flow - ref_flow)).item()
        
        # Overall max error
        max_error = max(image_max, mask_max, flow_max)
        
        metrics['image_l1'] = image_l1
        metrics['image_l2'] = image_l2
        metrics['image_max'] = image_max
        metrics['mask_l1'] = mask_l1
        metrics['mask_max'] = mask_max
        metrics['flow_l1'] = flow_l1
        metrics['flow_l2'] = flow_l2
        metrics['flow_max'] = flow_max
        metrics['max_error'] = max_error
        
        # Relative errors
        ref_image_norm = torch.norm(ref_image)
        if ref_image_norm > 1e-10:
            metrics['image_rel_error'] = (torch.norm(out_image - ref_image) / ref_image_norm).item() * 100
        else:
            metrics['image_rel_error'] = 0.0 if image_max < self.tolerance else 100.0
        
        ref_flow_norm = torch.norm(ref_flow)
        if ref_flow_norm > 1e-10:
            metrics['flow_rel_error'] = (torch.norm(out_flow - ref_flow) / ref_flow_norm).item() * 100
        else:
            metrics['flow_rel_error'] = 0.0 if flow_max < self.tolerance else 100.0
        
        # Check if pass (within tolerance)
        metrics['pass'] = max_error < self.tolerance
        
        return metrics
    
    def test_apply_distortion(self, impl_func, test_case):
        """Test apply_distortion function."""
        try:
            # Clone test case to avoid modification
            flow = test_case['flow_apply2_gt_or_img']
            if flow is not None:
                flow = flow.clone()
            
            start_time = time.time()
            output = impl_func(
                flow,
                test_case['lens_net'],
                test_case['P_view_insidelens_direction'],
                test_case['P_sensor'],
                test_case['viewpoint_cam'],
                test_case['image'],
                apply2gt=test_case['apply2gt'],
                flow_scale=test_case['flow_scale']
            )
            exec_time = time.time() - start_time
            
            # Get reference output
            flow_ref = test_case['flow_apply2_gt_or_img']
            if flow_ref is not None:
                flow_ref = flow_ref.clone()
            
            reference = ref_apply_distortion(
                flow_ref,
                test_case['lens_net'],
                test_case['P_view_insidelens_direction'],
                test_case['P_sensor'],
                test_case['viewpoint_cam'],
                test_case['image'],
                apply2gt=test_case['apply2gt'],
                flow_scale=test_case['flow_scale']
            )
            
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
            
            result = self.test_apply_distortion(impl_func, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (img_L1={result.get('image_l1', 0):.2e}, flow_L2={result.get('flow_l2', 0):.2e}, "
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
        image_l1s = []
        image_l2s = []
        flow_l2s = []
        max_errors = []
        exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                image_l1s.append(result.get('image_l1', 0))
                image_l2s.append(result.get('image_l2', 0))
                flow_l2s.append(result.get('flow_l2', 0))
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
            
            if image_l1s:
                summary['avg_image_l1'] = sum(image_l1s) / len(image_l1s)
                summary['avg_image_l2'] = sum(image_l2s) / len(image_l2s)
                summary['avg_flow_l2'] = sum(flow_l2s) / len(flow_l2s)
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
        
        if 'avg_image_l1' in summary:
            print(f"  Avg image L1 error: {summary['avg_image_l1']:.2e}")
            print(f"  Avg image L2 error: {summary['avg_image_l2']:.2e}")
            print(f"  Avg flow L2 error: {summary['avg_flow_l2']:.2e}")
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
        
        # Save structured summary for aggregation
        self.save_summary_to_file(all_summaries)
        
        return all_summaries
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*110}")
        print("COMPARISON SUMMARY")
        print(f"{'='*110}\n")
        
        # Header
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Img L1':<12} {'Img L2':<12} {'Flow L2':<12} {'Max Err':<12} {'Time':<12}")
        print("-" * 110)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            img_l1 = f"{summary.get('avg_image_l1', 0):.2e}" if 'avg_image_l1' in summary else "N/A"
            img_l2 = f"{summary.get('avg_image_l2', 0):.2e}" if 'avg_image_l2' in summary else "N/A"
            flow_l2 = f"{summary.get('avg_flow_l2', 0):.2e}" if 'avg_flow_l2' in summary else "N/A"
            max_err = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {img_l1:<12} {img_l2:<12} {flow_l2:<12} {max_err:<12} {avg_time:<12}")
        
        print("-" * 110)
        
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
        """Save structured test summary JSON aligned with schema.json."""
        try:
            script_dir = Path(__file__).parent
            project_id = script_dir.parent.name
            unittest_id = script_dir.name.replace("unittest", "")
            suite_path = f"{project_id}/{script_dir.name}"
            
            if output_path is None:
                output_path = script_dir / "test_summary.json"
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            implementations = []
            for summary in all_summaries or []:
                name = summary.get("implementation", "")
                test_total = summary.get("total_test_count", summary.get("total_tests", 0))
                test_pass = summary.get("total_pass_count", 0)
                
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
                    "num_tests_requested": int(self.num_tests),
                },
                "timestamp_utc": timestamp_utc,
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
    
    parser = argparse.ArgumentParser(description='Test runner for apply_distortion')
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

