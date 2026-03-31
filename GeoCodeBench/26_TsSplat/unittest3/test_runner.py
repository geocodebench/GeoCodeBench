"""
Test Runner for _compute_vertex_normal and _compute_vertex_tangent functions
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import torch
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

from reference_implementation import _compute_vertex_normal as ref_compute_vertex_normal
from reference_implementation import _compute_vertex_tangent as ref_compute_vertex_tangent
from test_generator import TestDataGenerator, MockObject


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=10, verbose=True, tolerance=1e-4):
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
            
            if not hasattr(module, '_compute_vertex_normal'):
                raise AttributeError(f"No _compute_vertex_normal function found in {filepath}")
            if not hasattr(module, '_compute_vertex_tangent'):
                raise AttributeError(f"No _compute_vertex_tangent function found in {filepath}")
            
            return module._compute_vertex_normal, module._compute_vertex_tangent
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def compute_error(self, output, reference, metric_name="output"):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check type
        if not isinstance(output, torch.Tensor):
            metrics['error'] = f"{metric_name} is not a Tensor (got {type(output)})"
            return metrics
        
        # Check shape
        if output.shape != reference.shape:
            metrics['error'] = f"{metric_name} shape mismatch: {output.shape} vs {reference.shape}"
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
    
    def test_functions(self, normal_func, tangent_func, test_case):
        """Test both vertex normal and tangent functions."""
        results = {}
        
        # Test _compute_vertex_normal
        try:
            mock_obj = MockObject(
                v_pos=test_case['v_pos'],
                t_pos_idx=test_case['t_pos_idx']
            )
            
            start_time = time.time()
            output_normal = normal_func(mock_obj)
            normal_time = time.time() - start_time
            
            reference_normal = ref_compute_vertex_normal(mock_obj)
            normal_metrics = self.compute_error(output_normal, reference_normal, "vertex_normal")
            normal_metrics['execution_time'] = normal_time
            results['normal'] = normal_metrics
        except Exception as e:
            results['normal'] = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        # Test _compute_vertex_tangent
        try:
            mock_obj = MockObject(
                v_pos=test_case['v_pos'],
                t_pos_idx=test_case['t_pos_idx'],
                v_tex=test_case['v_tex'],
                t_tex_idx=test_case['t_tex_idx'],
                v_nrm=test_case['v_nrm']
            )
            
            start_time = time.time()
            output_tangent = tangent_func(mock_obj)
            tangent_time = time.time() - start_time
            
            reference_tangent = ref_compute_vertex_tangent(mock_obj)
            tangent_metrics = self.compute_error(output_tangent, reference_tangent, "vertex_tangent")
            tangent_metrics['execution_time'] = tangent_time
            results['tangent'] = tangent_metrics
        except Exception as e:
            results['tangent'] = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
        
        return results
    
    def test_single_implementation(self, impl_path):
        """Test a single LLM implementation."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # Load implementation
        normal_func, tangent_func = self.load_llm_implementation(impl_path)
        
        if normal_func is None or tangent_func is None:
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
            
            result = self.test_functions(normal_func, tangent_func, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                # Print normal results
                normal_res = result['normal']
                if normal_res.get('pass', False):
                    print(f"  Normal: ✓ Pass (L1={normal_res.get('l1_error', 0):.2e}, "
                          f"L2={normal_res.get('l2_error', 0):.2e}, "
                          f"max={normal_res.get('max_error', 0):.2e}, "
                          f"time={normal_res.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  Normal: ✗ Fail - {normal_res.get('error', 'Error exceeds tolerance')}")
                    if 'max_error' in normal_res:
                        print(f"    Max error: {normal_res['max_error']:.2e} (tolerance: {self.tolerance:.2e})")
                
                # Print tangent results
                tangent_res = result['tangent']
                if tangent_res.get('pass', False):
                    print(f"  Tangent: ✓ Pass (L1={tangent_res.get('l1_error', 0):.2e}, "
                          f"L2={tangent_res.get('l2_error', 0):.2e}, "
                          f"max={tangent_res.get('max_error', 0):.2e}, "
                          f"time={tangent_res.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  Tangent: ✗ Fail - {tangent_res.get('error', 'Error exceeds tolerance')}")
                    if 'max_error' in tangent_res:
                        print(f"    Max error: {tangent_res['max_error']:.2e} (tolerance: {self.tolerance:.2e})")
            
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
        
        normal_passes = []
        tangent_passes = []
        normal_l1 = []
        normal_l2 = []
        normal_max = []
        normal_time = []
        tangent_l1 = []
        tangent_l2 = []
        tangent_max = []
        tangent_time = []
        
        for test_result in all_results:
            result = test_result['result']
            
            # Normal results
            if result['normal'].get('pass', False):
                normal_passes.append(True)
                normal_l1.append(result['normal'].get('l1_error', 0))
                normal_l2.append(result['normal'].get('l2_error', 0))
                normal_max.append(result['normal'].get('max_error', 0))
                normal_time.append(result['normal'].get('execution_time', 0))
            else:
                normal_passes.append(False)
            
            # Tangent results
            if result['tangent'].get('pass', False):
                tangent_passes.append(True)
                tangent_l1.append(result['tangent'].get('l1_error', 0))
                tangent_l2.append(result['tangent'].get('l2_error', 0))
                tangent_max.append(result['tangent'].get('max_error', 0))
                tangent_time.append(result['tangent'].get('execution_time', 0))
            else:
                tangent_passes.append(False)
        
        # Calculate metrics
        if normal_passes:
            summary['normal_pass_rate'] = sum(normal_passes) / len(normal_passes) * 100
            summary['normal_pass_count'] = sum(normal_passes)
            if normal_l1:
                summary['normal_avg_l1'] = sum(normal_l1) / len(normal_l1)
                summary['normal_avg_l2'] = sum(normal_l2) / len(normal_l2)
                summary['normal_avg_max'] = sum(normal_max) / len(normal_max)
                summary['normal_avg_time'] = sum(normal_time) / len(normal_time)
        else:
            summary['normal_pass_rate'] = 0.0
            summary['normal_pass_count'] = 0
        
        if tangent_passes:
            summary['tangent_pass_rate'] = sum(tangent_passes) / len(tangent_passes) * 100
            summary['tangent_pass_count'] = sum(tangent_passes)
            if tangent_l1:
                summary['tangent_avg_l1'] = sum(tangent_l1) / len(tangent_l1)
                summary['tangent_avg_l2'] = sum(tangent_l2) / len(tangent_l2)
                summary['tangent_avg_max'] = sum(tangent_max) / len(tangent_max)
                summary['tangent_avg_time'] = sum(tangent_time) / len(tangent_time)
        else:
            summary['tangent_pass_rate'] = 0.0
            summary['tangent_pass_count'] = 0
        
        # Overall metrics
        total_passes = sum(normal_passes) + sum(tangent_passes)
        total_tests = len(normal_passes) + len(tangent_passes)
        summary['overall_pass_rate'] = (total_passes / total_tests * 100) if total_tests > 0 else 0.0
        summary['total_pass_count'] = total_passes
        summary['total_test_count'] = total_tests
        
        return summary
    
    def print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Summary for {summary['implementation']}:")
        print(f"  Total tests: {summary['total_tests']}")
        
        print(f"\n  Vertex Normal:")
        print(f"    Pass rate: {summary.get('normal_pass_rate', 0.0):.1f}%")
        if 'normal_avg_l1' in summary:
            print(f"    Avg L1 error: {summary['normal_avg_l1']:.2e}")
            print(f"    Avg L2 error: {summary['normal_avg_l2']:.2e}")
            print(f"    Avg max error: {summary['normal_avg_max']:.2e}")
            print(f"    Avg time: {summary['normal_avg_time']:.4f}s")
        
        print(f"\n  Vertex Tangent:")
        print(f"    Pass rate: {summary.get('tangent_pass_rate', 0.0):.1f}%")
        if 'tangent_avg_l1' in summary:
            print(f"    Avg L1 error: {summary['tangent_avg_l1']:.2e}")
            print(f"    Avg L2 error: {summary['tangent_avg_l2']:.2e}")
            print(f"    Avg max error: {summary['tangent_avg_max']:.2e}")
            print(f"    Avg time: {summary['tangent_avg_time']:.4f}s")
        
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
        
        # Save structured results to JSON
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
        print(f"{'Implementation':<20} {'Normal Pass':<13} {'Normal Avg Max':<15} "
              f"{'Tangent Pass':<14} {'Tangent Avg Max':<16} {'Overall':<12}")
        print("-" * 120)
        
        for summary in all_summaries:
            name = summary['implementation']
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<20} {'N/A':<13} {'N/A':<15} {'N/A':<14} {'N/A':<16} {'0.0%':<12}")
                continue
            
            normal_pass = f"{summary.get('normal_pass_rate', 0.0):.1f}%"
            normal_max = f"{summary.get('normal_avg_max', 0):.2e}" if 'normal_avg_max' in summary else "N/A"
            tangent_pass = f"{summary.get('tangent_pass_rate', 0.0):.1f}%"
            tangent_max = f"{summary.get('tangent_avg_max', 0):.2e}" if 'tangent_avg_max' in summary else "N/A"
            overall = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            
            print(f"{name:<20} {normal_pass:<13} {normal_max:<15} "
                  f"{tangent_pass:<14} {tangent_max:<16} {overall:<12}")
        
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

                    f.write(f"Total tests: {summary.get('total_tests', 0)}\n")
                    f.write(f"Vertex Normal: Pass rate {summary.get('normal_pass_rate', 0.0):.1f}%\n")
                    if 'normal_avg_l1' in summary:
                        f.write(f"  Avg L1: {summary['normal_avg_l1']:.2e}, Avg max: {summary['normal_avg_max']:.2e}, Avg time: {summary['normal_avg_time']:.4f}s\n")
                    f.write(f"Vertex Tangent: Pass rate {summary.get('tangent_pass_rate', 0.0):.1f}%\n")
                    if 'tangent_avg_l1' in summary:
                        f.write(f"  Avg L1: {summary['tangent_avg_l1']:.2e}, Avg max: {summary['tangent_avg_max']:.2e}, Avg time: {summary['tangent_avg_time']:.4f}s\n")
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    f.write(f"Overall: {summary.get('overall_pass_rate', 0.0):.1f}% ({pass_count}/{test_count} tests passed)\n")
                    f.write("\n")

                # Write comparison table (match print_comparison)
                f.write("\n" + "="*120 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*120 + "\n\n")

                f.write(f"{'Implementation':<20} {'Normal Pass':<13} {'Normal Avg Max':<15} "
                        f"{'Tangent Pass':<14} {'Tangent Avg Max':<16} {'Overall':<12}\n")
                f.write("-" * 120 + "\n")

                for summary in all_summaries:
                    name = summary['implementation']

                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<20} {'N/A':<13} {'N/A':<15} {'N/A':<14} {'N/A':<16} {'0.0%':<12}\n")
                        continue

                    normal_pass = f"{summary.get('normal_pass_rate', 0.0):.1f}%"
                    normal_max = f"{summary.get('normal_avg_max', 0):.2e}" if 'normal_avg_max' in summary else "N/A"
                    tangent_pass = f"{summary.get('tangent_pass_rate', 0.0):.1f}%"
                    tangent_max = f"{summary.get('tangent_avg_max', 0):.2e}" if 'tangent_avg_max' in summary else "N/A"
                    overall = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
                    f.write(f"{name:<20} {normal_pass:<13} {normal_max:<15} "
                            f"{tangent_pass:<14} {tangent_max:<16} {overall:<12}\n")

                f.write("-" * 120 + "\n")

                # Write ranking
                f.write(f"\n{'OVERALL RANKING':<20} {'Pass Rate':<15} {'Pass Count':<15}\n")
                f.write("-" * 52 + "\n")
                sorted_summaries = sorted(all_summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
                for i, summary in enumerate(sorted_summaries, 1):
                    name = summary['implementation']
                    overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    count_str = f"{pass_count}/{test_count}"
                    f.write(f"{i}. {name:<30} {overall_rate:<15} {count_str:<15}\n")
                f.write("-" * 52 + "\n")
            
            print(f"\n✓ Results saved to: {output_file}")
            return str(output_file)
        
        except Exception as e:
            print(f"\n✗ Error saving results to file: {e}")
            return None

    def save_summary_to_file(self, all_summaries, output_path=None):
        """Save structured per-implementation test counts to `test_summary.json`."""
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
            payload["implementations"].append(
                {
                    "name": summary.get("implementation", "unknown"),
                    "test_total": int(summary.get("total_test_count", 0)),
                    "test_pass": int(summary.get("total_pass_count", 0)),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(output_path)



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for vertex normal and tangent computation')
    parser.add_argument('--num-tests', type=int, default=10,
                       help='Number of test cases to run (default: 10, recommended: 10)')
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

