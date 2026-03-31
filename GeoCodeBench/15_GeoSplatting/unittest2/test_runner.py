"""
Test Runner for make() function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import json
import torch
import os
import sys
import importlib.util
import time
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementation and test generator
try:
    from reference_implementation import MGAdapter as RefMGAdapter
except ImportError as e:
    print(f"Error: Could not import from reference_implementation.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
try:
    from test_generator import TestDataGenerator
except ImportError as e:
    print(f"Error: test_generator.py not found: {e}")
    import traceback
    traceback.print_exc()
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
            # LLM implementations now have fallback imports, so they should work directly
            # No need for mocking since each file has try/except ImportError fallback
            import sys
            
            # Get module name from file path
            module_name = Path(filepath).stem
            if module_name in sys.modules:
                # If already loaded, get it
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load spec for {filepath}")
                
                module = importlib.util.module_from_spec(spec)
                # Set __file__ and __name__ to fix dataclass issues
                module.__file__ = filepath
                module.__name__ = module_name
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            
            if not hasattr(module, 'MGAdapter'):
                raise AttributeError(f"No MGAdapter class found in {filepath}")
            
            return module.MGAdapter
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_error(self, output_splats, output_offsets, reference_splats, reference_offsets):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        try:
            # Check if outputs are tuples
            if not isinstance(output_splats, type(reference_splats)):
                metrics['error'] = f"Output splats type mismatch: {type(output_splats)} vs {type(reference_splats)}"
                return metrics
            
            # Check shapes
            if output_splats.means.shape != reference_splats.means.shape:
                metrics['error'] = f"Splats means shape mismatch: {output_splats.means.shape} vs {reference_splats.means.shape}"
                return metrics
            
            if output_offsets.shape != reference_offsets.shape:
                metrics['error'] = f"Offsets shape mismatch: {output_offsets.shape} vs {reference_offsets.shape}"
                return metrics
            
            # Compare splats components
            splats_errors = {}
            
            # Means error
            means_diff = output_splats.means - reference_splats.means
            splats_errors['means_l1'] = torch.mean(torch.abs(means_diff)).item()
            splats_errors['means_l2'] = torch.sqrt(torch.mean(means_diff ** 2)).item()
            splats_errors['means_max'] = torch.max(torch.abs(means_diff)).item()
            
            # Scales error
            scales_diff = output_splats.scales - reference_splats.scales
            splats_errors['scales_l1'] = torch.mean(torch.abs(scales_diff)).item()
            splats_errors['scales_l2'] = torch.sqrt(torch.mean(scales_diff ** 2)).item()
            splats_errors['scales_max'] = torch.max(torch.abs(scales_diff)).item()
            
            # Quats error
            quats_diff = output_splats.quats - reference_splats.quats
            splats_errors['quats_l1'] = torch.mean(torch.abs(quats_diff)).item()
            splats_errors['quats_l2'] = torch.sqrt(torch.mean(quats_diff ** 2)).item()
            splats_errors['quats_max'] = torch.max(torch.abs(quats_diff)).item()
            
            # Colors error
            colors_diff = output_splats.colors - reference_splats.colors
            splats_errors['colors_l1'] = torch.mean(torch.abs(colors_diff)).item()
            splats_errors['colors_l2'] = torch.sqrt(torch.mean(colors_diff ** 2)).item()
            splats_errors['colors_max'] = torch.max(torch.abs(colors_diff)).item()
            
            # Opacities error
            opacities_diff = output_splats.opacities - reference_splats.opacities
            splats_errors['opacities_l1'] = torch.mean(torch.abs(opacities_diff)).item()
            splats_errors['opacities_l2'] = torch.sqrt(torch.mean(opacities_diff ** 2)).item()
            splats_errors['opacities_max'] = torch.max(torch.abs(opacities_diff)).item()
            
            # Overall max error for splats
            splats_max_error = max(
                splats_errors['means_max'],
                splats_errors['scales_max'],
                splats_errors['quats_max'],
                splats_errors['colors_max'],
                splats_errors['opacities_max'],
            )
            
            # Offsets error
            offsets_diff = output_offsets - reference_offsets
            offsets_l1 = torch.mean(torch.abs(offsets_diff)).item()
            offsets_l2 = torch.sqrt(torch.mean(offsets_diff ** 2)).item()
            offsets_max = torch.max(torch.abs(offsets_diff)).item()
            
            # Relative errors (if reference is not zero)
            ref_norm = torch.norm(reference_splats.means)
            if ref_norm > 1e-10:
                relative_error = (torch.norm(output_splats.means - reference_splats.means) / ref_norm).item() * 100
            else:
                relative_error = 0.0 if splats_max_error < self.tolerance else 100.0
            
            metrics.update(splats_errors)
            metrics['offsets_l1'] = offsets_l1
            metrics['offsets_l2'] = offsets_l2
            metrics['offsets_max'] = offsets_max
            metrics['relative_error'] = relative_error
            metrics['max_error'] = max(splats_max_error, offsets_max)
            
            # Check if pass (within tolerance)
            metrics['pass'] = metrics['max_error'] < self.tolerance
            
        except Exception as e:
            metrics['error'] = f"Error computing metrics: {str(e)}"
            metrics['pass'] = False
            import traceback
            traceback.print_exc()
        
        return metrics
    
    def test_make(self, impl_class, test_case):
        """Test make() function."""
        mesh = test_case['mesh']
        normal_interpolation = test_case['normal_interpolation']
        
        try:
            start_time = time.time()
            adapter = impl_class()
            output_splats, output_offsets = adapter.make(mesh, normal_interpolation=normal_interpolation)
            exec_time = time.time() - start_time
            
            ref_adapter = RefMGAdapter()
            reference_splats, reference_offsets = ref_adapter.make(mesh, normal_interpolation=normal_interpolation)
            
            metrics = self.compute_error(output_splats, output_offsets, reference_splats, reference_offsets)
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
            import traceback
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
            # If loading fails, still report the expected number of tests
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': len(self.test_cases),  # Report expected test count
                'total_tests': len(self.test_cases)
            }
        
        all_results = []
        
        # Run all test cases
        for i, test_case in enumerate(self.test_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(self.test_cases)}: {test_case['description']}")
            
            result = self.test_make(impl_class, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (max_error={result.get('max_error', 0):.2e}, "
                          f"time={result.get('execution_time', 0):.4f}s)")
                else:
                    error_msg = result.get('error', 'Error exceeds tolerance')
                    print(f"  ✗ Fail - {error_msg}")
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
        max_errors = []
        exec_times = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
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
            
            if max_errors:
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
        
        if 'avg_max' in summary:
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
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg Max Error':<15} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<15} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_max:<15} {avg_time:<12}")
        
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

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured test summary aligned with `schema.json`."""
        if not all_results:
            return None

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
        for summary in all_results:
            impl_name = summary.get("implementation")

            test_total = summary.get("total_test_count", summary.get("total_tests", 0))
            if test_total in (None, "", "N/A"):
                test_total = 0
            try:
                test_total = int(test_total)
            except Exception:
                test_total = 0

            test_pass = summary.get("total_pass_count", 0)
            try:
                test_pass = int(test_pass)
            except Exception:
                test_pass = 0

            implementations.append(
                {
                    "name": impl_name,
                    "test_total": test_total,
                    "test_pass": test_pass,
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
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving test_summary.json: {e}")
            return None



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for make() function')
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

