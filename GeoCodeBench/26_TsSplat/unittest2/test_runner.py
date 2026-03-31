"""
Test Runner for get_surface_vf function
Supports batch testing of multiple LLM implementations.
"""

from __future__ import annotations

import numpy as np
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

from reference_implementation import get_surface_vf as ref_get_surface_vf
from test_generator import TestDataGenerator


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-10):
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
            
            if not hasattr(module, 'get_surface_vf'):
                raise AttributeError(f"No get_surface_vf function found in {filepath}")
            
            return module.get_surface_vf
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Unpack outputs
        try:
            output_vertices, output_triangles = output
            ref_vertices, ref_triangles = reference
        except Exception as e:
            metrics['error'] = f"Failed to unpack output: {e}"
            return metrics
        
        # Check types
        if not isinstance(output_vertices, np.ndarray):
            metrics['error'] = f"output_vertices is not ndarray (got {type(output_vertices)})"
            return metrics
        
        if not isinstance(output_triangles, np.ndarray):
            metrics['error'] = f"output_triangles is not ndarray (got {type(output_triangles)})"
            return metrics
        
        # Check shapes
        if output_vertices.shape != ref_vertices.shape:
            metrics['error'] = f"vertices shape mismatch: {output_vertices.shape} vs {ref_vertices.shape}"
            return metrics
        
        if output_triangles.shape != ref_triangles.shape:
            metrics['error'] = f"triangles shape mismatch: {output_triangles.shape} vs {ref_triangles.shape}"
            return metrics
        
        # Check values - vertices should match exactly
        vertices_match = np.allclose(np.sort(output_vertices), np.sort(ref_vertices), atol=self.tolerance)
        
        if not vertices_match:
            metrics['error'] = f"Surface vertices do not match"
            max_diff = np.max(np.abs(np.sort(output_vertices) - np.sort(ref_vertices)))
            metrics['vertices_max_diff'] = float(max_diff)
            return metrics
        
        # Check triangles - need to account for potential reordering
        # Sort both arrays for comparison
        output_tri_sorted = np.sort(output_triangles, axis=0)
        ref_tri_sorted = np.sort(ref_triangles, axis=0)
        
        triangles_match = np.array_equal(output_tri_sorted, ref_tri_sorted)
        
        if not triangles_match:
            # Try more flexible comparison
            output_tri_set = set(map(tuple, np.sort(output_triangles, axis=1)))
            ref_tri_set = set(map(tuple, np.sort(ref_triangles, axis=1)))
            
            if output_tri_set == ref_tri_set:
                triangles_match = True
            else:
                metrics['error'] = f"Surface triangles do not match"
                metrics['triangles_diff_count'] = len(output_tri_set.symmetric_difference(ref_tri_set))
                return metrics
        
        # All checks passed
        metrics['vertices_l1'] = 0.0
        metrics['triangles_l1'] = 0.0
        metrics['vertices_count'] = len(output_vertices)
        metrics['triangles_count'] = len(output_triangles)
        metrics['pass'] = True
        
        return metrics
    
    def test_get_surface_vf(self, impl_func, test_case):
        """Test get_surface_vf function."""
        faces = test_case['faces']
        
        try:
            start_time = time.time()
            output = impl_func(faces)
            exec_time = time.time() - start_time
            
            reference = ref_get_surface_vf(faces)
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
            
            result = self.test_get_surface_vf(impl_func, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    print(f"  ✓ Pass (vertices={result.get('vertices_count', 0)}, "
                          f"triangles={result.get('triangles_count', 0)}, "
                          f"time={result.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Unknown error')}")
                    if 'vertices_max_diff' in result:
                        print(f"    Vertices max diff: {result['vertices_max_diff']}")
                    if 'triangles_diff_count' in result:
                        print(f"    Triangles diff count: {result['triangles_diff_count']}")
            
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
        exec_times = []
        vertices_counts = []
        triangles_counts = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                exec_times.append(result.get('execution_time', 0))
                vertices_counts.append(result.get('vertices_count', 0))
                triangles_counts.append(result.get('triangles_count', 0))
            else:
                passes.append(False)
        
        # Calculate metrics
        if passes:
            pass_rate = sum(passes) / len(passes) * 100
            summary['pass_rate'] = pass_rate
            summary['total_pass_count'] = sum(passes)
            summary['total_test_count'] = len(passes)
            
            if exec_times:
                summary['avg_time'] = sum(exec_times) / len(exec_times)
                summary['avg_vertices'] = sum(vertices_counts) / len(vertices_counts) if vertices_counts else 0
                summary['avg_triangles'] = sum(triangles_counts) / len(triangles_counts) if triangles_counts else 0
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
        
        if 'avg_time' in summary:
            print(f"  Avg time: {summary['avg_time']:.4f}s")
            print(f"  Avg vertices: {summary['avg_vertices']:.1f}")
            print(f"  Avg triangles: {summary['avg_triangles']:.1f}")
        
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
        
        # Save structured results to JSON
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
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg Vertices':<15} {'Avg Triangles':<15} {'Avg Time':<12}")
        print("-" * 100)
        
        for summary in all_summaries:
            name = summary['implementation'][:23]
            
            # Check if there was an error loading
            if 'error' in summary and 'results' not in summary:
                print(f"{name:<25} {'0.0%':<12} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
                continue
            
            pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
            avg_vertices = f"{summary.get('avg_vertices', 0):.1f}" if 'avg_vertices' in summary else "N/A"
            avg_triangles = f"{summary.get('avg_triangles', 0):.1f}" if 'avg_triangles' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_vertices:<15} {avg_triangles:<15} {avg_time:<12}")
        
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

                    f.write(f"Total tests: {summary.get('total_tests', 0)}\n")
                    f.write(f"Pass rate: {summary.get('pass_rate', 0.0):.1f}%\n")
                    if 'avg_time' in summary:
                        f.write(f"Avg time: {summary['avg_time']:.4f}s\n")
                        f.write(f"Avg vertices: {summary['avg_vertices']:.1f}\n")
                        f.write(f"Avg triangles: {summary['avg_triangles']:.1f}\n")
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    f.write(f"Overall: {summary.get('overall_pass_rate', 0.0):.1f}% ({pass_count}/{test_count} tests passed)\n")
                    f.write("\n")

                # Write comparison table (match print_comparison)
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")

                f.write(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg Vertices':<15} {'Avg Triangles':<15} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")

                for summary in all_summaries:
                    name = summary['implementation'][:23]

                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<25} {'0.0%':<12} {'N/A':<15} {'N/A':<15} {'N/A':<12}\n")
                        continue

                    pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
                    avg_vertices = f"{summary.get('avg_vertices', 0):.1f}" if 'avg_vertices' in summary else "N/A"
                    avg_triangles = f"{summary.get('avg_triangles', 0):.1f}" if 'avg_triangles' in summary else "N/A"
                    avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
                    f.write(f"{name:<25} {pass_rate:<12} {avg_vertices:<15} {avg_triangles:<15} {avg_time:<12}\n")

                f.write("-" * 100 + "\n")

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
    
    parser = argparse.ArgumentParser(description='Test runner for get_surface_vf')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                       help='Error tolerance for pass/fail (default: 1e-10)')
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

