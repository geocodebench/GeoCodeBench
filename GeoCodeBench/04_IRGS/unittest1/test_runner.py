"""
Test Runner for rotation_between_z function
Supports batch testing of multiple LLM implementations.
"""

import torch
import os
import sys
import importlib.util
import time
import json
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import reference implementation
try:
    from reference_implementation import rotation_between_z as ref_rotation_between_z
    from test_generator import TestDataGenerator
except ImportError:
    print("Error: reference_implementation.py or test_generator.py not found!")
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
            
            if not hasattr(module, 'rotation_between_z'):
                raise AttributeError(f"No rotation_between_z function found in {filepath}")
            
            return module.rotation_between_z
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        # Check type
        if not isinstance(output, torch.Tensor):
            metrics['error'] = f"Output is not a torch.Tensor (got {type(output)})"
            return metrics
        
        # Check shape
        if output.shape != reference.shape:
            metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
            return metrics
        
        # Convert to float for comparison
        output_f = output.float()
        reference_f = reference.float()
        
        # L1 error (Mean Absolute Error)
        l1_error = torch.mean(torch.abs(output_f - reference_f)).item()
        metrics['l1_error'] = l1_error
        
        # L2 error (Root Mean Square Error)
        l2_error = torch.sqrt(torch.mean((output_f - reference_f) ** 2)).item()
        metrics['l2_error'] = l2_error
        
        # Max error
        max_error = torch.max(torch.abs(output_f - reference_f)).item()
        metrics['max_error'] = max_error
        
        # Relative error (avoid division by zero)
        ref_norm = torch.norm(reference_f)
        if ref_norm > 1e-10:
            relative_error = (torch.norm(output_f - reference_f) / ref_norm).item() * 100
        else:
            relative_error = 0.0 if max_error < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass (within tolerance)
        metrics['pass'] = max_error < self.tolerance
        
        return metrics
    
    def verify_rotation_matrix(self, R, vec):
        """Verify that rotation matrix correctly rotates z-axis to vec."""
        # R should rotate [0, 0, 1] to direction of vec
        z_axis = torch.zeros(vec.shape, device=vec.device)
        z_axis[..., 2] = 1.0
        
        # Apply rotation
        rotated = torch.matmul(R, z_axis.unsqueeze(-1)).squeeze(-1)
        
        # Normalize both vectors for comparison
        vec_norm = torch.nn.functional.normalize(vec, dim=-1)
        rotated_norm = torch.nn.functional.normalize(rotated, dim=-1)
        
        # Compute cosine similarity (should be close to 1)
        cos_sim = torch.sum(vec_norm * rotated_norm, dim=-1)
        
        return {
            'mean_cos_sim': cos_sim.mean().item(),
            'min_cos_sim': cos_sim.min().item(),
        }
    
    def test_rotation_between_z(self, impl_func, test_case):
        """Test rotation_between_z function."""
        vec = test_case['vec']
        
        try:
            start_time = time.time()
            output = impl_func(vec)
            exec_time = time.time() - start_time
            
            reference = ref_rotation_between_z(vec)
            metrics = self.compute_error(output, reference)
            metrics['execution_time'] = exec_time
            
            # Additional verification
            if metrics.get('pass', False):
                verification = self.verify_rotation_matrix(output, vec)
                metrics.update(verification)
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
            
            result = self.test_rotation_between_z(impl_func, test_case)
            
            test_result = {
                'test_idx': i,
                'description': test_case['description'],
                'vec_shape': tuple(test_case['vec'].shape),
                'result': result
            }
            
            if self.verbose:
                if result.get('pass', False):
                    cos_sim = result.get('mean_cos_sim', 1.0)
                    print(f"  ✓ Pass (L1={result.get('l1_error', 0):.2e}, L2={result.get('l2_error', 0):.2e}, "
                          f"cos_sim={cos_sim:.6f}, time={result.get('execution_time', 0):.4f}s)")
                else:
                    print(f"  ✗ Fail - {result.get('error', 'Error exceeds tolerance')}")
            
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
        exec_times = []
        cos_sims = []
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                l1_errors.append(result.get('l1_error', 0))
                l2_errors.append(result.get('l2_error', 0))
                exec_times.append(result.get('execution_time', 0))
                if 'mean_cos_sim' in result:
                    cos_sims.append(result['mean_cos_sim'])
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
                summary['avg_time'] = sum(exec_times) / len(exec_times)
            
            if cos_sims:
                summary['avg_cos_sim'] = sum(cos_sims) / len(cos_sims)
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
            print(f"  Avg time: {summary['avg_time']:.4f}s")
        
        if 'avg_cos_sim' in summary:
            print(f"  Avg cosine similarity: {summary['avg_cos_sim']:.6f}")
        
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

        # Save structured summary to JSON
        self.save_summary_to_file(all_summaries)
        
        return all_summaries

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured test summary aligned with schema.json."""
        if output_path is None:
            output_path = Path(__file__).parent / "test_summary.json"
        else:
            output_path = Path(output_path)

        script_dir = Path(__file__).parent
        project_id = script_dir.parent.name
        unittest_id = script_dir.name.replace("unittest", "")

        implementations = []
        for summary in all_results or []:
            name = summary.get("implementation", "unknown")
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
                "suite_path": f"{project_id}/{script_dir.name}",
                "num_tests_requested": int(self.num_tests),
            },
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": implementations,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(output_path)
    
    def print_comparison(self, all_summaries):
        """Print comparison table."""
        if not all_summaries:
            return
        
        print(f"\n{'='*100}")
        print("COMPARISON SUMMARY")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg CosSim':<12} {'Avg Time':<12}")
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
            avg_cos = f"{summary.get('avg_cos_sim', 0):.6f}" if 'avg_cos_sim' in summary else "N/A"
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
            print(f"{name:<25} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_cos:<12} {avg_time:<12}")
        
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
                f.write("="*100 + "\n\n")
                
                # Write detailed results for each implementation
                for summary in all_summaries:
                    f.write("="*80 + "\n")
                    f.write(f"Implementation: {summary['implementation']}\n")
                    f.write("="*80 + "\n")
                    
                    # Check if there was an error loading
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"ERROR: {summary['error']}\n")
                        f.write("Pass Rate: 0/0 (0.0%)\n")
                        f.write("\n")
                        continue
                    
                    # Write test summary
                    total_tests = summary.get('total_tests', 0)
                    passed_tests = summary.get('total_pass_count', 0)
                    failed_tests = max(total_tests - passed_tests, 0)
                    f.write(f"Total tests: {total_tests}\n")
                    f.write(f"Successful tests: {passed_tests}\n")
                    f.write(f"Failed tests: {failed_tests}\n")
                    f.write(f"Pass rate: {summary.get('pass_rate', 0.0):.1f}%\n\n")
                    
                    # Write detailed test results if available
                    if 'results' in summary:
                        f.write("Detailed Test Results:\n")
                        f.write("-" * 80 + "\n")
                        for i, test_result in enumerate(summary['results'], 1):
                            result = test_result.get('result', {})
                            test_desc = test_result.get('description', f'Test {i}')
                            f.write(f"\nTest {i}: {test_desc}\n")
                            
                            if result.get('pass', False):
                                f.write("  ✓ Success\n")
                                f.write(f"    Shape: {test_result.get('vec_shape')}\n")
                                f.write(f"    Execution time: {result.get('execution_time', 0):.4f}s\n")
                                f.write(f"    L1: {result.get('l1_error', 0):.2e}\n")
                                f.write(f"    L2: {result.get('l2_error', 0):.2e}\n")
                                f.write(f"    Max error: {result.get('max_error', 0):.2e}\n")
                                if 'mean_cos_sim' in result:
                                    f.write(f"    Mean cosine similarity: {result['mean_cos_sim']:.6f}\n")
                            else:
                                f.write(f"  ✗ Failed: {result.get('error', 'Error exceeds tolerance')}\n")
                                if result.get('max_error') is not None:
                                    f.write(f"    Max error: {result.get('max_error', 0):.2e}\n")
                                if result.get('execution_time', 0) > 0:
                                    f.write(f"    Execution time: {result.get('execution_time', 0):.4f}s\n")
                        
                        f.write("\n" + "-" * 80 + "\n\n")
                    
                    # Write statistics summary if available
                    if summary.get('total_pass_count', 0) > 0:
                        f.write("Summary Statistics:\n")
                        if 'avg_l1' in summary:
                            f.write(f"  Average L1 error: {summary['avg_l1']:.2e}\n")
                        if 'avg_l2' in summary:
                            f.write(f"  Average L2 error: {summary['avg_l2']:.2e}\n")
                        if 'avg_time' in summary:
                            f.write(f"  Average execution time: {summary['avg_time']:.4f}s\n")
                        if 'avg_cos_sim' in summary:
                            f.write(f"  Average cosine similarity: {summary['avg_cos_sim']:.6f}\n")
                    
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg CosSim':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation'][:23]
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<25} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        continue
                    
                    pass_rate = f"{summary.get('pass_rate', 0.0):.1f}%"
                    avg_l1 = f"{summary.get('avg_l1', 0):.2e}" if 'avg_l1' in summary else "N/A"
                    avg_l2 = f"{summary.get('avg_l2', 0):.2e}" if 'avg_l2' in summary else "N/A"
                    avg_cos = f"{summary.get('avg_cos_sim', 0):.6f}" if 'avg_cos_sim' in summary else "N/A"
                    avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
                    
                    f.write(f"{name:<25} {pass_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_cos:<12} {avg_time:<12}\n")
                
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
                    f.write(f"{i}. {name:<30} {overall_rate:<20} {count_str:<15}\n")
                f.write("-" * 57 + "\n")
            
            print(f"\n✓ Results saved to: {output_file}")
            return str(output_file)
        
        except Exception as e:
            print(f"\n✗ Error saving results to file: {e}")
            return None



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for rotation_between_z')
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
