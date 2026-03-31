"""
Test Runner for unmixField.get_outputs() function
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
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import reference implementation and test generator
try:
    from reference_implementation import ReferenceUnmixField
    from test_generator import TestDataGenerator
except ImportError as e:
    print(f"Error: Failed to import: {e}")
    sys.exit(1)


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, tolerance=1e-4, device='cpu'):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.device = device
        self.test_generator = TestDataGenerator(device=device)
    
    def load_llm_implementation(self, filepath):
        """Load LLM implementation from a file."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'unmixField'):
                raise AttributeError(f"No unmixField class found in {filepath}")
            
            return module.unmixField
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_error(self, output, reference):
        """Compute error metrics between output and reference."""
        metrics = {}
        
        if not isinstance(output, dict) or not isinstance(reference, dict):
            metrics['error'] = f"Type mismatch: expected dict, got {type(output)} vs {type(reference)}"
            return metrics
        
        # Check all keys match
        if set(output.keys()) != set(reference.keys()):
            missing = set(reference.keys()) - set(output.keys())
            extra = set(output.keys()) - set(reference.keys())
            metrics['error'] = f"Key mismatch: missing={missing}, extra={extra}"
            return metrics
        
        total_l1 = 0
        total_l2 = 0
        total_max = 0
        key_count = 0
        
        for key in reference.keys():
            out_tensor = output[key]
            ref_tensor = reference[key]
            
            if not isinstance(out_tensor, torch.Tensor):
                metrics['error'] = f"Output[{key}] is not a tensor (got {type(out_tensor)})"
                return metrics
            
            if out_tensor.shape != ref_tensor.shape:
                metrics['error'] = f"Shape mismatch in {key}: {out_tensor.shape} vs {ref_tensor.shape}"
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
            
            key_count += 1
        
        # Average across keys
        if key_count > 0:
            metrics['l1_error'] = total_l1 / key_count
            metrics['l2_error'] = total_l2 / key_count
            metrics['max_error'] = total_max
        
        # Relative error
        ref_norm = sum(torch.norm(t).item() for t in reference.values())
        if ref_norm > 1e-10:
            out_diff_norm = sum(torch.norm(output[k] - reference[k]).item() for k in reference.keys())
            relative_error = (out_diff_norm / ref_norm) * 100
        else:
            relative_error = 0.0 if metrics['max_error'] < self.tolerance else 100.0
        metrics['relative_error'] = relative_error
        
        # Check if pass
        metrics['pass'] = metrics['max_error'] < self.tolerance
        
        return metrics
    
    def create_field_model(self, impl_class, ref_model):
        """Create a field model instance with same weights as reference."""
        # Get initialization parameters from reference model
        aabb = ref_model.aabb
        num_images = ref_model.num_images if hasattr(ref_model, 'num_images') else 1
        
        # Try to get implementation type
        implementation = 'torch'
        if hasattr(ref_model, 'implementation'):
            implementation = ref_model.implementation
        elif hasattr(ref_model, 'mlp_head') and hasattr(ref_model.mlp_head, 'implementation'):
            implementation = ref_model.mlp_head.implementation
        
        # Get num_layers_color from mlp_head if available
        num_layers_color = 3
        if hasattr(ref_model, 'mlp_head') and hasattr(ref_model.mlp_head, 'num_layers'):
            num_layers_color = ref_model.mlp_head.num_layers
        
        # Create new model with same config
        model = impl_class(
            aabb=aabb,
            num_images=num_images,
            implementation=implementation,
            num_layers_color=num_layers_color,
            hidden_dim_color=64,
            wavelengths=ref_model.wavelengths,
            method=ref_model.method,
            num_classes=ref_model.num_classes,
            feature_dim=ref_model.feature_dim,
            temperature=ref_model.temperature,
            pred_specular=ref_model.pred_specular,
        )
        model.eval()
        
        # Copy weights from reference model
        try:
            model.load_state_dict(ref_model.state_dict(), strict=False)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not copy all weights: {e}")
        
        return model
    
    def test_get_outputs(self, impl_class, test_case, ref_model):
        """Test get_outputs function."""
        ray_samples = test_case['ray_samples']
        density_embedding = test_case['density_embedding']
        
        try:
            # Create model instance
            model = self.create_field_model(impl_class, ref_model)
            model.eval()
            
            start_time = time.time()
            with torch.no_grad():
                output = model.get_outputs(ray_samples, density_embedding=density_embedding)
            exec_time = time.time() - start_time
            
            with torch.no_grad():
                reference = ref_model.get_outputs(ray_samples, density_embedding=density_embedding)
            
            metrics = self.compute_error(output, reference)
            metrics['execution_time'] = exec_time
        except Exception as e:
            metrics = {
                'error': str(e),
                'pass': False,
                'execution_time': 0
            }
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        return metrics
    
    def test_single_implementation(self, impl_path, ref_model):
        """Test a single LLM implementation."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # Load implementation
        impl_class = self.load_llm_implementation(impl_path)
        
        if impl_class is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'overall_pass_rate': 0.0,
                'total_pass_count': 0,
                'total_test_count': 0
            }
        
        # Generate test cases
        test_cases = self.test_generator.generate_test_suite(self.num_tests, field_model=ref_model)
        
        all_results = []
        
        # Run all test cases
        for i, test_case in enumerate(test_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(test_cases)}: {test_case['description']}")
            
            result = self.test_get_outputs(impl_class, test_case, ref_model)
            
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
        
        for test_result in all_results:
            result = test_result['result']
            if result.get('pass', False):
                passes.append(True)
                l1_errors.append(result.get('l1_error', 0))
                l2_errors.append(result.get('l2_error', 0))
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
            print(f"  Avg max error: {summary['avg_max']:.2e}")
            print(f"  Avg time: {summary['avg_time']:.4f}s")
        
        pass_count = summary.get('total_pass_count', 0)
        test_count = summary.get('total_test_count', 0)
        print(f"  Overall: {summary.get('overall_pass_rate', 0.0):.1f}% ({pass_count}/{test_count} tests passed)")
        print(f"{'='*80}")
    
    def batch_test(self, implementations_dir, ref_model):
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
            summary = self.test_single_implementation(str(impl_file), ref_model)
            all_summaries.append(summary)
        
        # Print comparison
        self.print_comparison(all_summaries)
        
        # Save results to file
        self.save_results_to_file(all_summaries)
        self.save_summary_to_file(all_summaries)
        
        return all_summaries

    def save_summary_to_file(self, all_results, output_path: Optional[str] = None):
        """Write structured `test_summary.json` aligned with `schema.json`."""
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
        for impl_summary in all_results:
            implementations.append(
                {
                    "name": impl_summary.get("implementation"),
                    "test_total": int(impl_summary.get("total_test_count", 0)),
                    "test_pass": int(impl_summary.get("total_pass_count", 0)),
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
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"✓ Summary saved to: {output_path}")

        return str(output_path)
    
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
            avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
            
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
                    
                    # Write overall statistics
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    overall_rate = summary.get('overall_pass_rate', 0.0)
                    f.write(f"Overall Average Pass Rate: {overall_rate:.1f}% ({pass_count}/{test_count} tests passed)\n")
                    
                    if 'avg_l1' in summary:
                        f.write(f"Avg L1 error: {summary['avg_l1']:.2e}\n")
                        f.write(f"Avg L2 error: {summary['avg_l2']:.2e}\n")
                        f.write(f"Avg max error: {summary['avg_max']:.2e}\n")
                        f.write(f"Avg time: {summary['avg_time']:.4f}s\n")
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<20} {'Pass Rate':<12} {'Avg L1':<12} {'Avg L2':<12} {'Avg Max':<12} {'Avg Time':<12}\n")
                f.write("-" * 100 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'results' not in summary:
                        f.write(f"{name:<30} {'ERROR':<20} {'0.0%':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        f.write("-" * 100 + "\n")
                        continue
                    
                    overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
                    pass_count = summary.get('total_pass_count', 0)
                    test_count = summary.get('total_test_count', 0)
                    count_info = f"({pass_count}/{test_count})"
                    
                    avg_l1 = f"{summary.get('avg_l1', 0):.2e}" if 'avg_l1' in summary else "N/A"
                    avg_l2 = f"{summary.get('avg_l2', 0):.2e}" if 'avg_l2' in summary else "N/A"
                    avg_max = f"{summary.get('avg_max', 0):.2e}" if 'avg_max' in summary else "N/A"
                    avg_time = f"{summary.get('avg_time', 0):.4f}s" if 'avg_time' in summary else "N/A"
                    
                    f.write(f"{name:<30} {count_info:<20} {overall_rate:<12} {avg_l1:<12} {avg_l2:<12} {avg_max:<12} {avg_time:<12}\n")
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


def create_reference_model(device='cpu'):
    """Create a reference model for testing."""
    # Create a simple aabb
    aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device)
    
    # Create reference model
    model = ReferenceUnmixField(
        aabb=aabb,
        num_images=1,
        implementation='torch',  # Use torch to avoid tcnn dependency
        num_layers_color=3,
        hidden_dim_color=64,
        wavelengths=128,
        method='spectral',
        num_classes=4,
        feature_dim=256,
        temperature=0.5,
        pred_specular=False,
    )
    model.eval()
    
    # Ensure endmembers are on CPU
    if hasattr(model, 'endmembers'):
        if isinstance(model.endmembers, nn.Parameter):
            model.endmembers = nn.Parameter(model.endmembers.data.to(device))
        else:
            model.endmembers = model.endmembers.to(device)
    
    return model


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner for unmixField.get_outputs()')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Error tolerance for pass/fail (default: 1e-4)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    impl_dir = script_dir / args.impl_dir
    
    # Create reference model
    print("Creating reference model...")
    ref_model = create_reference_model(device=args.device)
    
    # Create test runner
    runner = TestRunner(
        num_tests=args.num_tests,
        verbose=not args.quiet,
        tolerance=args.tolerance,
        device=args.device
    )
    
    # Run tests
    results = runner.batch_test(str(impl_dir), ref_model)
    
    return results


if __name__ == '__main__':
    main()
