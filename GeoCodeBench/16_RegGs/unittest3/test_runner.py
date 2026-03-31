"""
Test Runner for compute_mw2_loss method (class TemplateAligner)
Mirrors unittest2 style: batch testing, numeric comparisons, CPU-only.
"""

from __future__ import annotations

import json
import os
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import torch

# Add current directory for imports
sys.path.insert(0, os.path.dirname(__file__))

# Reference implementation and test generator
try:
    from reference_implementation import TemplateAligner as RefAligner
    from test_generator import TestDataGenerator
except Exception as e:
    print(f"Error: reference_implementation.py or test_generator.py not found or failed to import: {e}")
    sys.exit(1)


class TestRunner:
    def __init__(self, num_tests: int = 5, verbose: bool = True, tolerance: float = 1e-5):
        self.num_tests = num_tests
        self.verbose = verbose
        self.tolerance = tolerance
        self.generator = TestDataGenerator()
        self.test_cases = self.generator.generate(num_tests)
        self.ref_impl = RefAligner()

    def load_llm(self, filepath: str):
        try:
            spec = importlib.util.spec_from_file_location('llm_impl', filepath)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            if not hasattr(module, 'TemplateAligner'):
                raise AttributeError(f"No TemplateAligner in {filepath}")
            return module.TemplateAligner()
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def compute_error(self, out_val: float, ref_val: float) -> Dict[str, float | bool]:
        abs_err = abs(out_val - ref_val)
        sq_err = (out_val - ref_val) ** 2
        rel_err = abs_err / (abs(ref_val) + 1e-9)
        return {
            'abs': abs_err,
            'rmse': sq_err ** 0.5,
            'rel': rel_err,
            'pass': abs_err < self.tolerance,
        }

    def test_single_case(self, aligner, case: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ref = self.ref_impl.compute_mw2_loss(
                case['main_component'], case['mini_component'],
                case['w2c'], case['scale'], case['cam_rot'], case['cam_trans']
            )
            ref_v = float(ref.detach().cpu().item())
            got = aligner.compute_mw2_loss(
                case['main_component'], case['mini_component'],
                case['w2c'], case['scale'], case['cam_rot'], case['cam_trans']
            )
            if isinstance(got, torch.Tensor):
                got_v = float(got.detach().cpu().item())
            else:
                got_v = float(got)
            metrics = self.compute_error(got_v, ref_v)
            return metrics
        except Exception as e:
            return {'error': str(e), 'pass': False}

    def test_impl(self, impl_path: str) -> Dict[str, Any]:
        name = Path(impl_path).stem
        if self.verbose:
            print(f"\n{'='*80}\nTesting: {name}\n{'='*80}")
        aligner = self.load_llm(impl_path)
        if aligner is None:
            return {'implementation': name, 'overall_pass_rate': 0.0, 'total_pass_count': 0, 'total_test_count': 0}

        results: List[Dict[str, Any]] = []
        for i, case in enumerate(self.test_cases):
            if self.verbose:
                print(f"Test {i+1}/{len(self.test_cases)}: {case['description']}")
            res = self.test_single_case(aligner, case)
            results.append({'test_idx': i, 'description': case['description'], 'result': res})
            if self.verbose:
                if res.get('pass', False):
                    print(f"  ✓ Pass (abs={res.get('abs', 0):.2e}, rel={res.get('rel', 0):.2e})")
                else:
                    print(f"  ✗ Fail - {res.get('error', 'exceeds tolerance')} ")

        return self._summary(name, results)

    def _summary(self, name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        passes = [r['result'].get('pass', False) for r in results]
        abs_list = [r['result'].get('abs', 0.0) for r in results if r['result'].get('pass', False)]
        rel_list = [r['result'].get('rel', 0.0) for r in results if r['result'].get('pass', False)]
        rmse_list = [r['result'].get('rmse', 0.0) for r in results if r['result'].get('pass', False)]
        summary: Dict[str, Any] = {
            'implementation': name,
            'total_tests': len(results),
            'results': results,
            'pass_rate': (sum(passes) / len(passes) * 100.0) if results else 0.0,
            'total_pass_count': sum(passes),
            'total_test_count': len(passes),
        }
        if abs_list:
            summary.update({
                'avg_abs': sum(abs_list) / len(abs_list),
                'avg_rel': sum(rel_list) / len(rel_list),
                'avg_rmse': sum(rmse_list) / len(rmse_list),
            })
        summary['overall_pass_rate'] = summary.get('pass_rate', 0.0)
        return summary

    @staticmethod
    def print_comparison(summaries: List[Dict[str, Any]]) -> None:
        if not summaries:
            return
        print(f"\n{'='*100}\nCOMPARISON SUMMARY\n{'='*100}\n")
        print(f"{'Implementation':<25} {'Pass Rate':<12} {'Avg abs':<12} {'Avg rmse':<12} {'Avg rel':<12}")
        print('-' * 100)
        for s in summaries:
            name = s['implementation'][:23]
            pr = f"{s.get('overall_pass_rate', 0.0):.1f}%"
            avga = f"{s.get('avg_abs', 0):.2e}" if 'avg_abs' in s else 'N/A'
            avgr = f"{s.get('avg_rmse', 0):.2e}" if 'avg_rmse' in s else 'N/A'
            avgre = f"{s.get('avg_rel', 0):.2e}" if 'avg_rel' in s else 'N/A'
            print(f"{name:<25} {pr:<12} {avga:<12} {avgr:<12} {avgre:<12}")
        print('-' * 100)
        
        # Print ranking
        print(f"\n{'OVERALL RANKING':<25} {'Pass Rate':<15} {'Pass Count':<15}")
        print("-" * 57)
        sorted_summaries = sorted(summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
        for i, summary in enumerate(sorted_summaries, 1):
            name = summary['implementation'][:23]
            overall_rate = f"{summary.get('overall_pass_rate', 0.0):.1f}%"
            pass_count = summary.get('total_pass_count', 0)
            test_count = summary.get('total_test_count', 0)
            count_str = f"{pass_count}/{test_count}"
            print(f"{i}. {name:<30} {overall_rate:<15} {count_str:<15}")
        print("-" * 57)

    def batch_test(self, implementations_dir: str) -> List[Dict[str, Any]]:
        impl_dir = Path(implementations_dir)
        if not impl_dir.exists():
            print(f"Error: Directory {implementations_dir} does not exist")
            return []
        files = [f for f in impl_dir.glob('*.py') if f.stem not in ['__init__', 'llm_template']]
        if not files:
            print(f"No implementation files found in {implementations_dir}")
            return []
        print(f"\nFound {len(files)} implementations to test")
        print(f"Running {self.num_tests} test cases per implementation\n")
        summaries: List[Dict[str, Any]] = []
        for f in files:
            summaries.append(self.test_impl(str(f)))
        self.print_comparison(summaries)
        
        # Save results to file
        self.save_results_to_file(summaries)
        # Save structured test summary for aggregation
        self.save_summary_to_file(summaries)
        
        return summaries

    


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
    import argparse
    parser = argparse.ArgumentParser(description='Test runner for compute_mw2_loss (CPU-only)')
    parser.add_argument('--num-tests', type=int, default=5, help='Number of test cases (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations', help='Directory of LLM implementations')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Error tolerance (default: 1e-5)')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    runner = TestRunner(num_tests=args.num_tests, verbose=not args.quiet, tolerance=args.tolerance)
    _ = runner.batch_test(str(Path(__file__).parent / args.impl_dir))


if __name__ == '__main__':
    main()


