"""
Test Runner for SinkhornDistance.compute_cost_matrix
Batch tests multiple LLM files and compares to reference numerically (CPU-only).
"""

import os
import sys
import time
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
import json

import torch
import inspect

# Ensure local imports (reference_implementation)
sys.path.insert(0, os.path.dirname(__file__))

try:
    from reference_implementation import SinkhornDistance as RefSinkhorn
    from test_generator import TestDataGenerator
except Exception as e:
    print(f"Error: failed to import reference_implementation or test_generator: {e}")
    sys.exit(1)


class TestRunner:
    def __init__(self, num_tests: int = 5, tolerance: float = 1e-5, verbose: bool = True):
        self.num_tests = num_tests
        self.tolerance = tolerance
        self.verbose = verbose
        self.generator = TestDataGenerator()
        self.cases = self.generator.generate(num_tests)
        self.ref_model = RefSinkhorn()

    def load_impl(self, filepath: str):
        """Load an implementation module, returning (module, class_or_None, func_or_None)."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            impl_cls = getattr(module, 'SinkhornDistance', None)
            impl_fn = getattr(module, 'compute_cost_matrix', None)
            if impl_cls is None and impl_fn is None:
                raise AttributeError(f"Neither SinkhornDistance nor compute_cost_matrix found in {filepath}")
            return module, impl_cls, impl_fn
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None, None

    def compute_metrics(self, output: torch.Tensor, reference: torch.Tensor):
        metrics = {}
        if not isinstance(output, torch.Tensor):
            metrics['error'] = f"Output is not a torch.Tensor (got {type(output)})"
            metrics['pass'] = False
            return metrics
        if output.shape != reference.shape:
            metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
            metrics['pass'] = False
            return metrics
        diff = output - reference
        l2 = torch.sqrt(torch.mean(diff**2)).item()
        mse = torch.mean(diff**2).item()
        l1 = torch.mean(torch.abs(diff)).item()
        max_err = torch.max(torch.abs(diff)).item()
        ref_norm = torch.norm(reference)
        rel = (torch.norm(diff) / ref_norm).item() if ref_norm > 1e-12 else (0.0 if max_err < 1e-12 else 1.0)
        metrics.update({
            'l1': l1,
            'l2': l2,
            'mse': mse,
            'max': max_err,
            'rel': rel,
            'pass': max_err <  self.tolerance,
        })
        return metrics

    def run_one(self, impl_cls, impl_fn, case):
        # Try class-based API first, then fallback to standalone function
        try:
            if impl_cls is not None:
                model = impl_cls()
                start = time.time()
                out = model.compute_cost_matrix(case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
                dt = time.time() - start
            elif impl_fn is not None:
                start = time.time()
                try:
                    sig = inspect.signature(impl_fn)
                    params = list(sig.parameters.values())
                except Exception:
                    params = []
                if params and params[0].name == 'self':
                    class _Dummy:
                        def __init__(self, helper):
                            self.matrix_sqrt_eigh = helper
                    helper = getattr(self.ref_model, 'matrix_sqrt_eigh')
                    dummy = _Dummy(helper)
                    out = impl_fn(dummy, case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
                else:
                    out = impl_fn(case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
                dt = time.time() - start
            else:
                return {'error': 'No valid implementation found', 'pass': False, 'time': 0.0}

            ref = self.ref_model.compute_cost_matrix(case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
            metrics = self.compute_metrics(out, ref)
            metrics['time'] = dt
            return metrics
        except NameError as e:
            # Fallback: if NameError (e.g., 'self' not defined), try standalone function if available
            try:
                # If class path failed with 'self' NameError and we have a function, try function with dummy self if needed
                if impl_fn is not None:
                    start = time.time()
                    # Same 'self' handling for fallback
                    try:
                        sig = inspect.signature(impl_fn)
                        params = list(sig.parameters.values())
                    except Exception:
                        params = []
                    if params and params[0].name == 'self':
                        class _Dummy:
                            def __init__(self, helper):
                                self.matrix_sqrt_eigh = helper
                        helper = getattr(self.ref_model, 'matrix_sqrt_eigh')
                        dummy = _Dummy(helper)
                        out = impl_fn(dummy, case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
                    else:
                        out = impl_fn(case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
                    dt = time.time() - start
                    ref = self.ref_model.compute_cost_matrix(case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
                    metrics = self.compute_metrics(out, ref)
                    metrics['time'] = dt
                    metrics['note'] = 'fallback:function'
                    return metrics
                # If function path is not available but class exists, try re-instantiating class and call method
                if impl_cls is not None:
                    model = impl_cls()
                    start = time.time()
                    out = model.compute_cost_matrix(case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
                    dt = time.time() - start
                    ref = self.ref_model.compute_cost_matrix(case['mu_A'], case['cov_A'], case['mu_B'], case['cov_B'])
                    metrics = self.compute_metrics(out, ref)
                    metrics['time'] = dt
                    metrics['note'] = 'retry:class'
                    return metrics
            except Exception as e2:
                return {'error': f"fallback failed: {e2}", 'pass': False, 'time': 0.0}
            return {'error': str(e), 'pass': False, 'time': 0.0}
        except Exception as e:
            return {'error': str(e), 'pass': False, 'time': 0.0}

    def test_file(self, filepath: str):
        name = Path(filepath).stem
        if self.verbose:
            print(f"\n{'='*80}\nTesting: {name}\n{'='*80}")
        module, impl_cls, impl_fn = self.load_impl(filepath)
        if module is None and impl_cls is None and impl_fn is None:
            return {'implementation': name, 'error': 'load_failed', 'overall_pass_rate': 0.0}
        results = []
        pass_count = 0
        for i, case in enumerate(self.cases):
            if self.verbose:
                print(f"Test {i+1}/{len(self.cases)}: {case['desc']}")
            m = self.run_one(impl_cls, impl_fn, case)
            results.append(m)
            ok = m.get('pass', False)
            pass_count += 1 if ok else 0
            if self.verbose:
                if ok:
                    print(f"  ✓ Pass (MSE={m.get('mse',0):.2e}, L2={m.get('l2',0):.2e}, Max={m.get('max',0):.2e}, t={m.get('time',0):.4f}s)")
                else:
                    print(f"  ✗ Fail - {m.get('error','error exceeds tolerance')}")
        total = len(results)
        avg = lambda k: sum(r.get(k, 0.0) for r in results if k in r) / max(1, sum(1 for r in results if k in r))
        summary = {
            'implementation': name,
            'total_tests': total,
            'total_pass_count': pass_count,
            'total_test_count': total,
            'pass_rate': (pass_count/total*100.0) if total else 0.0,
            'avg_mse': avg('mse'),
            'avg_l2': avg('l2'),
            'avg_l1': avg('l1'),
            'avg_max': avg('max'),
            'avg_rel': avg('rel'),
            'avg_time': avg('time'),
            'results': results,
            'overall_pass_rate': (pass_count/total*100.0) if total else 0.0,
        }
        if self.verbose:
            print(f"\nSummary for {name}: pass {pass_count}/{total} ({summary['pass_rate']:.1f}%), avg MSE={summary['avg_mse']:.2e}")
        return summary

    def batch_dir(self, implementations_dir: str):
        impl_dir = Path(implementations_dir)
        if not impl_dir.exists():
            print(f"Error: {implementations_dir} not found")
            return []
        files = [p for p in impl_dir.glob('*.py') if p.stem not in ['__init__', 'llm_template']]
        if self.verbose:
            print(f"Found {len(files)} implementations; running {self.num_tests} tests each\n")
        summaries = [self.test_file(str(p)) for p in files]
        self.print_comparison(summaries)
        
        # Save results to file
        self.save_results_to_file(summaries)
        # Save structured test summary for aggregation
        self.save_summary_to_file(summaries)
        
        return summaries

    @staticmethod
    def print_comparison(summaries):
        if not summaries:
            return
        print("\n" + "="*100)
        print("COMPARISON SUMMARY")
        print("="*100)
        print(f"{'Implementation':<22} {'Pass%':<8} {'Avg MSE':<12} {'Avg L2':<12} {'Avg Max':<12} {'Avg Time':<10}")
        print("-"*100)
        for s in summaries:
            name = s.get('implementation','')[:22]
            pr = f"{s.get('pass_rate',0.0):.1f}%"
            print(f"{name:<22} {pr:<8} {s.get('avg_mse',0.0):.2e} {s.get('avg_l2',0.0):.2e} {s.get('avg_max',0.0):.2e} {s.get('avg_time',0.0):.4f}s")
        print("-"*100)
        
        # Print ranking
        print(f"\n{'OVERALL RANKING':<25} {'Pass Rate':<15} {'Pass Count':<15}")
        print("-" * 57)
        sorted_summaries = sorted(summaries, key=lambda x: x.get('overall_pass_rate', 0.0), reverse=True)
        for i, summary in enumerate(sorted_summaries, 1):
            name = summary.get('implementation', '')[:23]
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
    import argparse
    parser = argparse.ArgumentParser(description='Test runner for compute_cost_matrix')
    parser.add_argument('--num-tests', type=int, default=10)
    parser.add_argument('--impl-dir', type=str, default='llm_implementations')
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    runner = TestRunner(num_tests=args.num_tests, tolerance=args.tolerance, verbose=not args.quiet)
    script_dir = Path(__file__).parent
    impl_dir = script_dir / args.impl_dir
    runner.batch_dir(str(impl_dir))


if __name__ == '__main__':
    main()


