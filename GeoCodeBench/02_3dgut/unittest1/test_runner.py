"""
Comprehensive Unit Test Framework for _isect_tiles() and _isect_offset_encode()

This framework supports:
1. Batch testing multiple LLM implementations
2. Pure numerical computation and comparison
3. Minimal dependencies (only torch/numpy)
4. Configurable number of test cases

Usage:
    python test_isect_functions.py --num-tests 5 --impl-dir llm_implementations
"""

import torch
import numpy as np
import struct
import math
import os
import sys
import importlib.util
import time
import signal
import subprocess
import json
import pickle
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Optional


# Import split modules to avoid duplication in this test runner.
from reference_implementation import reference_isect_offset_encode, reference_isect_tiles
from test_generator import TestDataGenerator


# ==============================================================================
# TEST RUNNER
# ==============================================================================

class TimeoutException(Exception):
    """Exception raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Test execution timed out")


class TestRunner:
    """Test runner for comparing LLM implementations against reference."""
    
    def __init__(self, num_tests=5, verbose=True, device='cpu', timeout=30):
        self.num_tests = num_tests
        self.verbose = verbose
        self.device = device
        self.timeout = timeout  # Timeout in seconds for each function call
        self.test_generator = TestDataGenerator()
        self.test_cases = self.test_generator.generate_test_suite(num_tests)
    
    def load_llm_implementation(self, filepath):
        """Dynamically load an LLM implementation."""
        try:
            spec = importlib.util.spec_from_file_location("llm_impl", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, '_isect_tiles') or not hasattr(module, '_isect_offset_encode'):
                raise AttributeError(f"Missing required functions in {filepath}")
            
            return module._isect_tiles, module._isect_offset_encode
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def compute_metrics_isect_tiles(self, output, reference):
        """Compute comparison metrics for _isect_tiles output."""
        tiles_per_gauss_out, isect_ids_out, flatten_ids_out = output
        tiles_per_gauss_ref, isect_ids_ref, flatten_ids_ref = reference
        
        metrics = {}
        
        # Check shapes
        metrics['tiles_per_gauss_shape_match'] = tiles_per_gauss_out.shape == tiles_per_gauss_ref.shape
        metrics['isect_ids_shape_match'] = isect_ids_out.shape == isect_ids_ref.shape
        metrics['flatten_ids_shape_match'] = flatten_ids_out.shape == flatten_ids_ref.shape
        
        if not all([metrics['tiles_per_gauss_shape_match'], 
                   metrics['isect_ids_shape_match'], 
                   metrics['flatten_ids_shape_match']]):
            metrics['error'] = f"Shape mismatch detected"
            return metrics
        
        # Exact match
        metrics['tiles_per_gauss_exact'] = torch.equal(tiles_per_gauss_out, tiles_per_gauss_ref)
        metrics['isect_ids_exact'] = torch.equal(isect_ids_out, isect_ids_ref)
        metrics['flatten_ids_exact'] = torch.equal(flatten_ids_out, flatten_ids_ref)
        
        # Numerical differences
        metrics['tiles_per_gauss_mae'] = torch.mean(
            torch.abs(tiles_per_gauss_out.float() - tiles_per_gauss_ref.float())
        ).item()
        
        if len(isect_ids_out) > 0:
            # For sorted IDs, compare bit patterns
            metrics['isect_ids_match_rate'] = (isect_ids_out == isect_ids_ref).float().mean().item()
            metrics['flatten_ids_match_rate'] = (flatten_ids_out == flatten_ids_ref).float().mean().item()
        else:
            metrics['isect_ids_match_rate'] = 1.0
            metrics['flatten_ids_match_rate'] = 1.0
        
        # Overall success
        metrics['success'] = (
            metrics['tiles_per_gauss_exact'] and 
            metrics['isect_ids_exact'] and 
            metrics['flatten_ids_exact']
        )
        
        return metrics
    
    def compute_metrics_isect_offset(self, output, reference):
        """Compute comparison metrics for _isect_offset_encode output."""
        metrics = {}
        
        # Check shape
        metrics['shape_match'] = output.shape == reference.shape
        
        if not metrics['shape_match']:
            metrics['error'] = f"Shape mismatch: {output.shape} vs {reference.shape}"
            return metrics
        
        # Exact match
        metrics['exact_match'] = torch.equal(output, reference)
        
        # Numerical differences
        metrics['mae'] = torch.mean(torch.abs(output.float() - reference.float())).item()
        metrics['rmse'] = torch.sqrt(torch.mean((output.float() - reference.float()) ** 2)).item()
        metrics['max_error'] = torch.max(torch.abs(output.float() - reference.float())).item()
        
        # Element-wise match rate
        metrics['match_rate'] = (output == reference).float().mean().item()
        
        metrics['success'] = metrics['exact_match']
        
        return metrics
    
    def run_single_test(self, isect_tiles_func, isect_offset_func, test_case, test_idx):
        """Run a single test case with comprehensive error handling."""
        results = {}
        
        # Prepare inputs
        try:
            means2d = test_case['means2d'].to(self.device)
            radii = test_case['radii'].to(self.device)
            depths = test_case['depths'].to(self.device)
            tile_size = test_case['tile_size']
            tile_width = test_case['tile_width']
            tile_height = test_case['tile_height']
        except Exception as e:
            results['error'] = f"Failed to prepare inputs: {str(e)}"
            results['overall_success'] = False
            return results
        
        # Test _isect_tiles with exception handling
        tiles_success = False
        try:
            # Set up timeout if on Unix-like system
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            
            start_time = time.time()
            tiles_out, isect_ids_out, flatten_ids_out = isect_tiles_func(
                means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
            )
            time_isect_tiles = time.time() - start_time
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            # Reference for _isect_tiles
            try:
                tiles_ref, isect_ids_ref, flatten_ids_ref = reference_isect_tiles(
                    means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
                )
                
                # Metrics for _isect_tiles
                metrics_tiles = self.compute_metrics_isect_tiles(
                    (tiles_out, isect_ids_out, flatten_ids_out),
                    (tiles_ref, isect_ids_ref, flatten_ids_ref)
                )
                metrics_tiles['execution_time'] = time_isect_tiles
                results['isect_tiles'] = metrics_tiles
                tiles_success = metrics_tiles.get('success', False)
                
            except Exception as e:
                results['isect_tiles'] = {
                    'success': False,
                    'error': f"Comparison failed: {str(e)}",
                    'execution_time': time_isect_tiles
                }
                
        except TimeoutException:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            results['isect_tiles'] = {
                'success': False,
                'error': f'Timeout (>{self.timeout}s)',
                'execution_time': self.timeout
            }
        except MemoryError:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            results['isect_tiles'] = {
                'success': False,
                'error': 'Memory Error (OOM)',
                'execution_time': 0
            }
        except KeyboardInterrupt:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            raise  # Allow user to interrupt
        except Exception as e:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            error_type = type(e).__name__
            results['isect_tiles'] = {
                'success': False,
                'error': f'{error_type}: {str(e)}',
                'execution_time': 0
            }
        
        # Test _isect_offset_encode with exception handling
        offset_success = False
        try:
            I = math.prod(means2d.shape[:-2])
            
            # Use reference isect_ids to test offset encoding
            if tiles_success and 'isect_tiles' in results:
                # Use the implementation's output if tiles test succeeded
                test_isect_ids = isect_ids_ref
            else:
                # Generate reference for testing offset function independently
                _, isect_ids_ref, _ = reference_isect_tiles(
                    means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
                )
                test_isect_ids = isect_ids_ref
            
            # Set up timeout if on Unix-like system
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            
            start_time = time.time()
            offsets_out = isect_offset_func(test_isect_ids, I, tile_width, tile_height)
            time_offset = time.time() - start_time
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            # Reference for _isect_offset_encode
            try:
                offsets_ref = reference_isect_offset_encode(test_isect_ids, I, tile_width, tile_height)
                
                # Metrics for _isect_offset_encode
                metrics_offset = self.compute_metrics_isect_offset(offsets_out, offsets_ref)
                metrics_offset['execution_time'] = time_offset
                results['isect_offset'] = metrics_offset
                offset_success = metrics_offset.get('success', False)
                
            except Exception as e:
                results['isect_offset'] = {
                    'success': False,
                    'error': f"Comparison failed: {str(e)}",
                    'execution_time': time_offset
                }
                
        except TimeoutException:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            results['isect_offset'] = {
                'success': False,
                'error': f'Timeout (>{self.timeout}s)',
                'execution_time': self.timeout
            }
        except MemoryError:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            results['isect_offset'] = {
                'success': False,
                'error': 'Memory Error (OOM)',
                'execution_time': 0
            }
        except KeyboardInterrupt:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            raise  # Allow user to interrupt
        except Exception as e:
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            error_type = type(e).__name__
            results['isect_offset'] = {
                'success': False,
                'error': f'{error_type}: {str(e)}',
                'execution_time': 0
            }
        
        # Overall success
        results['overall_success'] = tiles_success and offset_success
        
        # Set overall error message if both failed
        if not results['overall_success'] and 'error' not in results:
            errors = []
            if 'isect_tiles' in results and not results['isect_tiles'].get('success', False):
                errors.append(f"_isect_tiles: {results['isect_tiles'].get('error', 'Failed')}")
            if 'isect_offset' in results and not results['isect_offset'].get('success', False):
                errors.append(f"_isect_offset: {results['isect_offset'].get('error', 'Failed')}")
            if errors:
                results['error'] = '; '.join(errors)
        
        return results
    
    def test_implementation(self, impl_path):
        """Test a single LLM implementation."""
        impl_name = Path(impl_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {impl_name}")
            print(f"{'='*80}")
        
        # Load implementation with error handling
        try:
            isect_tiles_func, isect_offset_func = self.load_llm_implementation(impl_path)
        except Exception as e:
            error_type = type(e).__name__
            if self.verbose:
                print(f"Error loading implementation: {error_type}: {str(e)}")
            return {
                'implementation': impl_name,
                'error': f'Failed to load implementation: {error_type}: {str(e)}',
                'total_tests': self.num_tests,
                'successful_tests': 0,
                'failed_tests': self.num_tests,
                'test_results': []
            }
        
        if isect_tiles_func is None or isect_offset_func is None:
            return {
                'implementation': impl_name,
                'error': 'Failed to load implementation',
                'total_tests': self.num_tests,
                'successful_tests': 0,
                'failed_tests': self.num_tests,
                'test_results': []
            }
        
        # Run all test cases
        test_results = []
        for i, test_case in enumerate(self.test_cases):
            if self.verbose:
                print(f"\nTest {i+1}/{len(self.test_cases)}: {test_case['description']}")
            
            try:
                result = self.run_single_test(isect_tiles_func, isect_offset_func, test_case, i)
                result['test_description'] = test_case['description']
                test_results.append(result)
                
                if self.verbose:
                    if result.get('overall_success', False):
                        print(f"  ✓ Success")
                        tiles_m = result.get('isect_tiles', {})
                        offset_m = result.get('isect_offset', {})
                        print(f"    _isect_tiles:")
                        print(f"      Time: {tiles_m.get('execution_time', 0):.4f}s")
                        print(f"      Tiles exact: {tiles_m.get('tiles_per_gauss_exact', False)}")
                        print(f"      IDs match: {tiles_m.get('isect_ids_match_rate', 0)*100:.2f}%")
                        print(f"    _isect_offset_encode:")
                        print(f"      Time: {offset_m.get('execution_time', 0):.4f}s")
                        print(f"      Exact match: {offset_m.get('exact_match', False)}")
                        print(f"      MAE: {offset_m.get('mae', 0):.6f}")
                    else:
                        print(f"  ✗ Failed")
                        # Show detailed error information
                        tiles_m = result.get('isect_tiles', {})
                        offset_m = result.get('isect_offset', {})
                        
                        if tiles_m and not tiles_m.get('success', True):
                            print(f"    _isect_tiles: ✗ {tiles_m.get('error', 'Failed')}")
                            if tiles_m.get('execution_time', 0) > 0:
                                print(f"      Time: {tiles_m['execution_time']:.4f}s")
                        elif tiles_m:
                            print(f"    _isect_tiles: ✓")
                        
                        if offset_m and not offset_m.get('success', True):
                            print(f"    _isect_offset_encode: ✗ {offset_m.get('error', 'Failed')}")
                            if offset_m.get('execution_time', 0) > 0:
                                print(f"      Time: {offset_m['execution_time']:.4f}s")
                        elif offset_m:
                            print(f"    _isect_offset_encode: ✓")
                        
                        # Show overall error if present
                        if 'error' in result and result['error']:
                            print(f"    Overall: {result['error']}")
            except MemoryError as e:
                # Handle memory errors gracefully
                error_result = {
                    'test_description': test_case['description'],
                    'overall_success': False,
                    'isect_tiles': {
                        'success': False,
                        'error': f'Memory Error (OOM): {str(e)}',
                        'execution_time': 0
                    },
                    'isect_offset': {
                        'success': False,
                        'error': f'Memory Error (OOM): {str(e)}',
                        'execution_time': 0
                    },
                    'error': f'Memory Error (OOM): {str(e)}'
                }
                test_results.append(error_result)
                if self.verbose:
                    print(f"  ✗ Failed")
                    print(f"    Memory Error (OOM): {str(e)}")
                    print(f"    Skipping remaining tests for this case due to memory constraints")
            except Exception as e:
                # Catch any other unexpected errors
                error_type = type(e).__name__
                error_result = {
                    'test_description': test_case['description'],
                    'overall_success': False,
                    'isect_tiles': {
                        'success': False,
                        'error': f'{error_type}: {str(e)}',
                        'execution_time': 0
                    },
                    'isect_offset': {
                        'success': False,
                        'error': f'{error_type}: {str(e)}',
                        'execution_time': 0
                    },
                    'error': f'{error_type}: {str(e)}'
                }
                test_results.append(error_result)
                if self.verbose:
                    print(f"  ✗ Failed")
                    print(f"    Unexpected error: {error_type}: {str(e)}")
                    import traceback
                    print(f"    Traceback: {traceback.format_exc()}")
        
        # Compute summary
        successful_tests = [r for r in test_results if r.get('overall_success', False)]
        
        summary = {
            'implementation': impl_name,
            'total_tests': len(test_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(test_results) - len(successful_tests),
            'test_results': test_results
        }
        
        if successful_tests:
            summary['avg_time_isect_tiles'] = np.mean([
                r['isect_tiles']['execution_time'] for r in successful_tests
            ])
            summary['avg_time_offset'] = np.mean([
                r['isect_offset']['execution_time'] for r in successful_tests
            ])
            summary['avg_offset_mae'] = np.mean([
                r['isect_offset'].get('mae', 0) for r in successful_tests
            ])
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Summary for {impl_name}:")
            print(f"  Tests passed: {summary['successful_tests']}/{summary['total_tests']}")
            if successful_tests:
                print(f"  Avg time (_isect_tiles): {summary['avg_time_isect_tiles']:.4f}s")
                print(f"  Avg time (_isect_offset): {summary['avg_time_offset']:.4f}s")
                print(f"  Avg MAE (offsets): {summary['avg_offset_mae']:.6f}")
            print(f"{'='*80}")
        
        return summary
    
    def test_implementation_in_subprocess(self, impl_file):
        """Test a single implementation in a subprocess to isolate crashes."""
        impl_name = Path(impl_file).stem
        script_dir = Path(__file__).parent
        
        # Create a temporary script to run the test
        test_script = f"""
import sys
import os
sys.path.insert(0, r'{script_dir}')
from test_runner import TestRunner
import pickle
import json
import numpy as np

# Create test runner with same config
runner = TestRunner(
    num_tests={self.num_tests},
    verbose={self.verbose},
    device='{self.device}',
    timeout={self.timeout}
)

# Run test
try:
    result = runner.test_implementation(r'{impl_file}')
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {{k: convert_to_serializable(v) for k, v in obj.items()}}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    result_serializable = convert_to_serializable(result)
    print("SUCCESS_MARKER")
    print(json.dumps(result_serializable))
except Exception as e:
    import traceback
    print("ERROR_MARKER")
    print(json.dumps({{
        'implementation': '{impl_name}',
        'error': f'{{type(e).__name__}}: {{str(e)}}',
        'traceback': traceback.format_exc(),
        'total_tests': {self.num_tests},
        'successful_tests': 0,
        'failed_tests': {self.num_tests},
        'test_results': []
    }}))
    sys.exit(1)
"""
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            temp_script = f.name
        
        try:
            # Run in subprocess with timeout
            timeout_seconds = self.timeout * self.num_tests * 2 + 60  # Allow extra time
            
            # If verbose, show output in real-time, otherwise capture it
            if self.verbose:
                # Show output in real-time but still capture it
                process = subprocess.Popen(
                    [sys.executable, temp_script],
                    cwd=str(script_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                stdout_lines = []
                try:
                    for line in process.stdout:
                        print(line, end='')
                        stdout_lines.append(line)
                    process.wait(timeout=timeout_seconds)
                    returncode = process.returncode
                    stdout = ''.join(stdout_lines)
                    stderr = ''
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    return {
                        'implementation': impl_name,
                        'error': f'Subprocess timeout after {timeout_seconds}s',
                        'total_tests': self.num_tests,
                        'successful_tests': 0,
                        'failed_tests': self.num_tests,
                        'test_results': []
                    }
            else:
                # Capture output silently
                result = subprocess.run(
                    [sys.executable, temp_script],
                    cwd=str(script_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )
                returncode = result.returncode
                stdout = result.stdout
                stderr = result.stderr
            
            # Parse output
            output_lines = stdout.split('\n')
            error_lines = stderr.split('\n') if stderr else []
            
            # Check if process was killed (SIGKILL = -9)
            if returncode == -9 or (returncode != 0 and returncode != 1 and ('killed' in stderr.lower() or 'killed' in stdout.lower())):
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"ERROR: {impl_name} process was killed (likely OOM)")
                    print(f"{'='*80}")
                return {
                    'implementation': impl_name,
                    'error': 'Process killed (likely OOM or system error)',
                    'total_tests': self.num_tests,
                    'successful_tests': 0,
                    'failed_tests': self.num_tests,
                    'test_results': []
                }
            
            # Try to find success marker
            success_idx = None
            for i, line in enumerate(output_lines):
                if 'SUCCESS_MARKER' in line:
                    success_idx = i + 1
                    break
            
            if success_idx is not None and success_idx < len(output_lines):
                # Parse JSON result
                json_str = output_lines[success_idx].strip()
                try:
                    result_dict = json.loads(json_str)
                    return result_dict
                except json.JSONDecodeError:
                    pass
            
            # Try to find error marker
            error_idx = None
            for i, line in enumerate(output_lines):
                if 'ERROR_MARKER' in line:
                    error_idx = i + 1
                    break
            
            if error_idx is not None and error_idx < len(output_lines):
                json_str = output_lines[error_idx].strip()
                try:
                    result_dict = json.loads(json_str)
                    return result_dict
                except json.JSONDecodeError:
                    pass
            
            # If we get here, something went wrong
            error_msg = '\n'.join(error_lines[-10:]) if error_lines else 'Unknown error'
            return {
                'implementation': impl_name,
                'error': f'Subprocess failed (return code {returncode}): {error_msg}',
                'total_tests': self.num_tests,
                'successful_tests': 0,
                'failed_tests': self.num_tests,
                'test_results': []
            }
            
        except subprocess.TimeoutExpired:
            return {
                'implementation': impl_name,
                'error': f'Subprocess timeout after {timeout_seconds}s',
                'total_tests': self.num_tests,
                'successful_tests': 0,
                'failed_tests': self.num_tests,
                'test_results': []
            }
        except Exception as e:
            return {
                'implementation': impl_name,
                'error': f'Error running subprocess: {type(e).__name__}: {str(e)}',
                'total_tests': self.num_tests,
                'successful_tests': 0,
                'failed_tests': self.num_tests,
                'test_results': []
            }
        finally:
            # Clean up temp script
            try:
                os.unlink(temp_script)
            except:
                pass
    
    def batch_test(self, implementations_dir, use_subprocess=True):
        """Test all implementations in a directory.
        
        Args:
            implementations_dir: Directory containing implementation files
            use_subprocess: If True, run each implementation in a separate subprocess
                          to isolate crashes (default: True)
        """
        impl_dir = Path(implementations_dir)
        
        if not impl_dir.exists():
            print(f"Error: Directory {implementations_dir} does not exist")
            return []
        
        # Find all Python files
        impl_files = list(impl_dir.glob("*.py"))
        impl_files = [f for f in impl_files if f.stem not in ['__init__'] and 'template' not in f.stem.lower()]
        
        if not impl_files:
            print(f"No implementation files found in {implementations_dir}")
            return []
        
        print(f"\nFound {len(impl_files)} implementations to test")
        print(f"Running {self.num_tests} test cases per implementation")
        if use_subprocess:
            print("Using subprocess isolation to prevent crashes from affecting other tests\n")
        else:
            print()
        
        # Test each implementation
        all_results = []
        for impl_file in impl_files:
            if use_subprocess:
                # Use subprocess to isolate crashes
                result = self.test_implementation_in_subprocess(impl_file)
                all_results.append(result)
                
                if self.verbose and result.get('error'):
                    print(f"\n{'='*80}")
                    impl_name = result.get('implementation', Path(impl_file).stem)
                    print(f"ERROR: {impl_name} failed")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    print(f"Continuing with next implementation...")
                    print(f"{'='*80}")
            else:
                # Original in-process testing (for backward compatibility)
                try:
                    result = self.test_implementation(str(impl_file))
                    all_results.append(result)
                except MemoryError as e:
                    # Handle memory errors for entire implementation
                    impl_name = Path(impl_file).stem
                    error_result = {
                        'implementation': impl_name,
                        'error': f'Memory Error (OOM) during testing: {str(e)}',
                        'total_tests': self.num_tests,
                        'successful_tests': 0,
                        'failed_tests': self.num_tests,
                        'test_results': []
                    }
                    all_results.append(error_result)
                    if self.verbose:
                        print(f"\n{'='*80}")
                        print(f"ERROR: {impl_name} failed due to Memory Error (OOM)")
                        print(f"Error: {str(e)}")
                        print(f"Skipping remaining tests for this implementation")
                        print(f"{'='*80}")
                except KeyboardInterrupt:
                    # Allow user to interrupt
                    if self.verbose:
                        print(f"\n\nTest interrupted by user")
                    raise
                except Exception as e:
                    # Catch any other unexpected errors
                    impl_name = Path(impl_file).stem
                    error_type = type(e).__name__
                    error_result = {
                        'implementation': impl_name,
                        'error': f'{error_type} during testing: {str(e)}',
                        'total_tests': self.num_tests,
                        'successful_tests': 0,
                        'failed_tests': self.num_tests,
                        'test_results': []
                    }
                    all_results.append(error_result)
                    if self.verbose:
                        print(f"\n{'='*80}")
                        print(f"ERROR: {impl_name} failed due to unexpected error")
                        print(f"Error type: {error_type}")
                        print(f"Error message: {str(e)}")
                        print(f"Continuing with next implementation...")
                        print(f"{'='*80}")
                        import traceback
                        print(f"Traceback:\n{traceback.format_exc()}")
        
        # Print comparison summary
        self.print_comparison_summary(all_results)
        
        # Save results to file
        self.save_results_to_file(all_results)
        self.save_summary_to_file(all_results)
        
        return all_results
    
    def print_comparison_summary(self, all_results):
        """Print comparison summary of all implementations."""
        if not all_results:
            return
        
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        # Create comparison table
        print(f"{'Implementation':<20} {'Pass Rate':<12} {'Tiles Time':<12} {'Offset Time':<12} {'Offset MAE':<12}")
        print("-" * 68)
        
        for result in all_results:
            name = result['implementation']
            
            # Handle failed to load implementations
            if 'error' in result and 'successful_tests' not in result:
                pass_rate = "LOAD ERROR"
                tiles_time = "N/A"
                offset_time = "N/A"
                mae = "N/A"
            else:
                pass_rate = f"{result['successful_tests']}/{result['total_tests']}"
                
                if result['successful_tests'] > 0:
                    tiles_time = f"{result['avg_time_isect_tiles']:.4f}s"
                    offset_time = f"{result['avg_time_offset']:.4f}s"
                    mae = f"{result['avg_offset_mae']:.6f}"
                else:
                    tiles_time = "N/A"
                    offset_time = "N/A"
                    mae = "N/A"
            
            print(f"{name:<20} {pass_rate:<12} {tiles_time:<12} {offset_time:<12} {mae:<12}")
        
        print("-" * 68)
    
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
                    if 'error' in summary and 'successful_tests' not in summary:
                        f.write(f"ERROR: {summary['error']}\n")
                        f.write(f"Pass Rate: 0/0 (0.0%)\n")
                        f.write("\n")
                        continue
                    
                    # Write test summary
                    f.write(f"Total tests: {summary.get('total_tests', 0)}\n")
                    f.write(f"Successful tests: {summary.get('successful_tests', 0)}\n")
                    f.write(f"Failed tests: {summary.get('failed_tests', 0)}\n\n")
                    
                    # Write detailed test results if available
                    if 'test_results' in summary:
                        f.write("Detailed Test Results:\n")
                        f.write("-" * 80 + "\n")
                        for i, test_result in enumerate(summary['test_results'], 1):
                            test_desc = test_result.get('test_description', f'Test {i}')
                            f.write(f"\nTest {i}: {test_desc}\n")
                            
                            if test_result.get('success', False):
                                f.write("  ✓ Success\n")
                                f.write(f"    Execution time: {test_result.get('execution_time', 0):.4f}s\n")
                                if 'tiles_time' in test_result:
                                    f.write(f"    Tiles time: {test_result['tiles_time']:.4f}s\n")
                                if 'offset_time' in test_result:
                                    f.write(f"    Offset time: {test_result['offset_time']:.4f}s\n")
                                if 'offset_mae' in test_result:
                                    f.write(f"    Offset MAE: {test_result['offset_mae']:.6f}\n")
                            else:
                                f.write(f"  ✗ Failed: {test_result.get('error', 'Unknown error')}\n")
                                if test_result.get('execution_time', 0) > 0:
                                    f.write(f"    Execution time: {test_result['execution_time']:.4f}s\n")
                        
                        f.write("\n" + "-" * 80 + "\n\n")
                    
                    # Write statistics summary if available
                    if summary.get('successful_tests', 0) > 0:
                        f.write("Summary Statistics:\n")
                        if 'avg_execution_time' in summary:
                            f.write(f"  Average execution time: {summary['avg_execution_time']:.4f}s\n")
                        if 'avg_tiles_time' in summary:
                            f.write(f"  Average tiles time: {summary['avg_tiles_time']:.4f}s\n")
                        if 'avg_offset_time' in summary:
                            f.write(f"  Average offset time: {summary['avg_offset_time']:.4f}s\n")
                        if 'avg_offset_mae' in summary:
                            f.write(f"  Average offset MAE: {summary['avg_offset_mae']:.6f}\n")
                    
                    f.write("\n")
                
                # Write comparison table
                f.write("\n" + "="*100 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("="*100 + "\n\n")
                
                # Write table header
                f.write(f"{'Implementation':<20} {'Pass Rate':<12} {'Tiles Time':<12} {'Offset Time':<12} {'Offset MAE':<12}\n")
                f.write("-" * 68 + "\n")
                
                # Write table rows
                for summary in all_summaries:
                    name = summary['implementation']
                    
                    if 'error' in summary and 'successful_tests' not in summary:
                        f.write(f"{name:<20} {'ERROR':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
                        continue
                    
                    pass_rate = f"{summary.get('successful_tests', 0)}/{summary.get('total_tests', 0)}"
                    
                    if summary.get('successful_tests', 0) > 0:
                        tiles_time = f"{summary.get('avg_tiles_time', 0):.4f}s" if 'avg_tiles_time' in summary else "N/A"
                        offset_time = f"{summary.get('avg_offset_time', 0):.4f}s" if 'avg_offset_time' in summary else "N/A"
                        offset_mae = f"{summary.get('avg_offset_mae', 0):.6f}" if 'avg_offset_mae' in summary else "N/A"
                    else:
                        tiles_time = "N/A"
                        offset_time = "N/A"
                        offset_mae = "N/A"
                    
                    f.write(f"{name:<20} {pass_rate:<12} {tiles_time:<12} {offset_time:<12} {offset_mae:<12}\n")
                
                f.write("-" * 68 + "\n")
                
                # Write ranking
                f.write(f"\n{'OVERALL RANKING':<20} {'Avg Pass Rate':<20} {'Pass Count':<15}\n")
                f.write("-" * 57 + "\n")
                sorted_summaries = sorted(all_summaries, key=lambda x: (x.get('successful_tests', 0) / max(x.get('total_tests', 1), 1)) * 100, reverse=True)
                for i, summary in enumerate(sorted_summaries, 1):
                    name = summary['implementation']
                    successful = summary.get('successful_tests', 0)
                    total = summary.get('total_tests', 0)
                    overall_rate = f"{(successful / max(total, 1)) * 100:.1f}%" if total > 0 else "0.0%"
                    count_str = f"{successful}/{total}"
                    f.write(f"{i}. {name:<30} {overall_rate:<20} {count_str:<15}\n")
                f.write("-" * 57 + "\n")
            
            print(f"\n✓ Results saved to: {output_file}")
            return str(output_file)
        
        except Exception as e:
            print(f"\n✗ Error saving results to file: {e}")
            return None

    def save_summary_to_file(self, all_results, output_path=None):
        """Save structured per-implementation pass/total summary to JSON."""
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

        implementations = []
        for r in all_results:
            name = str(r.get("implementation", "unknown"))
            test_total = int(r.get("total_tests", 0) or 0)
            test_pass = int(r.get("successful_tests", 0) or 0)
            implementations.append(
                {
                    "name": name,
                    "test_total": test_total,
                    "test_pass": test_pass,
                }
            )

        summary = {
            "suite": {
                "project_id": project_id,
                "unittest_id": unittest_id,
                "suite_path": suite_path,
                "num_tests_requested": self.num_tests,
            },
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "implementations": implementations,
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Test summary saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"\n✗ Error saving test summary JSON: {e}")
            return None


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test runner for _isect_tiles and _isect_offset_encode implementations'
    )
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test cases to run (default: 5)')
    parser.add_argument('--impl-dir', type=str, default='llm_implementations',
                       help='Directory containing LLM implementations')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run tests on (cpu or cuda)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout in seconds for each function call (default: 30)')
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    impl_dir = script_dir / args.impl_dir
    
    # Create test runner
    runner = TestRunner(
        num_tests=args.num_tests,
        verbose=not args.quiet,
        device=args.device,
        timeout=args.timeout
    )
    
    # Run tests
    results = runner.batch_test(str(impl_dir))
    
    return results


if __name__ == '__main__':
    main()

