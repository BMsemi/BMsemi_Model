"""Board adapter for the installed Qwen six-engine physical self-test.

Accepts platform job.json and writes result.json. Does not emulate hardware or
map unrelated canvas operations to a self-test. Run using the PYNQ interpreter.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/xilinx/qwen6_20261005')


def run(job, output):
    nodes = job.get('nodes', [])
    if len(nodes) != 1 or nodes[0].get('op') != 'core.pe_self_test':
        raise ValueError('This board adapter supports one core.pe_self_test node only. Other operations need their own FPGA adapter.')
    if job.get('edges') or nodes[0].get('params', {}):
        raise ValueError('PE self-test takes no parameters or edges')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = output.with_name('physical_pe_report.json')
    # Remove a previous report so a failed run can never reuse old success.
    if report.exists():
        report.unlink()
    started = time.monotonic()
    proc = subprocess.run([
        sys.executable, str(ROOT / 'fpga/qwen6/board_dma_heap.py'),
        str(ROOT / 'fpga/qwen6/board_smoke.py'),
        '--bitstream', str(ROOT / 'fpga/qwen6/overlay/qwen6.bit'),
        '--driver', str(ROOT / 'fpga/qwen6/axi_x2_mvm_driver.py'),
        '--report', str(report),
    ], cwd=str(ROOT), capture_output=True, text=True, timeout=480)
    if proc.returncode:
        raise RuntimeError((proc.stdout + proc.stderr)[-3000:])
    result = json.loads(report.read_text())
    if result.get('passed') is not True or result.get('physical_hardware') is not True:
        raise RuntimeError('Board did not report a successful physical hardware test')
    nid = nodes[0]['id']
    response = {'backend': 'fpga', 'order': [nid], 'nodes': {
        nid: {'status': 'ok', 'result': result,
              'ms': round((time.monotonic() - started) * 1000, 2)}}}
    output.write_text(json.dumps(response, indent=2))
    return response


if __name__ == '__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])
        print('Physical PE self-test completed successfully.')
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
