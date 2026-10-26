"""Package the built overlay, runner, original DMA driver and one real bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE/'deploy/qwen6_block1.zip')
    args = parser.parse_args()
    driver = ROOT.parent.parent/'TCAM_Search_FPGA/zcu106_tcam_rag_flow/xforge_zcu106/x2_zcu106/runtime/axi_x2_mvm_driver.py'
    sources = {
        'main.py':ROOT/'main.py',
        'src/__init__.py':ROOT/'src/__init__.py',
        'src/ps_int8_quantization.py':ROOT/'src/ps_int8_quantization.py',
        'src/qwen_fpga6.py':ROOT/'src/qwen_fpga6.py',
        'fpga/qwen6/board_smoke.py':HERE/'board_smoke.py',
        'fpga/qwen6/axi_x2_mvm_driver.py':driver,
        'fpga/qwen6/overlay/qwen6.bit':HERE/'overlay/qwen6.bit',
        'fpga/qwen6/overlay/qwen6.hwh':HERE/'overlay/qwen6.hwh',
        'block1_bundle.npz':args.bundle,
    }
    provenance = args.bundle.with_suffix('.source.json')
    if provenance.is_file():
        sources['block1_bundle.source.json'] = provenance
    sources['fpga/qwen6/validation.md'] = HERE/'validation.md'
    sources['docs/fpga_block1_prefill_six_engines.md'] = ROOT/'docs/fpga_block1_prefill_six_engines.md'
    for path in (HERE/'reports').glob('*'):
        if path.is_file():
            sources['fpga/qwen6/reports/'+path.name] = path
    for path in (ROOT/'x1_llm_compiler'/'x1_llm_compiler').rglob('*.py'):
        sources[path.relative_to(ROOT).as_posix()] = path
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f'Finish the build/export first: {path}')
    with np.load(args.bundle,allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
    # Bundling does not assert that this design has been tested on a board.
    manifest = {'board_tested':False, 'bundle_metadata':metadata, 'sha256':{}}
    common = '--fpga-bitstream fpga/qwen6/overlay/qwen6.bit --fpga-driver fpga/qwen6/axi_x2_mvm_driver.py'
    smoke = 'python3 fpga/qwen6/board_smoke.py --bitstream fpga/qwen6/overlay/qwen6.bit --driver fpga/qwen6/axi_x2_mvm_driver.py'
    run = 'python3 main.py --execution-target fpga6 --block-index 0 --prompt '+shlex.quote(metadata['prompt'])+' --fpga-bundle block1_bundle.npz '+common+' --out-dir results/fpga_block1'
    readme = ('Qwen6 block-1 prefill package\n\n'
              'Use the Linux serial console: COM17, 115200 baud, 8N1, no flow control.\n'
              'Ethernet/SSH is not required. Transfer this large ZIP by USB storage or an SD data partition.\n'
              'Extract this directory on the ZCU106. '
              'Use the Python interpreter from its PYNQ/NumPy environment and usual FPGA access permissions.\n'
              'This package was not validated on a physical board. Loading it replaces the active PL design.\n\n'
              'First run the six-engine smoke test:\n'+smoke+'\n\nThen run the real block:\n'+run+'\n\n'
              'Inspect results/fpga_block1/fpga_summary.json and fpga_block1_prefill.npz.\n'
              'Only transformer block index 0, prefill, is executed. No generation or decode.\n')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name,path in sources.items():
            with path.open('rb') as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
                manifest['sha256'][name] = digest.hexdigest()
            archive.write(path,name)
        archive.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
        archive.writestr('RUN_ON_BOARD.txt',readme)
    print(f'Board package: {args.out} ({args.out.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
