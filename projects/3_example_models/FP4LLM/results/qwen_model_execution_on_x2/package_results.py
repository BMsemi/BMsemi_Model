"""Package only the first-block results and verification evidence for handoff."""
import hashlib
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
names = ['fpga_block1_prefill.npz', 'fpga_summary.json', 'accuracy.json',
         'hello.source.json', 'hello_input.npz', 'host_validation.json',
         'host_validation_outputs.npz', 'board_hello_run.txt', 'HANDOFF.txt']
checksums = ''.join(hashlib.sha256((root / name).read_bytes()).hexdigest() + '  ' + name + '\n' for name in names)
(root / 'SHA256SUMS.txt').write_text(checksums)
archive = root / 'qwen_model_execution_on_x2.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in names + ['SHA256SUMS.txt']:
        z.write(root / name, 'qwen_model_execution_on_x2/' + name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(f'{archive}: {archive.stat().st_size} bytes')
print('SHA256 ' + hashlib.sha256(archive.read_bytes()).hexdigest())
