"""Verified serial bridge to physical Qwen block-0 prefill; no CPU fallback."""
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import uuid

import numpy as np
import serial

ROOT = Path(__file__).resolve().parent
BOARD = '/home/xilinx/qwen6_20261005'


class Console:
    def __enter__(self):
        self.port = serial.Serial('COM17', 115200, timeout=.2, write_timeout=10)
        self.port.write(b'\r')
        time.sleep(.3)
        self.port.read_all()
        self.command('stty -echo')
        return self

    def command(self, command, timeout=30):
        marker = 'CHAT_' + uuid.uuid4().hex
        self.port.write((command + "; printf '\\n" + marker + ":%s\\n' $?\r").encode())
        data = b''
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data += self.port.read(max(1, self.port.in_waiting))
            needle = ('\n'+marker+':').encode()
            if needle in data and b'\n' in data.split(needle, 1)[1]:
                output, suffix = data.split(needle, 1)
                code = suffix.splitlines()[0].strip()
                if code != b'0':
                    raise RuntimeError(output.decode(errors='replace')[-4000:])
                return output.decode(errors='replace')
        raise TimeoutError('FPGA console command timed out; no software fallback was used.')

    def upload(self, data, path):
        encoded = base64.b64encode(data).decode()
        self.command('true > '+path+'.b64')
        for start in range(0, len(encoded), 1800):
            self.command("printf '%s' '"+encoded[start:start+1800]+"' >> "+path+'.b64')
        self.command('base64 -d '+path+'.b64 > '+path)

    def __exit__(self, *exc):
        self.port.write(b'stty echo\r')
        self.port.close()


def prefill(model_path, ids, progress=lambda text: None):
    from export_fpga_bundle import Checkpoint
    from src.qwen_fpga6 import WEIGHT_KEYS
    from src.ps_int8_quantization import round_bf16
    if not 1 <= len(ids) <= 128:
        raise ValueError('FPGA prefill supports 1–128 tokens.')
    checkpoint = Checkpoint(model_path)
    config = json.loads((Path(model_path)/'config.json').read_text())
    hashes = {key: hashlib.sha256(checkpoint.read('model.layers.0.'+key).tobytes()).hexdigest() for key in WEIGHT_KEYS}
    frequency = 1 / (np.float32(config['rope_theta']) ** (np.arange(0, config['head_dim'], 2, dtype=np.float32)/config['head_dim']))
    phase = np.arange(len(ids), dtype=np.float32)[:, None]*frequency[None, :]
    phase = np.concatenate((phase, phase), axis=1)
    patch = io.BytesIO()
    np.savez_compressed(patch, input=checkpoint.read('model.embed_tokens.weight', ids),
                        rotary=round_bf16(np.concatenate((np.cos(phase), np.sin(phase)), axis=1)),
                        metadata=np.array(json.dumps({'config': config, 'token_ids': ids, 'hashes': hashes})))
    run_id = uuid.uuid4().hex
    out = ROOT/'results'/'fpga_chat'/run_id
    out.mkdir(parents=True)
    (out/'request.json').write_text(json.dumps({'token_ids': ids, 'weight_hashes': hashes}, indent=2))
    progress('Transferring this prompt to FPGA over COM17…')
    with Console() as console:
        console.command('mkdir -p '+BOARD+'/chat_runs/'+run_id)
        remote = BOARD+'/chat_runs/'+run_id
        console.upload(patch.getvalue(), remote+'/input.npz')
        console.upload((ROOT/'fpga/qwen6/chat_prefill.py').read_bytes(), remote+'/run.py')
        progress('FPGA block-0 prefill running on physical engines (typically several minutes)…')
        log = console.command('cd '+BOARD+' && sudo -n /usr/local/share/pynq-venv/bin/python3 fpga/qwen6/board_dma_heap.py '+remote+'/run.py '+remote, timeout=1800)
        (out/'board.log').write_text(log)
        response = console.command("/usr/local/share/pynq-venv/bin/python3 -c \"import base64;print('RESULT:'+base64.b64encode(open('"+remote+"/output.npz','rb').read()).decode())\"", timeout=120)
    encoded = response.split('RESULT:', 1)[1].splitlines()[0].strip()
    raw = base64.b64decode(encoded, validate=True)
    (out/'output.npz').write_bytes(raw)
    with np.load(io.BytesIO(raw), allow_pickle=False) as result:
        arrays = {key: result[key].copy() for key in ('output', 'key_cache', 'value_cache')}
        report = json.loads(str(result['report']))
    if report['token_ids'] != ids or report['weight_hashes'] != hashes or not report['physical_hardware']:
        raise ValueError('FPGA response provenance mismatch')
    expected = {'output': (len(ids), 2048), 'key_cache': (8, len(ids), 128), 'value_cache': (8, len(ids), 128)}
    for key, value in arrays.items():
        if value.shape != expected[key] or not np.isfinite(value).all():
            raise ValueError('Invalid FPGA output: '+key)
    (out/'fpga_summary.json').write_text(json.dumps(report, indent=2))
    progress('Verified physical FPGA output and all block-0 weight hashes.')
    return arrays, report
