"""Export original BF16 words for the saved run's linear inputs and outputs."""
from pathlib import Path
import base64
import hashlib
import json
import re
import struct

root = Path(__file__).resolve().parent
source = root.parent / 'x1_prompt_run/software_crossbar/x1_prompt_run_bf16_values.txt'
tensors = {}
current = None
with source.open(encoding='utf8') as handle:
    for line in handle:
        if line.startswith('# phase='):
            match = re.match(r'# phase=(\w+) tensor=(\w+) dtype=BF16 shape=(\[[^\]]+\]) numel=(\d+)', line)
            phase, name, shape, count = match.groups()
            current = {'shape': json.loads(shape), 'raw': bytearray(), 'count': int(count)}
            tensors[f'{phase}/{name}'] = current
        elif current is not None and not line.startswith('#'):
            fields = line.rstrip().split('\t')
            assert int(fields[2]) == len(current['raw']) // 2
            current['raw'].extend(struct.pack('<H', int(fields[4], 16)))
for value in tensors.values():
    assert len(value['raw']) == value.pop('count') * 2
    value['words'] = base64.b64encode(value.pop('raw')).decode()
data = {'source': str(source.relative_to(root.parents[1])).replace('\\', '/'),
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'tensors': tensors}
(root / 'activation-preview.js').write_text('export default ' + json.dumps(data, separators=(',', ':')) + ';\n', encoding='utf8')
print(f'Exported {len(tensors)} saved tensors with original BF16 words.')
