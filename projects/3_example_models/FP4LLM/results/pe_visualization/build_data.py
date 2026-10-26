"""Build the viewer's data from the selected mappings and saved compiler results."""
from pathlib import Path
import json
import sys
import hashlib

target = Path(__file__).resolve().parent
repo = target.parents[1]
sys.path.insert(0, str(repo / 'x1_llm_compiler'))
from x1_llm_compiler.mapping import parse_mapping_file
results = repo / 'results/x1_prompt_run'
manifest = json.loads((results / 'best_manifest.json').read_text())
stream = json.loads((results / 'best_stream_plan.json').read_text())
plans = {(p['layer']['phase'], p['layer']['name']): p for p in manifest['layers']}
data = {'fabric': stream['fabric'], 'summary': stream['summary'], 'config': manifest['compiler_config'], 'sources': {}, 'phases': {}}
for phase in ('prefill', 'decode'):
    path = results / f'best_{phase}_mapping.m'
    _, layers = parse_mapping_file(path, phase)
    assert len(layers) == 224
    data['sources'][phase] = {'file': str(path.relative_to(repo)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    data['phases'][phase] = []
    for layer in layers:
        plan = plans[phase, layer.name]
        assert plan['layer'] == layer.as_dict()
        assert plan['row_tiles'] * plan['col_tiles'] * plan['token_blocks'] == plan['fp4llm_tile_tasks']
        windows = [w for w in stream['windows'] if w['phase'] == phase and w['layer_name'] == layer.name]
        data['phases'][phase].append({'layer': layer.as_dict(), 'plan': {k: plan[k] for k in ('row_tiles','col_tiles','token_blocks','fp4llm_tile_tasks')}, 'windows': [{'id': w['window_id'], 'cells':w['programmed_cells'], 'die':w['assigned_die']} for w in windows]})
payload = json.dumps(data, separators=(',', ':'))
(target / 'mapping-data.json').write_text(payload, encoding='utf-8')
(target / 'mapping-data.js').write_text('export default ' + payload + ';\n', encoding='utf-8')
print(f'Validated and exported 448 layer plans; {len(payload):,} bytes.')
