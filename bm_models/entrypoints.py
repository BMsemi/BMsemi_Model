from .core.mapping import LayerDims, PEGeometry, map_layer as calculate
from neuromorphic.registry import OperationUnavailable

def map_layer(p):
    pe=PEGeometry(rows=int(p.get("pe_rows",64)),cols=int(p.get("pe_cols",32)))
    dims=LayerDims(**{k:int(p[k]) for k in ("K","C","R","S","Y","X") if k in p})
    return calculate(dims,pe).to_dict()

def pe_self_test(p):
    raise OperationUnavailable("PE self-test requires target=fpga and the ZCU106 board runner")
