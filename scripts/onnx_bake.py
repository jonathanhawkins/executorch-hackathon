"""Bake onnx2torch shims that read shape args from tensors into constants.

onnx2torch's Pad/Slice/Resize modules take their pads, bounds and scales as
runtime tensors, which torch.export cannot trace. Every one of them is a graph
initializer, so bake(m) folds the value in and drops the extra args. Without
it, exporting inswapper_128 dies on "unhashable type: non-nested SymInt".
"""

import torch, torch.nn as nn, onnx2torch
from onnx2torch.node_converters import pad as padmod

class ConstPad(nn.Module):
    def __init__(s, pads, mode, value):
        super().__init__(); s.pads=[int(x) for x in pads]; s.mode=mode; s.value=value
    def forward(s, x, *a):
        tp = padmod._torch_padding_to_mode_format(padmod._onnx_padding_to_torch(s.pads), s.mode)
        return torch.nn.functional.pad(x, tp, mode=s.mode, value=s.value)

class ConstSlice(nn.Module):
    def __init__(s, starts, ends, axes, steps):
        super().__init__(); s.sl=(starts, ends, axes, steps)
    def forward(s, x, *a):
        starts, ends, axes, steps = s.sl
        idx=[slice(None)]*x.dim()
        for st,en,ax,sp in zip(starts,ends,axes,steps):
            ax=int(ax)%x.dim(); idx[ax]=slice(int(st), min(int(en), 1<<31), int(sp))
        return x[tuple(idx)]

class ConstResize(nn.Module):
    def __init__(s, scales, mode):
        super().__init__(); s.scale=[float(x) for x in scales[-2:]]; s.mode=mode
    def forward(s, x, *a):
        return torch.nn.functional.interpolate(x, scale_factor=s.scale, mode=s.mode,
            align_corners=False if s.mode in ('bilinear','bicubic') else None)

def const_of(mod, node):
    v = mod.get_submodule(node.target)() if node.op=='call_module' else mod.get_buffer(node.target)
    return v.tolist()

def bake(m):
    g=m.graph; counts={}
    for node in list(g.nodes):
        if node.op!='call_module': continue
        sub=m.get_submodule(node.target); tn=type(sub).__name__
        parent,_,name=node.target.rpartition('.')
        host=m.get_submodule(parent) if parent else m
        try:
            if 'Pad' in tn:
                new=ConstPad(const_of(m,node.args[1]), getattr(sub,'mode','constant'), 0.0)
            elif 'Slice' in tn:
                a=[const_of(m,x) for x in node.args[1:]]
                starts,ends=a[0],a[1]
                axes=a[2] if len(a)>2 else list(range(len(starts)))
                steps=a[3] if len(a)>3 else [1]*len(starts)
                new=ConstSlice(starts,ends,axes,steps)
            elif 'Resize' in tn:
                vals=[const_of(m,x) for x in node.args[1:]]
                scales=next((v for v in vals if len(v)>=2 and all(float(z)>0 for z in v)), [1,1,2,2])
                mode=getattr(sub,'mode','nearest')
                new=ConstResize(scales, 'nearest' if 'nearest' in str(mode) else 'bilinear')
            else:
                continue
        except Exception as e:
            print("skip", tn, e); continue
        setattr(host,name,new); node.args=(node.args[0],)
        counts[tn]=counts.get(tn,0)+1
    g.eliminate_dead_code(); g.lint(); m.recompile()
    return counts
