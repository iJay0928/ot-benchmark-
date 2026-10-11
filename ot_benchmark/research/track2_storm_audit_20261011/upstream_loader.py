"""Load only inspected numerical methods from a hash-pinned public source.

No upstream imports, model downloads, or monkey-patching of source files.
"""
import ast
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

EXPECTED_SHA = '9a7756d28af7c076a381d9e9956b9d1a152e675e264a72c34fb86906c2d9eeb0'
METHODS = {
    '_compute_centroid_2d', '_exp_cost_weight', '_compute_cost_function',
    '_compute_cost_function_adj', 'sinkhorn', '_create_circular_distribution_image',
    '_create_circular_distribution', '_get_attention_maps_for_tokens',
    '_normalize_attention_map', '_apply_smoothing', '_storm_loss',
}


def load_numerics(source):
    source = Path(source)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError('Upstream source changed; inspect and repin before execution')
    tree = ast.parse(raw.decode('utf-8'))
    klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'StormPipeline')
    methods = [x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name in METHODS]
    if {x.name for x in methods} != METHODS:
        raise ValueError('Pinned numerical methods missing')
    isolated = ast.Module(body=[ast.ClassDef(name='StormNumerics', bases=[], keywords=[],
                                            body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = dict(torch=torch, F=F, Any=Any, Callable=Callable, Dict=Dict, List=List,
                     Optional=Optional, Tuple=Tuple, Union=Union,
                     logger=logging.getLogger('upstream_storm_audit'))
    exec(compile(isolated, str(source), 'exec'), namespace)
    instance = namespace['StormNumerics']()
    locations = {x.name: {'start': x.lineno, 'end': x.end_lineno} for x in methods}
    return instance, {'sha256': digest, 'methods': locations}
