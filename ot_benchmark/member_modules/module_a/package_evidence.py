"""Archive complete module evidence; omit environment/cache and forbid partial results."""
import argparse
import json
from pathlib import Path
import zipfile
from core import ROOT, sha, write_json

def package(out):
    audit=json.loads((out/'results/audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and audit['observed_calls']==432
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
              if p.is_file() and p.name!='manifest.json'}
    write_json(out/'manifest.json',manifest)
    archive=out.with_name(out.name+'_evidence.zip')
    prefix=Path('ot_benchmark/member_modules/module_a')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,str(prefix/out.name/p.relative_to(out)))
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in ('.py','.md','.txt'):z.write(p,str(prefix/p.name))
        shared=ROOT.parents[1]/'ot_benchmark/algorithms/transport_diagnostics.py'
        z.write(shared,'ot_benchmark/ot_benchmark/algorithms/transport_diagnostics.py')
        z.writestr('DELIVERY_README.txt',
            'Module A complete evidence and standalone module source.\n'
            'Python 3.12 required by the frozen dependency lock.\n'
            'Read ot_benchmark/member_modules/module_a/README.md.\n'
            'Run the reproduction commands from this extracted root using a new output directory.\n'
            'Original experiment arrays, all native logs and returned plans are included.\n'
            'No environment, unrelated caches, real datasets or new solver are included.\n')
    print(archive.resolve());print('SHA256',sha(archive))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    package(p.parse_args().output)
