"""Run with MERGED PYTHONPATH; stage objects and recalled pool are shared."""
import contextlib
from pathlib import Path
from harness import parser,run
from compare import report

if __name__=='__main__':
    ap=parser();ap.add_argument('--report',default='ablation.md');ap.add_argument('--csv',default='ablation-per-query.csv')
    args=ap.parse_args()
    from mnemostack.recall.pipeline.presets import build_stateless_pipeline
    order=[type(s).__name__ for s in build_stateless_pipeline().stages]
    assert order.index('CalibrateRelevance')<order.index('ExactTokenRescue'), 'ablation requires MERGED tree'
    result=run(args,ablation=True)
    with Path(args.report).open('w') as f,contextlib.redirect_stdout(f):
        report(result,result,args.csv,[('base_order','merged_order')])
