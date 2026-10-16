#!/usr/bin/env python
"""CellHit 单药 external_inference 两侧通用跑法。

用法（在仓库根目录执行）：
    SDAA 侧:  CELLHIT_DEVICE=cpu python sdaa/scripts/run_external_inference.py 1032
    CUDA 侧:  CUDA_VISIBLE_DEVICES=0 python sdaa/scripts/run_external_inference.py 1032

设备语义：CELLHIT_DEVICE 未设时按仓库原逻辑走 cuda:{gpu_id}。
SDAA 容器内 xgboost 为纯 CPU 构建，必须显式 CELLHIT_DEVICE=cpu。

前置：results/gdsc/search_and_inference/moa_primed/models/<drugID>.pkl（作者发布模型）
与 data/transcriptomics/celligner_CCLE_TCGA 软链已就位。
产出：results/gdsc/search_and_inference/moa_primed/full_inference/<drugID>.csv
"""
import importlib.util
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "ei_module", REPO / "scripts" / "search_and_inference" / "external_inference.py"
)
ei = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ei)

drug_id = int(sys.argv[1])
t0 = time.time()
ei.external_inference(
    drugID=drug_id,
    dataset="gdsc",
    external_dataset="None",  # 官方约定：无外部数据时传字符串 'None'
    results_path=REPO / "results",
    models_path=REPO / "results" / "gdsc" / "search_and_inference" / "moa_primed" / "models",
    return_stds=True,
    return_shaps=True,  # 部分上游版本 kwargs['return_shaps'] 无默认值，显式传对各版本均安全
    gpu_id=0,
    random_state=0,
    cv_iterations=15,
    data_path=REPO / "data",
    celligner_output_path=REPO / "data" / "transcriptomics" / "celligner_CCLE_TCGA",
)
print("ELAPSED_SECONDS=%.1f" % (time.time() - t0))
