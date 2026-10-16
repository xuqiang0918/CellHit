# CellHit SDAA 适配

## 结论

CellHit 为 XGBoost 药物响应预测模型（AutoXGBRegressor，scikit-learn / Optuna 体系，**不含 PyTorch**），
SDAA 适配路径为 **CPU**：SDAA 容器内 xgboost 2.1.1 为纯 CPU 构建（`USE_CUDA=False`，import 时提示
"not compiled with CUDA"），无 GPU 后端可走；CUDA 侧 xgboost 官方 wheel 自带 CUDA 构建可直接走 GPU。
适配改动 3 处文件，均为 XGBoost `device` 参数的环境变量化与 CPU 采样法回退，SDAA 侧以官方
`external_inference` 直调函数方式验收，drug 1032 全量外部推理结果与 CUDA 逐点一致。

## 环境

| 项      | SDAA 侧                                                        |
| ------- | -------------------------------------------------------------- |
| 机器    | 10.71.15.115 容器 `sdaa`（镜像 3.2.1-torch_sdaa3.2.1）         |
| Python  | 3.12.13（/home/py312/bin/python3）                             |
| xgboost | 2.1.1（纯 CPU 构建，import 时提示 "not compiled with CUDA"）   |
| 数据    | celligner_CCLE_TCGA.feather（1.93 GB）+ 作者发布 20 药模型 pkl |

## 改动

均为 `[sdaa-adapt]` 单笔提交；`CELLHIT_DEVICE` 未设置时默认值与原版 `cuda:{gpu_id}` 完全一致，
NVIDIA 平台行为不变。

- `CellHit/models/auto_models.py`：`AutoXGBRegressor` 超参搜索的 `device` 硬编码 `cuda:{self.gpuID}`，
  SDAA 上无 CUDA 后端无法运行。改为 `os.environ.get('CELLHIT_DEVICE', ...)`；CPU 设备不支持
  `gradient_based` 采样，非 cuda 时强制 `uniform`。
- `CellHit/search_and_inference/hp_search.py`：搜索得到的 `best_params['device']` 同样硬编码，
  改为同一环境变量控制。
- `scripts/search_and_inference/external_inference.py`：载入作者现成模型 pkl 后对 `best_params['device']`
  做相同参数化，并对 pkl 内已存的 `gradient_based` 采样法做 CPU 回退。

## 运行

external_inference 是仓库官方的外部推理入口
（`scripts/search_and_inference/external_inference.py` 的 `external_inference()` 函数）：
载入作者随仓库提供的现成模型 pkl（每药一个，如 `models/1032.pkl` —— 作者在 GDSC 数据集上
训练完存好的 20 个 XGBoost 模型集成与最优超参，无需自己重训），对 GDSC 训练集之外的转录组
数据（celligner 对齐后的 CCLE / TCGA 细胞系表达谱）预测药物响应 LN_IC50（log10 μM），
并输出逐样本 SHAP 贡献。本验收即用该官方入口直接调函数。

前置：`results/gdsc/search_and_inference/moa_primed/models/<drugID>.pkl`（作者现成模型，见上）与
`data/transcriptomics/celligner_CCLE_TCGA`（feather 同目录无扩展名软链）就位。

```bash
CELLHIT_DEVICE=cpu python sdaa/scripts/run_external_inference.py 1032
```

推理数据的指定（脚本内三处参数，各管一件事）：

- 推理用的表达谱由 `celligner_output_path` 指定：仓库 `data/transcriptomics/celligner_CCLE_TCGA`
  （1.93 GB feather，作者随仓库提供的 CCLE+TCGA 经 celligner 对齐后的合并表达谱，14197 个细胞系）。
  加载链路 `external_inference → prepare_data → DatasetLoader`，`dataset_loaders.py:34` 即
  `pd.read_feather(celligner_output_path)`。换推理数据 = 把该参数指向另一份同格式 feather。
- `dataset="gdsc"` 不是推理数据：它决定去 `results/gdsc/...` 找模型 pkl、写结果目录，
  以及 GDSC 训练数据的路径组织。
- `external_dataset="None"` 为官方约定（字符串 'None'），表示不追加额外外部数据集。

产出 `results/gdsc/search_and_inference/moa_primed/full_inference/1032.csv`
（14197 行 = CCLE 1450 + TCGA 12747，列 DrugName/DrugID/Index/Predictions/Stds/Source）；
`return_stds=True` 时同时产出 `full_inference_shaps/1032.csv`（SHAP 逐样本逐基因贡献）。

## 快速验证

公共变量：`REPO` 为 CellHit 仓库根；两侧各跑一次 `run_external_inference.py`，把两侧的
`full_inference/1032.csv` 与 `full_inference_shaps/1032.csv` 拷到一起再对照。

| 脚本                      | 用途                         | 命令                                                                                           | 产出                    |
| ------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------- |
| run_external_inference.py | 单药全量外部推理（两侧通用） | `CELLHIT_DEVICE=cpu python sdaa/scripts/run_external_inference.py 1032`                        | full_inference/1032.csv |
| compare_outputs.py        | 两平台输出指标对照           | `python sdaa/scripts/compare_outputs.py <sdaa_csv> <cuda_csv> --shaps <sdaa_shap> <cuda_shap>` | 终端打印对照指标        |

预期：逐点 `pearson r ≈ 0.994`、`absdiff mean ≈ 0.054`；SHAP `top50 overlap 42/50`、top5 基因同序。

## 性能

| 测例                         | 数据规模                     | CUDA 耗时（1×A100-40GB，device=cuda:0） | SDAA 耗时（CPU，xgboost 默认并行） | 倍数  |
| ---------------------------- | ---------------------------- | --------------------------------------- | ---------------------------------- | ----- |
| drug 1032 external_inference | 14197 样本 × 20 模型 × 15 折 | ≈230 s                                  | 1765 s                             | ≈7.7× |

CUDA 侧依据 10.10.4.14（taichu04）运行窗口：ei_1032.py 启动至 1032.csv 落盘 226 s；
h100-3 单卡复测主体 15 折推理 273 s 交叉佐证同量级。SDAA 侧 xgboost 无 GPU 后端，
CPU 路径即当前平台上限。

## 精度

对照方式：两侧同输入（同一 feather、同一模型 pkl、同 `random_state=0`）各跑一次 drug 1032
全量外部推理，`compare_outputs.py` 输出对照。

预测值逐点对照（SDAA vs CUDA，14197 样本）：

| 指标                     | 值                       |
| ------------------------ | ------------------------ |
| Pearson r                | 0.99409                  |
| absdiff mean / p95 / max | 0.0542 / 0.1410 / 0.3088 |
| <0.1 占比                | 85.0%                    |

SHAP 特征贡献对照（full_inference_shaps，6203 基因 mean|shap|）：

| 指标                   | SDAA vs CUDA                             |
| ---------------------- | ---------------------------------------- |
| mean\|shap\| Pearson r | 0.9802                                   |
| top50 基因重合         | 42/50                                    |
| top5 基因              | 完全同序：DAPP1, ERBB2, GRB7, WSB2, MFNG |

预测分布对照（Predictions 逐来源统计）：

| 来源            | SDAA mean | CUDA mean | SDAA med | CUDA med |
| --------------- | --------- | --------- | -------- | -------- |
| CCLE（n=1450）  | 1.6777    | 1.6733    | 1.7267   | 1.7201   |
| TCGA（n=12747） | 1.6684    | 1.6635    | 1.7342   | 1.7263   |
