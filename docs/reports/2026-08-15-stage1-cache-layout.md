# Stage 1 §2：TensorTeacherCache 真实缓存布局分析 → 最小 sufficient statistics

> 依据《Stage 1：Teacher Cache 存储架构重构提示词》§2，逐字段分析当前 `TensorTeacherCache`
> （`main/fullstack_opd_v2/cache.py`）的 top-K 稀疏缓存布局，确定训练真正需要的最小数据集，
> 凡可从其他字段恢复的数据一律不重复持久化。

## 当前 top-K 稀疏缓存字段（`cache.py` L42-49）

| field            | dtype   | shape      | bytes/元素 | required_for_training | can_recompute                      |
|------------------|---------|------------|-----------|-----------------------|------------------------------------|
| `ids`            | int64   | (N,T,K)    | 8         | 否（仅取 `Kt=size(-1)`，L143） | 可由 `ids_sorted` 排序反解（不存） |
| `rl_k`           | float32 | (N,T,K)    | 4         | 否                     | 可（`delta_k + ref_k`，不存）      |
| `ref_k`          | float32 | (N,T,K)    | 4         | 否                     | 可（`rl_k − delta_k`，不存）       |
| `delta_k`        | float32 | (N,T,K)    | 4         | 否（仅 build 中间量，L104） | 可（`rl_k − ref_k`，不存）    |
| `ids_sorted`     | int64   | (N,T,K)    | 8→4 (int32) | **是**（searchsorted 二分，L150） | 否                          |
| `delta_k_sorted` | float32 | (N,T,K)    | 4         | **是**（gather 对齐 Δ，L153） | 由 `ids_sorted` 的 sort order 重排 `delta_k`（但 build 一次算好即可） |
| `response_length`| uint32  | (N,)       | 4         | **是**（变长 mask，§7） | 否（需从数据算，`ΣL`）          |
| `vocab`          | int     | scalar     | —         | **是**（跨词表展开维度，L158） | 否                            |

## 消费点核对（训练热路径）

`_train_step`（`scheduler.py:312-338`）经 `TensorTeacherCache.delta_for_student_topk`
（`cache.py:121-165`）取 Δ_T，实际只读：
- `ids_sorted[idxs]`（L148）——searchsorted 定位 student top-K 在 teacher 支撑中的位置；
- `delta_k_sorted[idxs]`（L149）——gather 出对齐的 Δ。

`losses.py`（`pg_loss`/`low_var_kl_support`/`expected_reward`）只接收展开的 dense `delta` 或
top-K 支撑张量，**从不直接读 cache 字段**——一切展开/重排逻辑集中在 `delta_for_student_topk`。

## 判定

- **训练必需**：`ids_sorted`、`delta_k_sorted`、`vocab`（+ 变长 `response_length`）。
- **可舍（不持久化）**：`ids`、`rl_k`、`ref_k`、`delta_k`。它们全是 build 中间量：
  - `rl_k`/`ref_k` → 只为算 `delta_k = rl_k − ref_k`；
  - `delta_k` → 只为按 sort order gather 出 `delta_k_sorted`；
  - `ids` → 只为 sort 出 `ids_sorted`。
  这些在 build 后即无保留价值，落盘纯属浪费。

## 最小 sufficient statistics（本阶段持久化）

```
ids_sorted      int32   (N, T_max, K)   4 字节/元素
delta_k_sorted  float32 (N, T_max, K)   4 字节/元素
response_length uint32  (N,)            4 字节/元素
vocab           int     scalar
```

**落盘体积 = `8·N·T_max·K + 4·N` 字节**（当前持久化 6 张量 = `32·N·T_max·K`，**约 4× 缩减**）。

> 注：`ids_sorted` 由 int64 降为 int32——token id < 2³¹（真实词表 ≤152k），int32 足够且省一半。
> `ids`（未排序）在训练期不参与检索，故不持久化；磁盘上只存 `ids_sorted`。

## 规模核算（50K×8192×K16）

| 布局 | 字节 | 说明 |
|------|------|------|
| 当前全 pers 6 张量 | `32×50000×8192×16 ≈ 210GB` | 不可行 |
| 最小 + 磁盘 mmap | `8×50000×8192×16 + 4×50000 ≈ 52GB` | 磁盘驻留可行，GPU/RAM 只驻 batch 行 |

配合 §3 disk-backed mmap + §4 batch-local 加载，GPU 只持有当前 batch 的 `(B,T,K)` 切片，
不再需要全量 resident cache。