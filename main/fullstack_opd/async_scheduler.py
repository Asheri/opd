"""★ Stage 2 调度器 —— AsyncOPD 全异步调度器上跑 Direct-OPD 训练。

对应真实代码：async-opd/opd/coordinator/streaming.py::StreamCoordinator
  （4 线程 PromptFeeder / RolloutCollector / TeacherScorer / TrainDispatcher）
  + async-opd/opd/streaming_stages.py

本 demo 用同样的「四阶段解耦」结构，但 teacher 打分来自 Lightning 离线缓存
（无 live teacher），student 的奖励来自 Direct-OPD 的 Δ_T（迁移对象是策略偏移）：

  PromptFeeder      : 从 prompt 池喂样本索引
  RolloutCollector  : 用（可能陈旧的）student 权重快照生成 on-policy rollout，
                      打上生成时的版本号，投入 StalenessQueue
  TeacherScorer     : 从 Lightning 离线缓存取 Δ_T 贴到样本上（★无 live teacher）
  TrainDispatcher   : 取出（可能陈旧）样本，用【当前】student 重算 logp →
                      PPO clip 处理陈旧 → Direct-OPD 奖励 = E_student[Δ_T]
                      + low-var KL 正则（Lightning 隐式正则，防漂移）→ 更新并 publish 新权重

这样三重限制同时被打破：
  · 常驻教师  → TeacherScorer 查离线 cache，不启 teacher server
  · 同步等待  → rollout 与 learner 由队列解耦，消费陈旧样本
  · 迁移终态  → 奖励是 Δ_T（RL 策略偏移），作用于更强 student 自身的 on-policy 状态
"""

from __future__ import annotations

import queue
import threading

import torch

from .buffer import StalenessQueue, WeightStore, TeacherArtifactBuffer
from .direct_opd import delta_opd_reward_expected
from .losses import policy_gradient_kl, low_var_kl
from .models import ToyModel


class AsyncOPDScheduler:
    def __init__(self, student: ToyModel, cache, prompts: list, responses: list,
                 cfg: dict, device):
        self.student = student.to(device)
        self.cache = cache
        self.prompts = prompts
        self.responses = responses
        self.cfg = cfg
        self.device = device

        # 共享设施
        # ★ 修复：staleness_q 兼任「版本追踪 + scored 样本队列」（AsyncOPD 的真实形态），
        #    不再实例化一个只读版本号、队列本体闲置的摆设。
        self.staleness_q = StalenessQueue(cfg.get("staleness_threshold", 4))
        self.weight_store = WeightStore()
        self.teacher_buf = TeacherArtifactBuffer(cache, max_batches=3)
        self.stop = threading.Event()
        self.metrics: list = []

        # 一个独立 worker 模型，供 RolloutCollector / TeacherScorer 加载快照使用
        self.worker = ToyModel(vocab=student.vocab, d_model=student.d_model,
                               n_layers=_n_layers(student)).to(device)

        # student 自身初始分布作为 ref（防止策略漂移，对应 Lightning 隐式正则）
        self.student_ref_dists = []
        with torch.no_grad():
            for p, r in zip(prompts, responses):
                self.student_ref_dists.append(
                    student.response_distributions(p, r, device).cpu()
                )

        # 发布初始权重快照
        self._publish()

    # --------------------------- 权重同步 ---------------------------
    def _publish(self):
        v = self.weight_store.publish(self.student.state_dict())
        self.staleness_q.advance_version()
        return v

    def _acquire_snapshot(self):
        return self.weight_store.acquire()

    # --------------------------- 四个解耦阶段 ---------------------------
    def _prompt_feeder(self, n_steps: int):
        # 持续供给，直到 learner 完成（真实 async 模式：数据流式喂入，不被固定上限截断）
        i = 0
        while not self.stop.is_set():
            try:
                self._pq.put(i % len(self.prompts), timeout=0.5)
            except queue.Full:
                continue  # 队列满则让出，重新检查 stop
            i += 1

    def _rollout_collector(self):
        """用可能陈旧的 student 快照，对*离线固定 rollout* 重算 student 分布，打版本号入队。

        离线模式（Lightning-OPD 的设定）：训练数据 = 预收集的 SFT rollouts
        （self.responses），teacher 的 Δ_T 也是在这些固定序列上缓存的，因此 student
        评分必须对齐到同一序列（对齐保证「无常驻 teacher」下 Δ_T 仍然精确）。
        student 的参数仍是最新的（learner 重算 s_cur 时反传），梯度是 on-policy 的。
        """
        while not self.stop.is_set():
            try:
                idx = self._pq.get(timeout=1)
            except queue.Empty:
                continue
            snap, ver = self._acquire_snapshot()
            self.worker.load_state_dict(snap)
            self.worker.eval()
            p = self.prompts[idx].to(self.device)
            r = self.responses[idx].to(self.device)   # 离线固定 rollout（Lightning）
            # 用该（陈旧）快照重算 student 分布，作为 stale 数据
            with torch.no_grad():
                s_old = self.worker.response_distributions(p, r, self.device).cpu()
            try:
                self._rq.put((idx, r.cpu(), s_old, ver), timeout=0.5)
            except queue.Full:
                continue  # 队列满则丢弃该样本（async OPD 对丢样本鲁棒），重新检查 stop

    def _teacher_scorer(self):
        """从 Lightning 离线缓存取 Δ_T 贴到样本（★无 live teacher）。

        贴好标签的样本投入 StalenessQueue：入队侧先做一道陈旧度截断（put 返回
        False = 太旧直接丢弃），消费侧 dispatcher 还会再截一次（AsyncOPD 双保险）。
        """
        while not self.stop.is_set():
            try:
                item = self._rq.get(timeout=1)
            except queue.Empty:
                continue
            idx, r, s_old, ver = item
            rl_D, ref_D = self.teacher_buf.assemble(idx)   # 离线缓存
            try:
                self.staleness_q.put((idx, r, s_old, rl_D, ref_D), version=ver, timeout=0.5)
            except queue.Full:
                continue  # 队列满则丢弃，重新检查 stop

    def _train_dispatcher(self, n_steps: int):
        """learner：用当前 student 重算 → PPO clip 陈旧 → Direct-OPD 奖励 → 更新。

        ★ 修复为 token 级标准 PPO（对齐 verl）：
          - adv_t = E_{π_old}[Δ_T](s_t)：rollout 时刻确定的 token 级标量奖励，
                    随样本流转（可以是陈旧的）——即 verl 里的 rm_scores；
          - ratio_t = exp(logp_cur(a_t) − logp_old(a_t))：learner 时刻用当前
                    student 重算，只对 importance ratio 做 recompute（AsyncOPD 代理）。
        """
        opt = torch.optim.Adam(self.student.parameters(), lr=self.cfg.get("lr", 1e-3))
        kl_coef = self.cfg.get("kl_reg_coef", 0.05)
        clip_eps = self.cfg.get("clip_eps", 0.2)
        grad_clip = self.cfg.get("grad_clip", 1.0)
        done = 0
        while done < n_steps:
            try:
                (idx, r, s_old, rl_D, ref_D), ver, _age_at_put = self.staleness_q.get(timeout=10)
            except queue.Empty:
                continue
            # 陈旧度截断（消费侧）：丢弃过旧样本（AsyncOPD 的 staleness_threshold）
            if (version := self.staleness_q.current_version) - ver > self.cfg.get("staleness_threshold", 4):
                continue
            self.student.train()
            p = self.prompts[idx].to(self.device)
            rdev = r.to(self.device)
            rl_D = rl_D.to(self.device)
            ref_D = ref_D.to(self.device)
            s_old = s_old.to(self.device)

            # learner 时刻用【当前】student 重算分布（recompute 代理）
            s_cur = self.student.response_distributions(p, rdev, self.device)
            mask = torch.ones(s_cur.size(0), device=self.device)

            # Direct-OPD 迁移对象（密集奖励）：Δ_T = logπ_rl − logπ_ref
            delta = rl_D - ref_D                                  # (T, V)

            # PG 损失：按 π_old 加权的逐 vocab 重要性采样 + PPO clip 陈旧截断
            # （ratio=1 时精确等于 −E_{π_cur}[Δ_T]，一阶梯度非零；详见 losses.py 注释）
            pg_loss = policy_gradient_kl(s_cur, s_old, delta, mask, clip_eps)

            # 监控指标（不参与反传）：
            #   adv_tok         = rollout 时刻 π_old 下的 token 级期望奖励（随样本流转）
            #   expected_reward = 当前 student 的期望策略偏移（应随训练上升）
            adv = delta_opd_reward_expected(s_old, rl_D, ref_D, mask)
            expected_reward = delta_opd_reward_expected(s_cur, rl_D, ref_D, mask).mean()

            # low-var KL 正则到 student 初始分布（防漂移，Lightning 隐式正则）
            ref_d = self.student_ref_dists[idx].to(self.device)
            kl_loss = low_var_kl(s_cur, ref_d, mask)

            loss = pg_loss + kl_coef * kl_loss
            opt.zero_grad()
            loss.backward()
            # ★ 修复：梯度裁剪（真实 RL 训练标配，防止陈旧样本的大 ratio 造成梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), grad_clip)
            opt.step()

            version = self._publish()
            age = version - ver
            self.metrics.append({
                "step": done,
                "version": version,
                "age": age,
                "loss": float(loss.item()),
                "pg_loss": float(pg_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "adv_mean": float(adv.mean().item()),
                "reward": float(expected_reward.item()),
            })
            done += 1
        self.stop.set()  # 通知其余线程退出

    # --------------------------- 入口 ---------------------------
    def run(self, n_steps: int):
        self._pq: "queue.Queue" = queue.Queue(maxsize=self.cfg.get("queue_size", 8))
        self._rq: "queue.Queue" = queue.Queue(maxsize=self.cfg.get("queue_size", 8))
        # scored 队列 = self.staleness_q（构造时已建，版本追踪与样本流转一体）

        threads = [
            threading.Thread(target=self._rollout_collector, name="RolloutCollector"),
            threading.Thread(target=self._teacher_scorer, name="TeacherScorer"),
        ]
        for t in threads:
            t.start()

        # 喂 prompt（后台线程避免阻塞 dispatcher）
        feeder = threading.Thread(target=self._prompt_feeder, args=(n_steps,),
                                  name="PromptFeeder")
        feeder.start()

        self._train_dispatcher(n_steps)

        feeder.join(timeout=5)
        for t in threads:
            t.join(timeout=5)
        return self.metrics


def _n_layers(model: ToyModel) -> int:
    return len(model.enc.layers)
