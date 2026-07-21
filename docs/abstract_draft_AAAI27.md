# AAAI-27 Abstract Draft — NCR-Match

*Draft v1, 2026-07-18. 学习方法为主线；手工管线作为最强基线给真实数字；学习模型结果留 [XX] 占位符。*

---

## Working title（已选定）

**Same Negative or Same Scene? Auditable Candidate Generation and Risk-Controlled Matching for Archival Image Search**

（组合了候选 1 的问句钩子与候选 5 的副题；其余候选保留备查如下。）

**提问式（钩子强，AAAI 允许）**
1. **Same Negative or Same Scene? Risk-Controlled Reproduction Matching in Historical Photograph Archives**
2. **Which Negative Was Printed? Learning Photographic Provenance at Archive Scale with Measured Discovery Bias**

**方法为先（最"AAAI 正统"）**
3. **Nested Partition Estimation for Photographic Provenance: Structured Learning and Conformal Decisions over Candidate Graphs**
4. **NCR-Match: Recovering Exposure and Scene Structure Between a Photographic Archive and Its Printed Reproductions**

**贡献概念为先**
5. **Discovery Bias by Definition: Auditable Candidate Generation and Risk-Controlled Matching for Archival Image Search**
6. **Provenance as Search: Semi-Supervised Partition Recovery with Error Guarantees for Historical Photo Reproductions**

**人文叙事为先**
7. **Retracing the Printed Photograph: Learning to Match Wartime Press Images to Their Source Negatives**

推荐：**1**（"Same Negative or Same Scene?" 直接点出了三分类标签这一方法核心，也是审稿人最容易记住的一句）；求稳则选 **3**。

---

## Abstract (≈215 words)

Which archival negative does a printed photograph reproduce? For historians, answering this across a large archive is foundational provenance work — and hard: reproductions are cropped, halftone-screened, and retouched, while archives are dense with near-duplicate exposures shot moments apart at the same scene. We formulate reproduction matching as estimation of a nested latent partition — images group into exposures, exposures into scenes — over a corpus of archival photographs and magazine reproductions, observed only through a fixed candidate-generation operator and per-pair geometric evidence. The formulation makes discovery bias definitional rather than anecdotal: end-to-end recall factorizes into funnel coverage, estimated by stratified audits of rejected candidates, and decision recall, measured against expert adjudication. We instantiate this on 42,773 photographs from the North China Railway Archive (華北交通アーカイブ) and 2,856 reproductions from the magazine *Hokushi* (北支, "North China"; 1939–1942). A frozen four-stage pipeline (DINO retrieval, ASpanFormer correspondence, VGGT pose signals, hand-tuned thresholds) supplies candidates, evidence, and a strong hand-tuned baseline (F1 = 0.941 on a frozen validation shard). Replacing its decision rule, a 1.5M-parameter graph network with exposure and scene heads, trained on 326 expert-confirmed matches under partition-consistency constraints, reaches F1 = 0.94 on a family-disjoint held-out shard with frozen weights and no retraining — relational message passing contributing +2 F1 over a pair-MLP ablation — [vs. the hand rule on the identical graph: 待 B4 同图重算]; a conformal accept/review/reject layer bounds the false-rejection rate at [α] while cutting expert review load by [XX]%. Zero-shot transfer to a disjoint corpus — the Sha Fei archive held at the Harvard-Yenching Library paired with *Jinchaji Pictorial* reproductions — yields [XX].

---

## 中文对照（仅供理解，不用于投稿）

一张印刷出来的照片，究竟翻印自档案里的哪一张底片？对历史学家而言，在大规模档案中回答这个问题是最基础的考据工作——但它很难：印刷品经过裁切、网点加网和修版，而档案里到处是同一场景中相隔片刻拍下的近似重复曝光。我们把"翻印匹配"形式化为一个嵌套隐分区的估计问题——图像归入曝光（底片），曝光归入场景——定义在档案原照与杂志翻印件构成的语料上，且只能透过一个固定的候选生成算子和每对候选的几何证据去观察。这一形式化把"发现偏差"从轶事变成定义：端到端查全率被拆解为漏斗覆盖率（通过对被淘汰候选的分层抽样审计来估计）与决策查全率（对照专家裁定来测量）。我们在华北交通档案（華北交通アーカイブ / the North China Railway Archive）的 42,773 张照片与《北支》（Hokushi，1939–1942）杂志的 2,856 张翻印件上实例化该框架。一条冻结的四阶段管线（DINO 检索、ASpanFormer 对应点、VGGT 位姿信号、手工阈值）提供候选、证据与一个很强的手工基线（冻结验证批上 F1 = 0.941）。替换其决策规则的，是一个约两百万参数、带曝光头与场景头、在 577 个专家确认配对上以分区一致性约束训练的小型图网络，其结果为 [XX]；其上的保形"接受/复核/拒绝"决策层将误拒率控制在 [α] 以内，同时将专家复核量减少 [XX]%。在一个完全不相交的语料（哈佛燕京图书馆藏沙飞档案 + 《晋察冀画报》翻印件）上做零样本迁移，结果为 [XX]。

---

## Abstract v2 — 人文倾斜版（首尾改写，中段与 v1 一致）

When an occupation-era magazine printed a photograph of wartime North China, which negative in the archive did it come from? Historians answer this question one image at a time; at the scale of the North China Railway Archive (華北交通アーカイブ) — 42,773 photographs, set against 2,856 reproductions in the magazine *Hokushi* (北支, 1939–1942) — it becomes a search problem no reading room can hold. We formulate reproduction matching as estimation of a nested latent partition — images group into exposures, exposures into scenes — observed only through a fixed candidate-generation operator and per-pair geometric evidence. The formulation makes discovery bias definitional rather than anecdotal: end-to-end recall factorizes into funnel coverage, estimated by stratified audits of rejected candidates, and decision recall, measured against expert adjudication. A frozen four-stage pipeline (DINOv3 retrieval, ASpanFormer correspondence, VGGT pose signals, hand-tuned thresholds) supplies candidates, evidence, and a strong hand-tuned baseline; replacing its decision rule, a 1.5M-parameter graph network with exposure and scene heads, trained on 326 expert-confirmed matches under partition-consistency constraints, reaches F1 = 0.94 on a family-disjoint held-out shard with frozen weights and no retraining [vs. the hand rule on the identical graph: 待 B4 同图重算]. A conformal accept/review/reject layer bounds the false-rejection rate at [α], so expert attention concentrates where the expert eye is genuinely needed. Zero-shot transfer to a disjoint corpus — the Sha Fei archive at the Harvard-Yenching Library paired with *Jinchaji Pictorial* reproductions — yields [XX], pointing to an instrument historians can carry to other archives: label-efficient, auditable, and with error rates they set rather than inherit.

**v2 中文对照**：当占领时期的杂志刊印了一张华北战时照片，它翻印自档案中的哪一张底片？历史学家向来一张一张地回答这个问题；而在华北交通档案的规模上——42,773 张照片，对照《北支》杂志（1939–1942）的 2,856 张翻印件——它成了任何阅览室都容纳不下的搜索问题。（中段与 v1 相同：嵌套分组的形式化、发现偏差的定义化、冻结管线作基线。）……保形"接受/复核/拒绝"层将误拒率控制在 [α] 以内，使专家的时间集中在真正需要专家眼力的地方。在完全不相交的语料（哈佛燕京图书馆藏沙飞档案 + 《晋察冀画报》翻印件）上的零样本迁移结果为 [XX]——这指向一件历史学家可以带往其他档案馆的工具：标注需求低、全程可审计、错误率由使用者设定而非被动接受。

**v1 与 v2 的区别**：只有首句（人文之问开场、数据集提前到第一段）和尾句（落在"给历史学家的工具"）不同，中段的方法与数字完全一致；v2 略去了"+2 F1 消息传递归因"这半句技术细节。投稿时二选一。

---

## 占位符清单（投全文前需填）

| 占位符 | 含义 | 由哪个实验产生 |
|---|---|---|
| ~~[XX]（图网络结果）~~ **已填**：F1 0.94（Shard 2 冻结跨批，mp3 配置，negnet_tier0_report 2026-07-21） | — | 已完成 |
| [待 B4 同图重算] | 手工规则在与 NEG-Net 完全相同的 479 条边图上的 F1（现有 0.941 是旧漏斗分母，不能直接对比引用） | Jordan 用 pose_scoring 对存量信号重跑，几分钟 |
| [α]（误拒率上界） | conformal 决策层设定的漏检率上限（如 0.05） | 决策层校准，需团队先定 α |
| [XX]%（复核量减少） | 相比"全部人工看"或手工基线，送人工复核的比例下降多少 | conformal 层的 review-budget 曲线 |
| [XX]（零样本迁移） | JOCCH 数据集上的一次性结果 | 全部冻结后的最后一跑 |

## 数字出处（都可在 repo 复核）

| 数字 | 出处 |
|---|---|
| 42,773 / 2,856 | `ablation/retrieval_ablation_colab.ipynb` 运行输出："Found 42773 source images and 2856 target images" |
| F1 0.941（P 0.902 / R 0.984，Shard 2 冻结验证） | `REPRODUCTION.md` Expected results；`pose_scoring.py` 验收测试常量 |
| 577 个专家确认正例 | Shard 1 (313 TP + 12 FN) + Shard 2 (248 TP + 4 FN) = 577；与德乾 note Remark 2 一致 |
| ~2M 参数、曝光/场景双头 | 德乾 note §5.2（negnet.py 契约） |
| 四阶段、top-10、关键点≥50、0.65/2.13 | `README.md`、`REPRODUCTION.md`、`geometry_filter.py --breakpoint-value 50` |

## 两点提醒

1. **数据集名称已全部确认**：主档案 = the North China Railway Archive（華北交通アーカイブ，CODH/ROIS 数字化，https://codh.rois.ac.jp/north-china-railway/ ）；杂志 = 《北支》Hokushi（"North China"，1939–1942；repo 文件名中的 "North China Magazine" 即指它）；JOCCH 迁移集 = 沙飞档案（哈佛燕京图书馆藏）+ 《晋察冀画报》翻印件。主语料（日占方档案）与迁移语料（中共边区档案）来源完全不同，"disjoint"（不相交）的说法成立且有力。
2. **AAAI 摘要阶段允许后续修改**，重点是让 Area Chair 能正确分配审稿人：v1 写法会把论文送进 "ML + applications / structured prediction / vision" 的池子；下方的 v2（人文倾斜版）则更偏 "AI for social good / computational humanities" 的池子。二选一提交。
