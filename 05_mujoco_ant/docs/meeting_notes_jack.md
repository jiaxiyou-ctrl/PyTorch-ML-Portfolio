# Meeting with Jack Jones — 与 Jack Jones 见面笔记
**Date 日期:** Friday 25 April 2026, 14:30
**Who 对象:** Jack Jones — Lecturer in High Performance Computing, University of Bristol
**Purpose 目的:** Discuss GPU cluster access for RL training project
（讨论获取 GPU 集群资源来训练强化学习项目）

---

## 1. Opening — 开场（~2 min）

**Say 说：**
> Hi Jack, thank you so much for taking the time to meet with me.
> I really appreciate your help.

（你好 Jack，非常感谢你抽时间见我，真的很感谢你的帮助。）

**If he asks "Tell me about yourself" 如果他问你的背景：**
> I'm Jiaxi, a taught postgraduate student in Computer Science.
> My supervisor is Ruzanna Chitchyan. I'm very interested in
> Embodied AI and I'm currently building projects to prepare
> for PhD applications in this area.

（我是嘉希，计算机科学授课型硕士生，导师是 Ruzanna Chitchyan。
我对 Embodied AI 很感兴趣，目前在做项目为申请这个方向的博士做准备。）

---

## 2. Project Introduction — 项目介绍（~5 min）

**Say 说：**
> My project trains a simulated robot — the MuJoCo Ant — to
> perform different locomotion tasks using reinforcement learning.
> I built everything from scratch in PyTorch and the project has
> gone through three versions.

（我的项目用强化学习训练一个模拟机器人——MuJoCo Ant——执行不同的运动任务。
所有代码都是我用 PyTorch 从零写的，项目经历了三个版本的迭代。）

**Then explain the three versions 然后解释三个版本：**

> **Version 1** uses the robot's internal state — joint angles and
> velocities — as input, with a simple MLP network and PPO algorithm.

（**第一版**用机器人的内部状态——关节角度和速度——作为输入，
用简单的 MLP 网络和 PPO 算法。）

> **Version 2** learns directly from raw camera images — 84 by 84
> pixel RGB images. I added a CNN encoder to process the visual input.
> After 1 million steps on my laptop CPU, it achieved a reward of
> about 300.

（**第二版**直接从原始摄像头图像学习——84×84 的 RGB 像素图像。
我加了一个 CNN 编码器来处理视觉输入。
在我笔记本 CPU 上训练 100 万步后，奖励达到了约 300。）

> **Version 3** is the most advanced — it's a Vision-Language-Action
> architecture. The agent receives both pixel images AND natural
> language instructions like "walk forward" or "turn left." I use a
> frozen CLIP text encoder for language understanding and a multimodal
> fusion network to combine visual and language features. All 5 tasks
> converged successfully, with a best reward of 2644.5.

（**第三版**是最先进的——视觉-语言-动作架构。
智能体同时接收像素图像和自然语言指令，比如"向前走"或"左转"。
我用冻结的 CLIP 文本编码器理解语言，用多模态融合网络结合视觉和语言特征。
全部 5 个任务都成功收敛，最佳奖励达到 2644.5。）

**If he asks what the 5 tasks are 如果他问5个任务是什么：**
> The five tasks are: walk forward, walk backward, turn left,
> turn right, and stand still.

（五个任务是：向前走、向后走、左转、右转、站立不动。）

---

## 3. Hardware Requirements — 硬件需求（~3 min）

**Say 说：**
> For the hardware, here's what I need:

（硬件方面，我的需求如下：）

| Resource 资源 | Requirement 需求 | Notes 备注 |
|---|---|---|
| GPU | Single GPU, ≥ 8GB VRAM（单块 GPU，至少 8GB 显存） | V100 / A100 / RTX 3090 ideal（理想选择） |
| CPU | ≥ 4 cores（至少 4 核） | MuJoCo simulation runs on CPU（MuJoCo 仿真跑在 CPU 上） |
| RAM | ~32GB（约 32GB 内存） | CNN encoder + replay buffer need memory（CNN 编码器和回放缓冲区需要内存） |
| Storage | ~20GB（约 20GB 存储） | For checkpoints and logs（存模型检查点和日志） |
| Time per run | 12–48 hours（每次训练 12-48 小时） | 5–10M environment steps（500-1000 万步） |

**Say 说：**
> My software stack is all pip-installable — PyTorch with CUDA,
> Gymnasium, MuJoCo, and HuggingFace Transformers for the CLIP
> encoder in Version 3. No special system-level setup is needed.

（我的软件栈全部可以 pip 安装——带 CUDA 的 PyTorch、Gymnasium、MuJoCo，
还有 HuggingFace Transformers 用于第三版的 CLIP 编码器。
不需要任何特殊的系统级配置。）

> I've already prepared SLURM job scripts and made all my training
> code GPU-ready with automatic device detection and checkpoint
> saving, so I can resume training if a job gets interrupted.

（我已经准备好了 SLURM 作业脚本，所有训练代码都做了 GPU 适配，
包括自动设备检测和检查点保存，这样即使任务中断也能恢复训练。）

---

## 4. Questions for Jack — 要问 Jack 的问题（~10 min）

### Cluster Questions — 集群相关

**Q1:**
> Which cluster would you recommend for my use case — BluePebble
> or BlueCrystal?

（你推荐我用哪个集群——BluePebble 还是 BlueCrystal？）

💡 **Why this matters 为什么重要：** 不同集群有不同的 GPU 型号和排队策略，
选对集群能让你更快开始训练。

**Q2:**
> What is the maximum wall-clock time allowed per job?

（每个作业最长允许运行多少时间？）

💡 **Why 为什么问：** 如果最长只有 24 小时，你需要确保 checkpoint
机制能在时间到之前保存进度。

**Q3:**
> What CUDA versions are available on the cluster? Do I need to
> load specific modules?

（集群上有哪些 CUDA 版本？我需要加载特定的模块吗？）

💡 **Why 为什么问：** PyTorch 版本需要和 CUDA 版本匹配，
不匹配会导致 GPU 无法使用。

**Q4:**
> Can I set up a conda environment, or is there a preferred way
> to manage Python packages on the cluster?

（我可以用 conda 环境吗，还是集群上有推荐的 Python 包管理方式？）

💡 **Why 为什么问：** 有些集群不支持 conda，需要用 module 系统或 virtualenv。

**Q5:**
> Is it possible to use interactive sessions — for example with
> tmux or screen — for debugging before I submit batch jobs?

（可以使用交互式会话吗——比如 tmux 或 screen——在提交批处理任务前先调试？）

💡 **Why 为什么问：** 第一次跑代码需要先调试确保没有 bug，
不能直接盲提交 48 小时的任务。

### Technical Questions — 技术相关

**Q6:**
> Do you have any tips on parallelising MuJoCo environment rollouts
> across multiple CPU cores on the cluster?

（你对在集群上把 MuJoCo 环境采样并行化到多个 CPU 核心有什么建议？）

💡 **Why 为什么问：** 这是尊重他 HPC 专长的问题，
同时也是你项目实际面临的工程挑战。RL 训练中环境交互是瓶颈。

**Q7:**
> Are there any known issues with running MuJoCo on the cluster?
> For example, does offscreen rendering with osmesa or egl work
> out of the box?

（在集群上运行 MuJoCo 有什么已知问题吗？
比如用 osmesa 或 egl 的离屏渲染能直接用吗？）

💡 **Why 为什么问：** MuJoCo 的渲染在无显示器的服务器上经常出问题，
提前知道能省很多调试时间。

---

## 5. PhD Questions — 博士申请相关（~5 min）

**Transition 过渡句（自然引入话题）：**
> This project is really part of my preparation for PhD applications.
> I'm hoping to apply for programmes in Embodied AI, probably for
> the 2027 entry cycle. Since you're a lecturer here, I'd really
> value any advice you might have.

（这个项目其实是我准备博士申请的一部分。
我希望申请 Embodied AI 方向的博士，可能是 2027 年入学。
既然你是这里的讲师，你的建议对我来说非常宝贵。）

**Q8:**
> Are there any researchers at Bristol — or other UK universities —
> working on Embodied AI, robotics, or vision-based RL that you'd
> recommend I look into?

（Bristol 或者其他英国大学有没有做 Embodied AI、机器人或视觉 RL 的研究者，
你会推荐我去了解的？）

💡 **This is the most valuable question 这是最有价值的问题：**
他可能直接帮你引荐，这比你自己套磁有效得多。

**Q9:**
> Does Bristol have any internal PhD funding schemes or studentships
> that I should be aware of for the 2027 cycle?

（Bristol 有没有什么内部的博士资助或奖学金，是我应该了解的？）

💡 **Why 为什么问：** UK PhD 的资金来源很多样：EPSRC、CSC、校内奖学金等。
内部人告诉你的信息比你自己在网上搜的准确得多。

**Q10:**
> What do you think makes a PhD application stand out? Is it more
> important to have published papers, or is a strong project with
> reproducible code equally valued?

（你觉得什么样的博士申请会比较突出？
发表论文更重要，还是有一个代码可复现的好项目也同样受认可？）

💡 **Why 为什么问：** 你目前有项目但没有发表，了解招生者的真实看法很重要。

**Q11 (only if conversation is going well 只在聊得好的时候问):**
> In your experience, do PhD programmes in AI value candidates
> who also have strong systems or HPC skills?

（根据你的经验，AI 方向的博士项目会看重同时具备系统/HPC 技能的候选人吗？）

💡 **Why 为什么问：** 让他觉得你重视他的领域，
同时暗示你愿意在 HPC 方面也深入学习。

---

## 6. Closing — 结束（~2 min）

**Say 说：**
> This has been incredibly helpful. Thank you so much for your
> time and advice.

（这对我帮助非常大。非常感谢你的时间和建议。）

> Would it be alright if I follow up by email as my project
> progresses? I'd love to keep you updated on the results.

（我项目有进展后，可以通过邮件跟你汇报吗？我很想让你了解最新的结果。）

💡 **Why 为什么说：** 建立长期联系，让他成为你的 advocate。
定期汇报 → 他了解你的能力 → 未来潜在的推荐信。

> Also, is there anyone else you'd suggest I speak to about
> Embodied AI research?

（另外，关于 Embodied AI 研究，你还会建议我去找谁聊聊？）

💡 **The golden question 黄金问题：** 拓展人脉的最佳方式。

**Final 最后：**
> Thanks again, Jack. I really appreciate it.

（再次感谢，Jack。真的很感激。）

---

## 7. Key Numbers to Remember — 需要记住的关键数字

| Item 项目 | Number 数字 |
|---|---|
| V2 Pixel PPO reward | ~300 (at 1M steps on CPU) |
| V3 VLA best reward | 2644.5 |
| V3 tasks converged | 5 out of 5 |
| GPU VRAM needed | ≥ 8GB |
| CPU cores needed | ≥ 4 |
| RAM needed | ~32GB |
| Storage needed | ~20GB |
| Training steps planned | 5–10M |
| Estimated GPU time | 12–48 hours per run |

---

## 8. Things to Bring / Show — 带去/展示的东西

- [ ] 手机上打开 GitHub 仓库：https://github.com/jiaxiyou-ctrl/PyTorch-ML-Portfolio
- [ ] demo_videos/ 里的 Ant 行走视频（可以在手机上提前打开）
- [ ] results/training_reward_curve.png（训练曲线图）
- [ ] results/ant_walking.gif（Ant 行走动图）
- [ ] 这份笔记（存手机里或打印）
- [ ] 笔和纸/iPad（记录他的建议）

---

## 9. After Meeting — 见面后要做的事

### 24 小时内发感谢邮件：
```
Subject: Thank you for meeting today
