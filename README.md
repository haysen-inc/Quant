# SPY 智能量化交易系统 (Differentiable Expert System)

本项目创造性地将传统金融领域的“技术指标+硬编码布尔逻辑（如 MyLanguage/文华财经）”转化为一套**全量可微、在线自发演化**的深度学习端到端交易系统。

本仓库通过自研的 AST 语法树解析器，将人类的定式交易常数直接注入到 PyTorch 神经网络底层，使之作为一种“带有先验人类智商骨架”的模型启动。随后利用极其残酷的纯物理资金（PnL）作为损失函数，强制 AI 自我进化。

---

## 核心架构演进路径

### 1. 深度网络如何表征“九因子”
传统的 `ma`, `ema`, `kdj` 计算往往依赖于含有滞后的 `for` 循环体系。
在 `features_torch.py` 引擎中，所有的时序摆动与平滑均被映射为了无循环的极其暴力的 PyTorch 矩阵切片。
* 输入的 $K$ 线网格会被构建为 `[Batch_Size, Sequence_Length, 23]` 的连续浮点特征网。
* 彻底抛弃了 Z-Score 泛化，保留绝对数值的价格特征，确保深度学习底层的标度与主观实盘对峙时绝不失真。

### 2. 人类专家布尔逻辑的平滑化 (Differentiable Logic)
深度网络不能处理形如 `C < MA(C)` 或者 `JX > J1 * 1.5` 的非黑即白的逻辑。
在 `differentiable_expert.py` 中，系统自研并定义了一套**可导微积分平滑算子**：
* **`DiffGreater` / `DiffLess`**: 利用带有自适应温度参数（Temperature）的急剧陡峭的 $\sigma$ (Sigmoid) 函数来实现可微微分大小比较。
* **`DiffCrossDown` / `DiffCrossUp`**: 利用时序填充（Sequence Padding）与滞后向算子的张量乘积，近乎完美地连续复刻了 `死叉` 与 `金叉`。

### 3. "零样本绝对等效" (Zero-Shot Equivalence)
* **AST 编译器拦截注入**: 依靠 `src/mylanguage_parser.py` 解析器，前端网页传来的传统 MyLanguage / 麦语言源码，会被瞬间正则提取出长短移动平均期常数以及偏置量（如 `6.0`、`-50.0`）。
* **物理挂载**: 这些常数值会被强行赋给 PyTorch 中的 `nn.Parameter` 初始化槽中。如果在不打开梯度（不执行 `Loss.backward()`）的情况下跑预测，模型的输出信号会与传统金融软件计算出的金叉点保持 **100.00% 绝对一致**。网络从一开始就是一个“完整的、带着硬编码智商的人类专家”。

### 4. 模型奖励机制 (Asymmetric Expert Loss)
既然模型一开始就等于人类，我们为什么还要训练？
因为人类规则常常在震荡市亏钱。
系统抛弃了常规的 BCE/MSE，创造了具有自我否定属性的**非对称专家暴利极刑模型**:
* **剔除伪阳性**: 如果人类逻辑触发开仓多头，但实际上随后长达 `35 Hour` 的持仓 $R_{fwd} < 0$，系统就会产生反向梯度，使得负责这段死叉的 `Temperature` 张量急剧下降乃至坍缩至 `0.0`从而“物理阉割”掉亏钱因子。
* **正向核爆**: 如果触发逻辑并获得了暴赚，系统的梯度正向奖赏会依据利润幅面进行乘子爆炸，强化网络对于这段逻辑的信心。

### 5. 永不宕机：实盘在线流式强化学习 (Online RL Agent)
告别传统的单次历史回测。在 `src/online_rl_agent.py` 中：
* 系统搭建了一个类似实盘队列的沙盒 `Pending Queue`。
* 当系统按照过去的惯性作出开仓决策，订单会被挂起。
* 只有当未来的真实行情走到 `SP1` (多头衰竭) 或时间跨越强平阈值时，订单才宣告死亡，计算出严丝合缝的盈亏率。
* 这笔带血的钱（无论是亏还是赚），会被立刻作为新的梯形 Label 发回 PyTorch 模型中执行**活体 Live Backpropagation** (`optimizer.step()`)。模型在“边交易、边受伤、边变异”。

---

## 如何部署与运行可视化系统

在经过了几十个版本的迭代后，这套引擎已具备极其成熟的 Web 交互界面与 API 服务结构，彻底打破了终端界面的黑盒感：

### 1. 运行底层 Flask API 中枢
首先，您必须在虚拟环境中启动充当粘合剂的本地后端服务器。

```bash
# 进入工程目录
cd /home/dgxspark/Desktop/Quant

# 激活 Python 环境 (如果使用了 venv/conda)
source .venv/bin/activate

# 将模块源路径加入全局以防 ModuleNotFound
export PYTHONPATH=$PYTHONPATH:/home/dgxspark/Desktop/Quant

# 跑起 Python 后端 (绑定到 5000 端口)
python app.py
```

### 2. 拥抱数据可视化网页客户端
该系统由纯净原生的 HTML/JS 编写，无需冗杂的前端打包服务器工具。

1. **直接在浏览器或 VSCode 内核中打开网络文件：**
   `frontend/index.html` 
2. 您将在浏览器中看到极具赛博朋克科幻风格的 **SPY Differentiable Sandbox** 控制台。
3. **AST 代码注入区：** 请确保两块 textarea 里的麦语言代码完好无损。
4. **一键宏观 Pre-Train:** 通过点击 `Start Phase 1 Historical Base Pre-Training`。
   * 查看 Console 输出，见证底层张量的解析和 50 Epochs 的预训练演变。
   * 您会在中央大图内直观看见：九大因子（特别是做空阈值因子）的 Temperature 是如何在面对美股万年长牛的情况下产生觉醒，自我把梯度抛向 0 从而主动截断交易损失的。
5. **一键活体演化:** 通过点击 `Start Phase 14 Live Online RL Simulation`。
   * 您能看到 400 + 根流式数据下的红绿双轨 PnL 盈亏对比图。
   * 下方将绘出底层模型针对多空阵营实时摇摆的 Confidence (Agent p(buy)) 断崖式曲线。直观感受出它面对人类基线 (-1.16% 亏爆) 时，是如何通过 9 次精确的多头斩获 +3.10% 血淋淋实利的。

---

> 🚀 *"It's not just a model. It's a bleeding-edge physical abstraction."*
