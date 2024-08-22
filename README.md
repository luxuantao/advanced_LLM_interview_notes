# advanced_LLM_interview_notes
收集了一些进阶的大模型面经和相关知识，持续更新，来源均注明出处

## RLHF
https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/M.%E4%BA%BA%E7%B1%BB%E5%81%8F%E5%A5%BD%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83.md
### PPO
https://zhuanlan.zhihu.com/p/677607581
### DPO
https://zhuanlan.zhihu.com/p/642569664
### KTO
https://zhuanlan.zhihu.com/p/695992165

对正负样本进行了加权；DPO里面是使用正负样本的reward差值进行sigmoid映射，但是KTO里面使用reward模型与KL散度之间的差异
### SimPO
https://www.thepaper.cn/newsDetail_forward_27513961

采用了与生成指标直接对齐的隐式奖励形式，从而消除了对参考模型的需求。此外，其还引入了一个目标奖励差额 γ 来分离获胜和失败响应


## Advanced RAG
### 分层索引检索
利用文档摘要创建多层索引，优先检索与查询最相关的摘要部分，再深入到详细文档，提高检索效率
### query改写
### 使用假设文档嵌入修正查询与文档的非对称性 (HyDE)
在检索前，生成一个与用户查询相关的假设文档，并使用这个文档的嵌入来替代用户的查询进行语义搜索
### 相关文档重排或者直接用LLM打分
### 压缩搜索结果
### 加入反思

## LLM量化
https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/G.%E9%87%8F%E5%8C%96.md

按照量化发生的步骤区分，可以划分为PTQ（训练后量化，或离线量化）和QAT（训练感知型量化，或在线量化）。
PTQ量化可以分为data-free和calibration两种，前者不使用数据集进行校准直接计算量化因子，后者会根据少量真实数据进行统计分析并对量化因子进行额外校准，但耗费的时间更长。
QAT量化会先在待量化的算子上增加一个伪量化结构，并在训练时模拟量化过程并实时更新计算量化因子（类似反向传播过程）及原始权重。

按照量化方法可以划分为线性量化、非线性量化（如对数量化）等多种方式，目前较为常用的是线性量化。
其中线性量化又可以按照对称性划分为对称量化和非对称量化，非对称量化为了解决weight分布不均匀问题，其在公式中增加了zero_point项：qweight=round(weight/scale + zero_point)，使稠密数据部分可以得到更宽泛的数值范围。

按照量化粒度划分可以分为**逐层量化（每层使用一套量化因子）、逐组量化（在每层中按照group使用一套量化因子）、逐通道量化（按channel划分量化因子）**等几种方式。

### LLM.int8()
从输入的隐含状态中，按列提取异常值 (离群特征，即大于某个阈值的值)

对离群特征进行 FP16 矩阵运算，对非离群特征进行量化，做 INT8 矩阵运算

反量化非离群值的矩阵乘结果，并与离群值矩阵乘结果相加，获得最终的 FP16 结果

拖慢了推理速度

### GPTQ
采用 W4A16 的混合量化方案，其中模型权重被量化为 int4 数值类型，而激活值则保留在 float16，是一种仅权重量化方法

在推理阶段，模型权重被动态地反量化回 float16 并在该数值类型下进行实际的运算

GPTQ还是从单层量化的角度考虑，希望找到一个量化过的权重，使的新的权重和老的权重之间输出的结果差别最小
GPTQ 将权重分组（如：128列为一组）为多个子矩阵（block）。对某个 block 内的所有参数逐个量化，每个参数量化后，需要适当调整这个 block 内其他未量化的参数，以弥补量化造成的精度损失。因此，GPTQ 量化需要准备校准数据集。

### SmoothQuant
训练后量化 (PTQ) 方法， W8A8 量化。
由于权重很容易量化，而激活则较难量化，因此，SmoothQuant 引入平滑因子s来平滑激活异常值，通过数学上等效的变换将量化难度从激活转移到权重上。
AWQ
仅权重量化方法。通过保护更“重要”的权重不进行量化，从而在不进行训练的情况下提高准确率。只保留 0.1%-1% 的较大激活对应权重通道为 FP16 。

## LLM+tools
### Toolformer
通过in-context learning的方式从训练数据中采样出包含工具调用的数据

如果提供API和对应执行结果后生成答案的LM loss比啥也不提供的LM loss小一个阈值，那就认为这个API是有帮助的

### ToolLLM
从Rapidapi收集了一批API，当API响应结果长度超过2048个token，研究人员将tool的相关信息以及3个压缩样例融入prompt中，通过ChatGPT对响应结果进行压缩，如果压缩后的结果长度依旧超过限制，那就保留前面的2048个token。

构建了一个(Instruction, APIs)的数据集，一个(Instruction, solution)数据集。用(Instruction, APIs)数据集训练的API Retriever，用(Instruction, solution)数据集微调的LLaMa，LLaMa在根据Instruction寻找相应的API的时候调用BERT来进行API Retrieval

## LLama3.1中的scaling law
https://www.bilibili.com/video/BV1Q4421Z7Tj/?spm_id_from=333.999.top_right_bar_window_dynamic.content.click&vd_source=e78a159ff471c8c0cb1842c33e4b7879

固定算力消耗（时间*卡数）下，计算不同模型大小在不同训练数据量上（此消彼长）的validation loss，找到最优点，即最佳的模型大小和训练数据量

对不同的算力消耗都按上述步骤做一遍，找到最优点

根据这些最优点可以知道，针对不同算力消耗所需要的最佳训练数据量

还有一种scaling law是针对benchmark的，计算Negative Log-Likelihood Loss和acc之间的关系

## LLM长度外推
https://mp.weixin.qq.com/s/54YdSdB1uX-i7mt7kasFIQ
### 线性插值
对于向量的所有分组不加区分地缩小旋转弧度，降低旋转速度（进一步体现为对其正弦函数进行拉伸），会导致模型的高频信息缺失，从而影响模型的性能
### NTK-Aware Interpolation
高频信息对于神经网络非常重要，保留高频信息
高频分量旋转速度降幅低，低频分量旋转速度降幅高，即越靠后的分组旋转弧度缩小的倍数越大
在高频部分进行外推，低频部分进行内插。
### NTK-by-parts Interpolation
基于NTK-Aware Interpolation进行优化，不改变高频部分，仅缩小低频部分的旋转弧度。也就是不改变靠前分组的旋转弧度，仅减小靠后分组的旋转弧度
### Dynamic NTK Interpolation
当超出训练长度时，上述插值方法都比原模型直接外推的效果更好，但是它们都有一个共同的缺点，在训练长度内，推理表现都比原模型差

推理长度小于等于训练长度时，不进行插值

推理长度大于训练长度时，每一步都通过NTK-Aware Interpolation动态放大base，每一次生成都会重新调整旋转弧度，然后再进行下一次生成

## 多模态
https://github.com/wdndev/mllm_interview_note/blob/main/02.mllm%E8%AE%BA%E6%96%87/0.%E4%BB%8E%E8%A7%86%E8%A7%89%E8%A1%A8%E5%BE%81%E5%88%B0%E5%A4%9A%E6%A8%A1%E6%80%81%E5%A4%A7%E6%A8%A1%E5%9E%8B.md

## LLM幻觉
https://www.zhihu.com/people/swtheking/posts
### 分类
1.1 事实性问题（Factuality）
模型回答与事实不一致或在真实世界无法考证

1.2 忠诚度问题（Faithfulness）
模型回答没有遵从指令或者模型回答和上下文内容存在不一致

1.3 自我矛盾（self-Contradiction）
模型回答内部问题存在逻辑矛盾，比如COT多步推理之间存在矛盾

### 来源
数据源：错误训练数据，重复偏差，社会偏见，领域知识匮乏，知识过时未更新

训练：训练时teacher-force策略和推理策略的不一致性；指令微调样本的知识部分超出预训练知识的范畴，导致微调过程错误引导模型回答本身压缩知识范围之外的问题，从而加重了模型幻觉

推理：注意力机制的长程衰减；解码过程的错误累计

### 解决办法
高质量低事实错误的预训练数据集构建，通过模型、规则筛选高质量web数据源

降低重复偏见：使用SimHash、SemDeDup等消重技术对预训练数据进行消重

降低社会偏见

知识编辑

RAG

反思，后处理：利用模型自我修正能力，先让模型生成答案，再使用prompt让模型对答案进行多角度的校验提问，并回答这些提问，最后基于以上回答修正初始答案

## 百面LLM中比较好的问题
https://www.zhihu.com/people/swtheking/posts

### ROPE是低频部分保持远程衰减，还是高频部分保持远程衰减？
低频

### 为什么现在都是decoder-only架构
众所周知，Attention矩阵一般是由一个低秩分解的矩阵加softmax而来，具体来说是一个n×d的矩阵与d×n 的矩阵相乘后再加softmax（n≫d），这种形式的Attention的矩阵因为低秩问题而带来表达能力的下降。而Decoder-only架构的Attention矩阵是一个下三角阵，注意三角阵的行列式等于它对角线元素之积，由于softmax的存在，对角线必然都是正数，所以它的行列式必然是正数，即Decoder-only架构的Attention矩阵一定是满秩的！满秩意味着理论上有更强的表达能力，也就是说，Decoder-only架构的Attention矩阵在理论上具有更强的表达能力，改为双向注意力反而会变得不足。

decoder-only的预训练目标和下游任务一致。

下三角或上三角mask更能够把位置编码的信息处理得更好？带来了位置识别上的优势，它打破了transformer的置换不变性，直接引入了从左往右的序，所以甚至不加位置编码都行

### 为什么需要RLHF？SFT不够吗？
数据更好搞

除了正确答案之外还要知道错误答案

RLHF中的数据更多的包含模型输出的安全性、伦理性、政治以及用户的指令遵循

RLHF的泛化性能比SFT好，但多样性会有所降低。

### DPO是on-policy还是off-policy
DPO是一个off-policy的算法，因为训练DPO的pair数据不一定来自ref policy或者sft policy。优势是不需要对模型进行采样，然后标注，直接可以拿已有的数据集进行训练，这样的情况下包括采样的成本和标注的成本都可以节约。劣势是效果很难保证，尤其是你的模型本身能力和发布的pair数据不匹配的时候

### PPO是on-policy还是off-policy
近似on-policy

根据off-policy的定义，采样的网络和要优化的网络不是一个网络，那么对于PPO来说，使用一批数据从更新actor的第二个epoch开始，数据虽然都是旧的actor采样得到的，但是我们并没有直接使用这批数据去更新我们的新的actor，而是使用imporance sampling先将数据分布不同导致的误差进行了修正。那么这个importance sampling的目的就是让这两者数据分布之间的差异尽可能的缩小，那么就可以近似理解成做了importance sampling之后的数据就是我们的更新（这里的更新指的是多个epoch更新的中间过程）后的actor采样得来的，这样就可以理解成我们要优化得actor和采样得actor是同一个actor，那么他就是on-policy的。

### 可以跳过sft阶段直接进行rlhf吗
现阶段来看是不太可能的。模型如果纯进行RL的话，搜索空间过于庞大，消耗资源较多，利用sft首先做模仿学习缩小搜索空间，再利用RLHF进行进一步对齐是必要的。

### 同等MOE模型的loss能下降到和同等规模Dense模型的水准吗？
不能，因为MOE在训练中每个token forward和backward的实际的激活参数是远少于同等规模的Dense 模型的（Btw，尽管Dense模型训练完也是个偏向sparse的模型，也就是有少量神经元被激活，但是在训练中，Dense模型是可以自由选择激活哪部分神经元的。而Sparse Moe，通过训练路由来控制哪个token激活哪部分的expert，本质差距还蛮远的）那么从DeepseekV2-MOE-236B来看，激活21B，总参 236B，等效一个 90B 的Dense，从Deepseek-Coder-MOE-16B，激活2.4B，总参数16B，等效于一个7B模型。（等效计算是和激活参数，总参数都挂钩的函数计算出来的。）

### RLHF的performance上界是什么
RLHF的performance上界就是rm模型的泛化上界
