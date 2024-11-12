1. 场景1：旧的Bid价格能在新的Bid价格序列中找到
比如旧的Bid 2价格等于新的Bid 1价格，这表示该价位在新旧快照间保持不变，可以计算该档位的delta volume

2. 场景2：旧的Bid价格比新的Bid 1价格更高
这意味着旧快照中的Bid价格已被完全成交。可以确定这档位发生了交易，并记录成交量为旧快照上的volume。同样，如果旧的Ask价格比新的Ask 1价格更低，也视为该Ask档位订单已被完全成交。

3. 场景3：旧的Bid价格低于新的Bid 5价格
如果旧的Bid（或Ask）价格低于新的Bid 5价格（或Ask 5价格），则无法确定是否发生了交易，因此假设delta volume为0。

计算出所有Bid和Ask档位的delta volume，并将其汇总得到confirmed total volume和confirmed total turnover：

$$
\text{totalVolumeConfirmed} = \sum \text{delta volume on 10 positions}\\

\text{totalTurnoverConfirmed} = \sum \text{price} \times \text{delta volume on 10 positions}
$$

处理未确认交易量和金额

对比 $\text{totalVolumeConfirmed}$ 和真实的成交量$\text{Volume}$：

1. 如果 $\text{totalVolumeConfirmed} < \text{Volume}$：说明部分交易没有在前五档位上展示，假设这些交易发生在新的Bid 1或Ask 1上。
- 计算剩余交易量和成交金额：

$\text{VolumeLeft} = \text{Volume} - \text{totalVolumeConfirmed}$


$\text{TurnoverLeft} = \text{Turnover} - \text{totalTurnoverConfirmed}$

- 分配剩余交易量至新的Bid 1和Ask 1上：
- 估计Ask 1的交易量（假设Bid和Ask 1都有交易）：

$\text{guessVolumeAtAsk1} = \frac{\text{TurnoverLeft} - \text{Bid 1} \times \text{VolumeLeft}}{\text{Ask 1} - \text{Bid 1}}$

$\text{guessVolumeAtBid1} = \text{VolumeLeft} - \text{guessVolumeAtAsk1}$

检查计算结果：

- 若$\text{guessVolumeAtBid1} < 0$：表明交易集中于Ask一侧，假设所有交易发生在Ask 1上，重新计算：

$\text{guessAskPrice} = \frac{\text{TurnoverLeft}}{\text{VolumeLeft}}$

- 若 $\text{guessVolumeAtAsk1} < 0$：则集中于Bid一侧，假设所有交易发生在Bid 1上，重新计算：

$\text{guessBidPrice} = \frac{\text{TurnoverLeft}}{\text{VolumeLeft}}$

- 如果 $\text{totalVolumeConfirmed} > \text{Volume}$：则可能因为订单取消导致。类似分配剩余交易量的步骤，将负值分配至旧的Bid 1和Ask 1位置



一些可能的改进：

- **分配权重模型**：将剩余交易量按距离价格的相对权重分配到多个档位上。例如可以使用指数衰减权重，使得距离成交价格较近的档位获得较多的交易量分配。
- **概率成交模型**：构建一个简单的成交概率模型，使用历史数据分析在类似情况下的成交概率。若Bid价格低于新的Bid 5，但成交量未达到预期，可以设定一个较低概率的成交，以更准确反映市场行为。
- **成交量补充模型**：对不确定情况下的成交量，通过随机扰动来补充，使得delta volume在边界情况下不完全为0，这样能避免在不确定情况下可能存在的成交低估问题。