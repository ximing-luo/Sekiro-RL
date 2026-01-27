"""
模块用途：调度与定时相关的辅助函数（如学习率或任务计划）。

包含：
- 函数与类：常量/线性/分段调度

边界：
- 负责：纯工具函数
- 不负责：训练主体与环境交互
"""
'''
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
'''

class Schedule(object):
    """职责：调度器接口，约定提供 `value(t)`。

    函数用来干什么：为不同类型的调度器定义统一接口，外部只需调用 `value(t)` 获取在时刻 `t` 的值。
    参数说明：
    - （无构造参数）
    - 输入：`t`（int|float），时间步或时刻索引
    - 输出：返回该时刻的调度值（float）

    实现步骤：
    1) 抽象类定义；
    2) 在子类中实现 `value(t)` 返回对应时刻的值；
    """

    def value(self, t):
        """职责：返回时刻 `t` 的调度值。"""
        raise NotImplementedError()

class ConstantSchedule(object):
    """职责：提供恒定不变的调度值。

    类用来干什么：当需要一个固定值（如固定学习率或固定概率）时使用。
    """

    def __init__(self, value):
        """职责：初始化恒定值。

        实现步骤：
        1) 接收外部传入的常量值；
        2) 写入内部状态 `_v`；

        函数用来干什么：设置该调度器在任意时刻返回的固定值。
        参数说明：
        - 输入：`value`（float），固定返回值
        - 输出：无（初始化对象）
        """
        # 步骤 1：接收常量值
        # 步骤 2：写入内部状态
        self._v = value

    def value(self, t):
        """职责：返回恒定值，忽略 `t`。

        实现步骤：
        1) 直接返回 `_v`；

        函数用来干什么：获取在时刻 `t` 的调度值（恒定值）。
        参数说明：
        - 输入：`t`（int|float），时间步或时刻索引（未使用）
        - 输出：`_v`（float），固定返回值
        """
        # 步骤 1：返回恒定值
        return self._v

def linear_interpolation(l, r, alpha):
    """职责：在左右端 `l/r` 间做线性插值，权重为 `alpha`。

    实现步骤：
    1) 计算差值 `(r - l)`；
    2) 乘以比例 `alpha`；
    3) 加到左端点 `l`；

    函数用来干什么：作为 `PiecewiseSchedule` 的默认插值函数，计算端点之间的线性变化值。
    参数说明：
    - 输入：`l`（float，左端值）、`r`（float，右端值）、`alpha`（float，[0,1] 比例）
    - 输出：插值结果（float）
    """
    # 步骤 1：计算差值
    # 步骤 2：乘以比例
    # 步骤 3：加到左端点
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    """职责：根据给定时间-值端点，分段返回插值或端点值。

    类用来干什么：定义一组 `(time, value)` 端点，按时间在相邻端点之间进行插值或在端点处返回固定值。
    """

    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """职责：初始化分段调度的端点、插值函数与区间外返回值。

        实现步骤：
        1) 提取端点的时间索引并断言有序；
        2) 存储插值函数与外部返回值；
        3) 存储端点列表；

        函数用来干什么：构造一个分段调度器。
        参数说明：
        - 输入：
          - `endpoints`（list[tuple[int,float]]）：按时间递增排列的 `(time, value)` 列表
          - `interpolation`（callable）：插值函数，默认线性插值
          - `outside_value`（float|None）：当 `t` 不在任一区间时返回的值，为 None 则抛出断言
        - 输出：无（初始化对象）
        """
        # 步骤 1：提取时间索引并保证有序
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        # 步骤 2：保存插值函数与外部返回值
        self._interpolation = interpolation
        self._outside_value = outside_value
        # 步骤 3：保存端点列表
        self._endpoints      = endpoints

    def value(self, t):
        """职责：返回时刻 `t` 的分段值或两端线性插值，超界返回 `outside_value`。

        函数用来干什么：查询时刻 `t` 对应的调度值；若在两个端点之间则按插值返回。
        参数说明：
        - 输入：`t`（int|float），时间步或时刻索引
        - 输出：调度值（float）
        """
        # 步骤 1：遍历相邻端点区间，找到 t 所在区间
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                # 步骤 2：计算区间内比例 alpha
                alpha = float(t - l_t) / (r_t - l_t)
                # 步骤 3：按插值函数返回值
                return self._interpolation(l, r, alpha)

        # 步骤 4：超出所有区间时返回 outside_value（若为空则断言）
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    """职责：按时间线性插值从初始值到最终值。

    类用来干什么：为探索概率、学习率等提供线性退火/增长过程。
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """职责：初始化线性调度的时间长度与端点值。

        实现步骤：
        1) 保存线性插值总步数；
        2) 保存最终值与初始值；

        函数用来干什么：定义从 `initial_p` 到 `final_p` 的线性变化过程。
        参数说明：
        - 输入：
          - `schedule_timesteps`（int）：线性插值的总步数
          - `final_p`（float）：最终值
          - `initial_p`（float）：初始值，默认 1.0
        - 输出：无（初始化对象）
        """
        # 步骤 1：保存总步数
        self.schedule_timesteps = schedule_timesteps
        # 步骤 2：保存端点值
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """职责：返回时刻 `t` 的线性插值值。

        函数用来干什么：根据进度比例 `fraction` 返回线性插值结果。
        参数说明：
        - 输入：`t`（int|float），时间步或时刻索引
        - 输出：插值值（float）
        """
        # 步骤 1：计算进度比例 fraction（上限 1.0）
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        # 步骤 2：按线性插值公式返回值
        return self.initial_p + fraction * (self.final_p - self.initial_p)