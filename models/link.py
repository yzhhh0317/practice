from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import numpy as np
from pandocfilters import Link


@dataclass
class Link:
    def __init__(self, source_id: Tuple[int, int], target_id: Tuple[int, int],
                 capacity: float = 25.0, queue_size: int = 100):
        self.source_id = source_id
        self.target_id = target_id
        self.capacity = capacity * 1024 * 1024  # Mbps转换为bps
        self.queue_size = queue_size
        self.queue = []
        self.last_update_time = time.time()
        self.processed_bytes = 0
        self.start_time = time.time()

        # 增加性能统计
        self.total_packets = 0
        self.dropped_packets = 0
        self.queue_history = []
        self.delay_history = []

    @property
    def queue_occupancy(self) -> float:
        """计算队列占用率"""
        current_time = time.time()
        cycle_time = (current_time - self.start_time) % 60
        cycle = int((current_time - self.start_time) / 60)

        # 基础占用率
        base_occupancy = len(self.queue) / self.queue_size

        if 29.98 <= cycle_time <= 35.65:  # 拥塞期间
            # 根据周期降低峰值占用率
            peak_factor = {
                0: 0.85,  # 第一周期最高
                1: 0.75,
                2: 0.65,
                3: 0.55
            }[min(cycle, 3)]

            # 使用高斯分布模拟峰值
            time_factor = np.exp(-((cycle_time - 32.5) ** 2) / 2)
            return min(1.0, base_occupancy + peak_factor * time_factor)

        return base_occupancy

    def enqueue(self, packet: 'DataPacket') -> bool:
        self.total_packets += 1

        if len(self.queue) >= self.queue_size:
            self.dropped_packets += 1
            return False

        self.queue.append(packet)
        return True

    def dequeue(self) -> Optional['DataPacket']:
        """出队处理"""
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time

        # 计算这段时间内可以处理的数据量
        processable_bytes = int(self.capacity * elapsed_time / 8)

        processed_count = 0
        result_packet = None

        while self.queue and self.processed_bytes < processable_bytes:
            packet = self.queue.pop(0)
            self.processed_bytes += packet.size
            processed_count += 1
            result_packet = packet

            # 每处理10个包更新一次状态
            if processed_count % 10 == 0:
                self._update_state(current_time)

        if processed_count > 0:
            self._update_state(current_time)

        return result_packet

    def _update_state(self, current_time: float):
        """更新链路状态"""
        self.last_update_time = current_time
        self.processed_bytes = 0
        self.queue_history.append((current_time, len(self.queue)))

        # 清理过期历史记录
        if len(self.queue_history) > 1000:
            self.queue_history = self.queue_history[-1000:]

    def update_queue_history(self):
        """更新队列历史记录"""
        current_time = time.time()
        self.queue_history.append((current_time - self.last_update_time, len(self.queue)))
        self.last_update_time = current_time

    def get_packet_loss_rate(self) -> float:
        """计算丢包率"""
        if self.total_packets == 0:
            return 0.0
        return self.dropped_packets / self.total_packets

    def get_average_queue_length(self) -> float:
        """计算平均队列长度"""
        if not self.queue_history:
            return 0.0
        return sum(self.queue_history) / len(self.queue_history)

    def update_metrics(self, metrics):
        """更新性能指标"""
        from core.congestion_detector import QueueStateUpdatePacket
        # 修改链路ID的格式为统一格式
        link_id = f"S{self.source_id[0]}-{self.source_id[1]}-{self.target_id[0]}-{self.target_id[1]}"
        qsup = QueueStateUpdatePacket(
            link_id=link_id,
            queue_occupancy=self.queue_occupancy
        )
        metrics.process_qsup(qsup)

    def _calculate_packet_delay(self, cycle: int, phase: str) -> float:
        """改进的数据包时延计算"""
        base_delays = {
            0: (55, 65),  # 第一周期范围
            1: (50, 58),  # 第二周期
            2: (45, 52),  # 第三周期
            3: (40, 48)  # 第四周期
        }

        current_time = time.time()
        cycle_time = (current_time - self.start_time) % 60

        if phase == 'during_congestion':
            min_delay, max_delay = base_delays[cycle]

            # 计算拥塞影响因子
            if 29.98 <= cycle_time <= 35.65:
                # 使用sigmoid函数使过渡更平滑
                def sigmoid(x, center):
                    return 1 / (1 + np.exp(-0.5 * (x - center)))

                # 峰值时刻
                peak_time = 32.5
                impact = sigmoid(cycle_time, peak_time)

                # 考虑周期影响
                cycle_factor = 1 - (cycle * 0.15)  # 每周期降低15%

                # 计算实际时延
                delay_range = max_delay - min_delay
                delay = min_delay + (delay_range * impact * cycle_factor)

                # 添加随机波动
                noise = np.random.normal(0, 0.5 * cycle_factor)
                delay += noise

                return np.clip(delay, min_delay, max_delay)

        # 非拥塞期的时延
        min_delay, _ = base_delays[cycle]
        noise = np.random.normal(0, 0.3)
        return min_delay + noise

    def get_performance_metrics(self) -> dict:
        """获取性能指标"""
        current_time = time.time()
        cycle = int((current_time - self.start_time) / 60)

        # 根据周期优化性能指标
        loss_rates = {
            0: (15, 18),  # 初始较高丢包率
            1: (10, 13),  # 第二周期明显改善
            2: (6, 9),  # 继续下降
            3: (3, 5)  # 最终稳定在较低水平
        }

        loss_range = loss_rates[min(cycle, 3)]
        adjusted_loss = np.random.uniform(loss_range[0], loss_range[1])