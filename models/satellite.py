# models/satellite.py
from dataclasses import dataclass, field  # 添加field的导入
import time
from typing import Dict, Tuple

@dataclass
class Satellite:
    grid_pos: Tuple[int, int]
    links: Dict[str, 'Link'] = field(default_factory=dict)

    def __post_init__(self):
        self.local_metrics = {
            'processed_packets': 0,
            'total_delay': 0,
            'start_time': time.time()
        }

    def add_link(self, direction: str, target_sat: 'Satellite',
                 capacity: float = 25.0):
        """添加链路"""
        from models.link import Link  # 避免循环导入
        self.links[direction] = Link(self.grid_pos, target_sat.grid_pos, capacity)

    def update_metrics(self, packet_delay: float):
        """更新本地性能指标"""
        self.local_metrics['processed_packets'] += 1
        self.local_metrics['total_delay'] += packet_delay

    def get_average_delay(self) -> float:
        """获取平均时延"""
        if self.local_metrics['processed_packets'] == 0:
            return 0
        return (self.local_metrics['total_delay'] /
                self.local_metrics['processed_packets'])

    def get_current_cycle(self) -> int:
        """获取当前周期"""
        current_time = time.time()
        return min(3, int((current_time - self.local_metrics['start_time']) / 60))

    def get_link_states(self) -> dict:
        """获取所有链路状态"""
        states = {}
        for direction, link in self.links.items():
            states[f"{self.grid_pos}-{direction}"] = link.get_performance_metrics()
        return states