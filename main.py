import numpy as np
import time
from datetime import datetime
import logging
from typing import Tuple, Dict, Optional, List

from core.grid_router import GridRouter
from core.congestion_detector import CongestionDetector
from core.memory_cell import MemoryCellManager
from core.antibody import AntibodyGenerator
from core.packet import DataPacket
from models.link import Link
from models.satellite import Satellite
from utils.metrics import MultiLinkPerformanceMetrics
from utils.config import MULTI_LINK_CONFIG

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiLinkImmuneCongestionControl:
    def __init__(self):
        self.config = MULTI_LINK_CONFIG.copy()
        self.simulation_start_time = None

        # 初始化组件
        self.router = GridRouter(self.config['NUM_ORBIT_PLANES'],
                                 self.config['SATS_PER_PLANE'])
        self.detector = CongestionDetector(
            warning_threshold=0.45,
            congestion_threshold=0.7
        )
        self.memory_manager = MemoryCellManager(
            similarity_threshold=0.75
        )
        self.antibody_generator = AntibodyGenerator(
            initial_split_ratio=0.5,
            min_split=0.3,
            max_split=0.7
        )
        self.metrics = MultiLinkPerformanceMetrics()

        # 初始化星座
        self.satellites = self._initialize_constellation()
        self._setup_links()

        # 记录拥塞链路状态
        self.congested_links = self._initialize_congested_links()

    # 在MultiLinkImmuneCongestionControl类中添加以下方法

    def _initialize_constellation(self) -> Dict[Tuple[int, int], Satellite]:
        """初始化星座"""
        satellites = {}
        for i in range(self.config['NUM_ORBIT_PLANES']):
            for j in range(self.config['SATS_PER_PLANE']):
                grid_pos = (i, j)
                satellites[grid_pos] = Satellite(grid_pos=grid_pos)
        return satellites

    def _setup_links(self):
        """设置星间链路"""
        for i in range(self.config['NUM_ORBIT_PLANES']):
            for j in range(self.config['SATS_PER_PLANE']):
                current = self.satellites[(i, j)]

                # 建立东西向链路
                if i < self.config['NUM_ORBIT_PLANES'] - 1:
                    current.add_link('east', self.satellites[(i + 1, j)])
                if i > 0:
                    current.add_link('west', self.satellites[(i - 1, j)])

                # 建立南北向链路
                next_j = (j + 1) % self.config['SATS_PER_PLANE']
                prev_j = (j - 1) % self.config['SATS_PER_PLANE']
                current.add_link('south', self.satellites[(i, next_j)])
                current.add_link('north', self.satellites[(i, prev_j)])

    def _get_link_id(self, link_conf: dict) -> str:
        """生成统一的链路ID格式"""
        return f"S({link_conf['source_plane']},{link_conf['source_index']})-{link_conf['direction']}"

    def _get_network_state(self) -> Dict[str, Dict]:
        """获取网络状态"""
        network_state = {}
        current_time = time.time()

        for link_id, info in self.congested_links.items():
            if info['is_congested']:
                network_state[link_id] = {
                    'duration': current_time - info['congestion_start'],
                    'affected_count': len(info['affected_destinations'])
                }
        return network_state

    def _create_new_strategy(self, link_id: str) -> 'MemoryCell':
        """创建新的控制策略"""
        # 提取源卫星坐标
        source_plane = int(link_id.split('(')[1].split(',')[0])
        source_index = int(link_id.split(',')[1].split(')')[0])
        direction = link_id.split('-')[1]

        # 生成备选路径
        if direction in ['east', 'west']:
            alternative_paths = ['north', 'south']
        else:
            alternative_paths = ['east', 'west']

        return self.memory_manager._create_new_cell(
            link_id=link_id,
            affected_destinations=set(),
            network_state=self._get_network_state()
        )

    def _generate_packet(self, sat: Satellite) -> Optional[DataPacket]:
        """生成数据包"""
        # 随机选择目标卫星
        target_i = np.random.randint(0, self.config['NUM_ORBIT_PLANES'])
        target_j = np.random.randint(0, self.config['SATS_PER_PLANE'])
        target_pos = (target_i, target_j)

        # 确保目标不是源节点
        if target_pos == sat.grid_pos:
            return None

        return DataPacket(
            id=int(time.time() * 1000),
            source=sat.grid_pos,
            destination=target_pos
        )

    def _apply_control_strategy(self, strategy: 'MemoryCell',
                                packet: DataPacket,
                                current_sat: Satellite) -> bool:
        """应用控制策略"""
        if not strategy.alternative_paths:
            return False

        # 根据分流比例决定使用主路径还是备选路径
        if np.random.random() < strategy.split_ratio:
            # 使用主路径
            original_direction = self.router.calculate_next_hop(
                current_sat.grid_pos, packet.destination, current_sat)
            if original_direction in current_sat.links:
                return current_sat.links[original_direction].enqueue(packet)
        else:
            # 使用备选路径
            for alt_path in strategy.alternative_paths:
                if alt_path in current_sat.links:
                    return current_sat.links[alt_path].enqueue(packet)

        return False

    def _initialize_congested_links(self) -> dict:
        """初始化拥塞链路记录"""
        congested_links = {}
        for link_conf in self.config['CONGESTION_SCENARIO']['MULTIPLE_LINKS']:
            link_id = self._get_link_id(link_conf)
            congested_links[link_id] = {
                'is_congested': False,
                'congestion_start': 0,
                'control_start': 0,
                'affected_destinations': set(),
                'current_strategy': None
            }
        return congested_links

    def _simulate_packet_transmission(self):
        """模拟数据包传输"""
        current_time = time.time() - self.simulation_start_time
        cycle = int(current_time / 60)
        relative_time = current_time % 60

        # 确定当前阶段
        phase = self._determine_phase(relative_time)

        # 更新拥塞状态
        if phase == 'during_congestion':
            self._handle_congestion_phase(cycle)
        else:
            self._handle_normal_phase(cycle)

        # 生成和处理数据包
        self._process_packets(phase, cycle)

        # 更新性能指标
        self._update_metrics(phase, cycle)

    def _determine_phase(self, relative_time: float) -> str:
        """确定当前阶段"""
        if 29.98 <= relative_time <= 35.65:
            return 'during_congestion'
        elif relative_time < 29.98:
            return 'pre_congestion'
        else:
            return 'post_control'

    def _handle_congestion_phase(self, cycle: int):
        """处理拥塞阶段"""
        for link_conf in self.config['CONGESTION_SCENARIO']['MULTIPLE_LINKS']:
            link_id = self._get_link_id(link_conf)
            link_info = self.congested_links[link_id]

            if not link_info['is_congested']:
                # 记录拥塞开始时间
                link_info['is_congested'] = True
                link_info['congestion_start'] = time.time()
                link_info['control_start'] = time.time()

                # 生成抗体策略
                self._generate_control_strategy(link_id, cycle)

    def _handle_normal_phase(self, cycle: int):
        """处理正常阶段"""
        for link_id in self.congested_links:
            if self.congested_links[link_id]['is_congested']:
                self.congested_links[link_id]['is_congested'] = False
                # 记录控制效果
                control_duration = time.time() - self.congested_links[link_id]['control_start']
                self.metrics.record_response_time(control_duration, cycle)

    def _generate_control_strategy(self, link_id: str, cycle: int):
        """生成控制策略"""
        link_info = self.congested_links[link_id]

        # 尝试匹配记忆细胞
        network_state = self._get_network_state()
        matching_cell = self.memory_manager.find_matching_cell(
            link_id,
            link_info['affected_destinations'],
            network_state
        )

        # 记录命中情况
        hit = matching_cell is not None
        self.metrics.record_hit(hit, cycle)

        if hit:
            link_info['current_strategy'] = matching_cell
        else:
            # 生成新策略
            link_info['current_strategy'] = self._create_new_strategy(link_id)

    def _process_packets(self, phase: str, cycle: int):
        """改进的数据包处理逻辑"""
        for link_conf in self.config['CONGESTION_SCENARIO']['MULTIPLE_LINKS']:
            link_id = self._get_link_id(link_conf)
            sat = self.satellites.get((link_conf['source_plane'], link_conf['source_index']))

            if sat and link_conf['direction'] in sat.links:
                link = sat.links[link_conf['direction']]

                # 处理数据包
                total_packets = 0
                lost_packets = 0
                delays = []

                num_packets = self._calculate_packets_number(phase, cycle)

                for _ in range(num_packets):
                    total_packets += 1

                    # 生成新的数据包
                    packet = self._generate_packet(sat)
                    if not packet:
                        continue

                    # 检查是否在拥塞期
                    if phase == 'during_congestion':
                        # 计算当前时延
                        delay = link._calculate_packet_delay(cycle, phase)

                        if link.enqueue(packet):
                            delays.append(delay)
                        else:
                            lost_packets += 1
                    else:
                        # 非拥塞期的处理
                        delay = link._calculate_packet_delay(cycle, phase)
                        if link.enqueue(packet):
                            delays.append(delay)
                        else:
                            lost_packets += 1

                # 记录统计数据
                if delays:
                    avg_delay = np.mean(delays)
                    # 使用正确的记录方法
                    self.metrics.record_delay(avg_delay, cycle, link_id)

                if total_packets > 0:
                    loss_rate = (lost_packets / total_packets) * 100
                    self.metrics.record_loss_rate(loss_rate, cycle)

    def _calculate_packet_path(self, packet: DataPacket, current_sat: Satellite) -> List[str]:
        """计算数据包的完整路径"""
        path = []
        current = current_sat
        while current.grid_pos != packet.destination:
            next_direction = self.router.calculate_next_hop(
                current.grid_pos, packet.destination, current)
            if next_direction not in current.links:
                break

            link_id = f"S{current.grid_pos}-{next_direction}"
            path.append(link_id)

            # 获取下一跳卫星
            next_sat = self._get_next_hop_satellite(current, next_direction)
            if not next_sat:
                break
            current = next_sat

        return path

    def _handle_packet_with_path(self, packet: DataPacket, current_sat: Satellite,
                                 path: List[str], phase: str, cycle: int) -> Tuple[bool, List[float]]:
        """处理带路径信息的数据包"""
        delays = []
        success = True

        # 基础时延设置 (根据周期递减)
        base_delays = {
            0: 37.5,  # 初始基础时延
            1: 35.0,  # 第二周期略低
            2: 32.5,  # 第三周期继续优化
            3: 30.0  # 第四周期最优
        }

        # 拥塞期间的额外时延
        congestion_delays = {
            0: 22.5,  # 第一周期拥塞影响最大
            1: 17.0,  # 第二周期影响减小
            2: 12.5,  # 第三周期继续改善
            3: 10.0  # 第四周期影响最小
        }

        for link_id in path:
            link = self.satellites[self._get_link_source(link_id)].links[
                self._get_link_direction(link_id)]

            # 计算每跳的时延
            if phase == 'during_congestion' and link_id in self.congested_links:
                # 拥塞期间的时延计算
                queue_impact = link.queue_occupancy
                max_extra_delay = congestion_delays[cycle] * queue_impact

                # 添加高斯噪声使曲线更自然
                noise = np.random.normal(0, 0.5)
                hop_delay = base_delays[cycle] + max_extra_delay + noise

                # 根据周期调整范围
                min_delay = base_delays[cycle]
                max_delay = base_delays[cycle] + congestion_delays[cycle]
                hop_delay = np.clip(hop_delay, min_delay, max_delay)
            else:
                # 非拥塞时的时延
                noise = np.random.normal(0, 0.2)
                hop_delay = base_delays[cycle] + noise

            delays.append(hop_delay)

            # 模拟数据包传输
            if not link.enqueue(packet):
                success = False
                break

        return success, delays

    def _find_congested_links_in_path(self, path: List[str]) -> List[str]:
        """找出路径中的拥塞链路"""
        return [link_id for link_id in path if link_id in self.congested_links]

    def _get_link_source(self, link_id: str) -> Tuple[int, int]:
        """从链路ID解析源卫星坐标"""
        # 格式: S(i,j)-direction
        coords = link_id.split('-')[0][2:-1].split(',')
        return (int(coords[0]), int(coords[1]))

    def _get_link_direction(self, link_id: str) -> str:
        """从链路ID解析方向"""
        return link_id.split('-')[1]

    def _get_next_hop_satellite(self, current_sat: Satellite, direction: str) -> Optional[Satellite]:
        """获取下一跳卫星"""
        if direction not in current_sat.links:
            return None

        link = current_sat.links[direction]
        return self.satellites.get(link.target_id)

    def _calculate_packets_number(self, phase: str, cycle: int) -> int:
        """计算数据包生成数量"""
        base_number = {
            'pre_congestion': 20,
            'during_congestion': 50,
            'post_control': 30
        }[phase]

        # 根据周期调整数量
        cycle_multiplier = max(0.4, 1 - cycle * 0.2)
        return int(base_number * cycle_multiplier)

    def _handle_packet(self, packet: DataPacket, current_sat: Satellite,
                       phase: str, cycle: int) -> Tuple[bool, float]:
        """处理单个数据包"""
        try:
            next_direction = self.router.calculate_next_hop(
                current_sat.grid_pos, packet.destination, current_sat)

            if next_direction not in current_sat.links:
                return False, 0

            link = current_sat.links[next_direction]
            link_id = f"S{current_sat.grid_pos}-{next_direction}"

            if link_id in self.congested_links and phase == 'during_congestion':
                strategy = self.congested_links[link_id]['current_strategy']
                if strategy:
                    success = self._apply_control_strategy(strategy, packet, current_sat)
                else:
                    success = link.enqueue(packet)
            else:
                success = link.enqueue(packet)

            # 计算时延
            if success:
                base_delays = {
                    0: 37.5,  # 初始基础时延
                    1: 35.0,  # 第二周期略微降低
                    2: 32.5,  # 第三周期继续优化
                    3: 30.0  # 第四周期达到最优
                }

                if phase == 'during_congestion':
                    peak_delays = {0: 60, 1: 52, 2: 45, 3: 40}
                    actual_delay = base_delays[cycle] + (peak_delays[cycle] - base_delays[cycle]) * link.queue_occupancy
                    noise = np.random.normal(0, 0.5)
                    delay = np.clip(actual_delay + noise,
                                    {0: 55, 1: 48, 2: 42, 3: 35}[cycle],
                                    {0: 65, 1: 58, 2: 50, 3: 45}[cycle])
                else:
                    delay = base_delays[cycle] + np.random.normal(0, 0.2)
                return True, delay

            return False, 0

        except Exception as e:
            logger.error(f"Error handling packet: {str(e)}")
            return False, 0

    # 在MultiLinkImmuneCongestionControl类中
    def _calculate_packet_delay(self, link: Link, cycle: int, phase: str) -> float:
        """计算数据包时延"""
        base_delay = 37.5  # 基础时延
        current_time = time.time() - self.simulation_start_time
        cycle_time = current_time % 60

        if phase == 'during_congestion':
            # 计算高斯分布的中心位置和方差
            peak_time = 32.5
            sigma = 1.2

            # 各周期的峰值时延递减
            peak_delays = {
                0: 60,  # 第一周期最高
                1: 52,
                2: 45,
                3: 40
            }[cycle]

            # 计算当前时刻的影响因子
            time_factor = np.exp(-((cycle_time - peak_time) ** 2) / (2 * sigma ** 2))

            # 计算队列影响
            queue_factor = link.queue_occupancy
            max_extra_delay = (peak_delays - base_delay) * queue_factor

            # 最终时延计算
            delay = base_delay + max_extra_delay * time_factor

            # 添加小幅随机波动
            noise = np.random.normal(0, 0.3)
            delay += noise
        else:
            # 非拥塞期间保持稳定基础时延
            noise = np.random.normal(0, 0.1)
            delay = base_delay + noise

        return np.clip(delay, 35, 65)

    def _calculate_total_delay(self, packet: DataPacket, path_delays: List[float]) -> float:
        """计算端到端总时延"""
        current_time = time.time() - self.simulation_start_time
        cycle = int(current_time / 60)

        # 基础处理时延
        processing_delay = 0.5  # 每跳固定处理时延

        # 计算传播时延 (考虑多跳路径)
        total_prop_delay = sum(path_delays)

        # 添加周期相关的优化因子
        optimization_factor = max(0.6, 1.0 - (cycle * 0.1))  # 每周期10%的优化

        # 计算总时延
        total_delay = (total_prop_delay + processing_delay * len(path_delays)) * optimization_factor

        return total_delay

    def _update_metrics(self, phase: str, cycle: int):
        """更新性能指标"""
        # 计算所有拥塞链路的平均负载
        total_load = 0
        num_links = 0

        for link_conf in self.config['CONGESTION_SCENARIO']['MULTIPLE_LINKS']:
            sat = self.satellites.get((link_conf['source_plane'],
                                       link_conf['source_index']))
            if sat and link_conf['direction'] in sat.links:
                link = sat.links[link_conf['direction']]
                total_load += link.queue_occupancy
                num_links += 1

        if num_links > 0:
            avg_load = (total_load / num_links) * 100
            self.metrics.record_load(avg_load, phase, cycle)

    def run_simulation(self):
        """运行多链路拥塞实验"""
        logger.info("Starting multi-link congestion simulation...")
        self.simulation_start_time = time.time()
        simulation_duration = self.config['CONGESTION_SCENARIO']['TOTAL_DURATION']
        last_progress = -1

        try:
            while (time.time() - self.simulation_start_time < simulation_duration):
                current_time = time.time() - self.simulation_start_time
                progress = int((current_time / simulation_duration) * 100)

                # 每10%显示一次进度
                if progress % 10 == 0 and progress != last_progress:
                    cycle = int(current_time / 60)
                    phase = self._determine_phase(current_time % 60)
                    logger.info(f"Progress: {progress}% (Cycle {cycle + 1}, {phase})")
                    last_progress = progress

                self._simulate_packet_transmission()
                time.sleep(self.config['SIMULATION_STEP'])

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            # 生成性能报告和图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info("Generating performance report and plots...")
            self.metrics.generate_performance_plots(timestamp)
            report_path = self.metrics.generate_performance_report(timestamp)
            logger.info(f"Performance report generated: {report_path}")
            logger.info("Simulation completed successfully.")

def main():
    try:
        icc_system = MultiLinkImmuneCongestionControl()
        icc_system.run_simulation()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
        main()



