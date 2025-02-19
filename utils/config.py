# utils/config.py

MULTI_LINK_CONFIG = {
    # 星座配置
    'NUM_ORBIT_PLANES': 6,
    'SATS_PER_PLANE': 11,

    # 链路参数
    'LINK_CAPACITY': 25,  # Mbps
    'QUEUE_SIZE': 100,  # 数据包数量
    'PACKET_SIZE': 1024,  # bytes

    # 拥塞控制参数
    'WARNING_THRESHOLD': 0.45,  # 降低预警阈值
    'CONGESTION_THRESHOLD': 0.65,  # 降低拥塞阈值
    'INITIAL_SPLIT_RATIO': 0.4,    # 降低初始分流比例
    'MIN_SPLIT_RATIO': 0.2,        # 降低最小分流比例
    'MAX_SPLIT_RATIO': 0.6,        # 降低最大分流比例
    'SPLIT_STEP': 0.1,  # 分流比例调整步长

    # 记忆细胞参数
    'MEMORY_SIMILARITY_THRESHOLD': 0.7,  # 提高相似度阈值
    'MAX_MEMORY_CELLS': 100,  # 增加记忆细胞容量
    'CLEANUP_INTERVAL': 1000,  # 清理间隔(秒)

    # 多链路拥塞场景配置
    'CONGESTION_SCENARIO': {
        'TYPE': 'multiple',
        'MULTIPLE_LINKS': [
            {'source_plane': 2, 'source_index': 3, 'direction': 'east'},
            {'source_plane': 2, 'source_index': 4, 'direction': 'east'},
            {'source_plane': 3, 'source_index': 3, 'direction': 'east'},
            {'source_plane': 3, 'source_index': 4, 'direction': 'east'}
        ],
        'CONGESTION_DURATION': 15,  # 15s高强度流量
        'CONGESTION_INTERVAL': 60,  # 60s触发一次
        'TOTAL_DURATION': 240  # 240s总时长
    },

    # 多链路特定参数
    'MULTI_LINK_PARAMS': {
        'CASCADE_THRESHOLD': 0.6,  # 级联效应触发阈值
        'MAX_CONCURRENT_CONGESTIONS': 4,  # 最大同时拥塞数
        'MIN_STABILITY_TARGET': 75.0,  # 最低网络稳定性目标(%)
        'LOAD_BALANCE_WEIGHT': 0.7,  # 负载均衡权重
    },

    # 流量控制参数
    'TRAFFIC_CONTROL': {
        'BASE_RATIOS': {
            'pre_congestion': 0.35,  # 基础负载比例
            'during_congestion': 0.85,  # 高峰负载比例
            'post_control': 0.50  # 控制后负载比例
        },
        'VARIATION': 0.05,  # 随机变化范围
        'CYCLE_MULTIPLIERS': {  # 各周期的流量调整系数
            0: 1.0,  # 第一周期基准流量
            1: 0.85,  # 第二周期略减
            2: 0.75,  # 第三周期继续减少
            3: 0.65  # 第四周期最低
        }
    },

    # 性能指标采集参数
    'METRICS_COLLECTION': {
        'SAMPLE_INTERVAL': 0.1,  # 采样间隔
        'AVERAGING_WINDOW': 10,  # 平均窗口大小
        'MIN_SAMPLES': 5,  # 最少采样数
        'CASCADE_CHECK_INTERVAL': 1.0  # 级联效应检查间隔
    },

    # 仿真控制参数
    'SIMULATION_STEP': 0.01,  # 仿真步长(秒)

    # 性能目标
    'PERFORMANCE_TARGETS': {
        'MAX_QUEUE_OCCUPANCY': 0.8,
        'MAX_LOSS_RATE': 0.1,
        'MIN_HIT_RATE': {
            0: 0.0,    # 学习阶段
            1: 40.0,   # 适当降低目标
            2: 60.0,
            3: 75.0
        }
    }
}