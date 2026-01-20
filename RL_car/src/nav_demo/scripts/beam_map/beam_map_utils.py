import numpy as np

def generate_beam_map(scan_msg,num_beams = 512,max_range = 100.0):
    """
    参数：
        scan_msg:ROS LaserScan消息
        num_beams:网络输入的维度
        max_range:归一化的最大距离
    """
    # 1.处理原始雷达数据
    ranges = np.array(scan_msg.ranges)

    #处理inf和nan
    ranges = np.nan_to_num(ranges,posinf=max_range,neginf=0.0)
    ranges = np.clip(ranges,0,max_range)

    #2.降采样
    original_len = len(ranges)
    if original_len != num_beams:
        #使用线性插值重新采样到num_beams长度
        x = np.linspace(0,1,original_len)
        x_new = np.linspace(0,1,num_beams)
        beam_map = np.interp(x_new,x,ranges)
    else:
        beam_map = ranges
    
    #3.归一化到0-1
    normalized_map = beam_map / max_range

    return normalized_map