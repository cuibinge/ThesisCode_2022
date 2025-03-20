from detectron2.config import get_cfg, CfgNode

def print_config(cfg_node, indent=0):
    """递归打印配置树的实用函数"""
    for key, value in cfg_node.items():
        if isinstance(value, CfgNode):
            print(' ' * indent + f"▸ {key}:")
            print_config(value, indent + 2)
        else:
            print(' ' * indent + f"{key}: {repr(value)}")

# 初始化配置
cfg = get_cfg()

# 解冻并初始化SEM_SEG_HEAD节点
cfg.defrost()
cfg.MODEL.SEM_SEG_HEAD = CfgNode()
cfg.MODEL.SEM_SEG_HEAD.KG_PATH = ""  # 设置默认值
cfg.merge_from_file("/mnt/cat/CAT-Seg-main/configs/vitb_LoveDA4.yaml")
cfg.freeze()

# 打印完整配置树
print("完整配置参数列表：\n")
print_config(cfg)

# 单独验证KG_PATH
print("\n重点参数验证：")
print(f"MODEL.SEM_SEG_HEAD.KG_PATH = {cfg.MODEL.SEM_SEG_HEAD.KG_PATH}")
assert cfg.MODEL.SEM_SEG_HEAD.KG_PATH != "", "KG_PATH必须被正确定义"