import torch
import torch.nn.functional as F

# 定义模型输入参数
from models.networks01 import BASE_Transformer

input_nc = 3        # 输入通道数
output_nc = 1       # 输出通道数
with_pos = 'learned'  # 是否使用位置编码
token_len = 4       # token 的长度
token_trans = True  # 是否进行 token 转换
num_blocks = 6      # MLP-Mixer 的块数
hidden_dim = 32    # 隐藏层维度
tokens_mlp_dim = 8  # token MLP 维度
channels_mlp_dim = 256  # channel MLP 维度
tokenizer = True    # 是否使用 tokenizer
if_upsample_2x = True  # 是否进行 2x 上采样
pool_mode = 'max'   # 池化模式
pool_size = 2       # 池化尺寸
backbone = 'resnet18'  # 使用的 ResNet 背骨
decoder_softmax = True  # 解码器是否使用 softmax
with_decoder_pos = 'learned'  # 解码器是否使用位置编码
with_decoder = False  # 是否使用解码器

# 初始化模型
model = BASE_Transformer(input_nc, output_nc, with_pos, resnet_stages_num=5,
                         token_len=token_len, token_trans=token_trans,
                         num_blocks=num_blocks, hidden_dim=hidden_dim,
                         tokens_mlp_dim=tokens_mlp_dim, channels_mlp_dim=channels_mlp_dim,
                         tokenizer=tokenizer, if_upsample_2x=if_upsample_2x,
                         pool_mode=pool_mode, pool_size=pool_size,
                         backbone=backbone, decoder_softmax=decoder_softmax,
                         with_decoder_pos=with_decoder_pos, with_decoder=with_decoder)

# 创建测试输入数据
batch_size = 8
height = 256
width = 256
input_tensor1 = torch.randn(batch_size, input_nc, height, width)
input_tensor2 = torch.randn(batch_size, input_nc, height, width)

# 运行前向传递
output = model(input_tensor1, input_tensor2)

# 打印输出形状
print("Output shape:", output.shape)
