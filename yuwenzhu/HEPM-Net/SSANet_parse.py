import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')

# 选择数据集
cls_set = 5
##################################
"""
模型参数相关变量
"""
####################################################################
parser.add_argument('--epochs', type=int, default=100, help='模型迭代次数')
parser.add_argument('--batch', type=int, default=8, help='模型批次大小')
parser.add_argument('--pre_trained', type=bool, default=False, help='是否预训练')
# parser.add_argument('--cls', type=int, default=16, help='模型类别数量')
parser.add_argument('--lr', type=float, default=0.003, help='模型学习率')
parser.add_argument('--decay', type=float, default=1e-06, help='模型学习率')
# parser.add_argument('--bands', type=int, default=30, help='HSI波段数量')
parser.add_argument('--pca', type=int, default=32, help='波段经过pca降维后的数量')

parser.add_argument('--test_ratio', type=float, default=0.9837, help='测试样本占的比例IP0.9，SV/Honghu 0.99,Coast0.9719')
parser.add_argument('--val_ratio', type=float, default=0.0, help='测试样本占的比例IP0.9，SV/Honghu 0.99,Coast0.9719')
parser.add_argument('--patch', type=int, default=13, help='patch块的大小')
parser.add_argument('--cls_report', type=str, default='cls_result/classification_report_', help='cls report保存路径')


parser.add_argument('--model_pth', type=str, default='cls_params/checkpoint.pth', help='模型保存路径')
####################################################################

##################################
"""
数据集相关变量
"""
####################################################################
####################################################################
parser.add_argument('--dataset_name', type=str, default=["IP","Honghu","Salinas","Coast","KSC","NC13","yellow"][cls_set], help='数据集名称')
parser.add_argument('--cls', type=int, default=[16, 22, 16, 16,13, 13, 9][cls_set], help='模型类别数量')
parser.add_argument('--target_names', type=list, default=[
                    ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn','Grass-pasture',
                     'Grass-trees', 'Grass-pasture-mowed','Hay-windrowed','Oats',
                     'Soybean-notill', 'Soybean-mintill','Soybean-clean', 'Wheat', 'Woods',
                     'Buildings-Grass-Trees-Drives','Stone-Steel-Towers'],
                    ["Red roof","Road","Bare soil","Cotton","Cotton firewood",
                     "Rape","Chinese cabbage","Pakchoi","Cabbage","Tuber mustard",
                     "Brassica parachinensis","Brassica chinensis","Small Brassica chinensis",
                     "Lactuca sativa","Celtuce","Film covered lettuce","Romaine lettuce",
                     "Carrot","White radish","Garlic sprout","Broad bean","Tree"],
                    ["Brocoli_green_weeds_1","Brocoli_green_weeds_2","Fallow","Fallow_rough_plow",
                    "Fallow_smooth","Stubble","Celery","Grapes_untrained","Soil_vinyard_develop",
                     "Corn_senesced_green_weeds","Lettuce_romaine_4wk","Lettuce_romaine_5wk","Lettuce_romaine_6wk",
                     "Lettuce_romaine_7wk","Vinyard untrained","Vinyard _vertical_trellis"],
                    ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
                    "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                    "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk",
                    "Lettuce_romaine_7wk", "Vinyard untrained", "Vinyard _vertical_trellis"],
                    ["Healthy Grass","Stressed Grass","Synthetis Grass","Tree",
                     "Soil","Water","Residential","Commercial","Road","Highway"
                     "Railway","Parking Lot 1","Parking Lot 2","Tennis Court","Running Track"],
                    # ["Roof","Road","Meadow","Tree","Path","Water","Shadow"],
                    ['Spartina alternifora', 'Reed', 'Tarix','Suaeda salsa', 'Mixed Zone',
                     'Tdal flat','Yellow River', 'Pond', 'sea area',
                     "Roof","Road","Meadow","TreePath"],
                    ["Roof","Road","Meadow","Tree","Path","Water","Shadow", 'Tdal flat','Yellow River']][cls_set], help='datset’s target name')


####################################################################

args = parser.parse_args()

