require 'pl'
require 'image'
require 'optim'
require 'cutorch'
require 'gnuplot'
require 'nn'
require 'cudnn'


torch.setdefaulttensortype('torch.FloatTensor')
dofile('utils320.lua')





result = torch.load('/home/zxw/DeepGrasp/output/Resnet101_RGB_10x10x6_HV_R15deg_b16_lrd1_RN/1615epoch.t7')
cfg = result['cfg']
--print(cfg)
model = result['model']:cuda()
--print(model)
--


testgt = torch.load('/home/zxw/DeepGrasp/cache/RNcornell_5_Test.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/RNpaths_5_Test.t7')
traingt = torch.load('/home/zxw/DeepGrasp/cache/RNcornell_5_Train.t7')
trainpath = torch.load('/home/zxw/DeepGrasp/cache/RNpaths_5_Train.t7')


epoch = 390
per_img_acc = evaluate(model,testpath,'/home/zxw/DeepGrasp/data/cornell_dataset/image', epoch, cfg)
print(per_img_acc['total'])
print(per_img_acc['Avgtime'])
print(result['TestResult']['total'])
print(result['TestResult']['Avgtime'])
--]]
