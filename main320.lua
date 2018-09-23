require 'pl'
require 'image'
require 'optim'
require 'cutorch'
require 'gnuplot'

dofile('cfg320.lua')
opt = opts.parse(arg)
cutorch.setDevice(1)
createModel = dofile('ResNet.lua')
dofile('utils320_ang45.lua')
dofile('train320.lua')
print(opt)
print(cfg)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

local class_list = {'__background__', 'graspable'}

local class2num = {}
for i, v in pairs(class_list) do
  class2num[v] = i
end

if opt.pretrain ~= 'nil' then
  pretrain = paths.concat(opt.pretrain, 'resnet50_mainLeaky01.t7')
end
--[[
if opt.pretrain ~= 'nil' then
  pretrain = {paths.concat(opt.pretrain, 'main_branch.t7'), paths.concat(opt.pretrain, 'branch2.t7')}
end--]]

model = createModel(cfg.classes, cfg)

--[[result =torch.load('/home/zxw/DeepGrasp/output/resnet50_RGB_10x10x6_HV_R15deg_b16_lrd1/195epoch.t7')
model = result.model
opt = result['opt']--]]
--[[cfg = result['cfg']
opt.lr = opt.lr/math.sqrt(10)
opt.output = './output/VGGresblock_Leaky01_RGB_10x10x6_HV_R15deg_RN_conti'--]]
param, gparam = model:getParameters()

conf_mat = optim.ConfusionMatrix(class_list, #class_list)

opt_conf = {
  learningRate=opt.lr,
  learningRateDecay = opt.lrd,
  momentum=opt.momentum,
  nesterov = true,
  dampening = 0.0,
  weightDecay=opt.wd
}

FF = 1
traingt = torch.load('/home/zxw/DeepGrasp/cache/ow/RN_ow_cornell_'..FF..'_Train.t7')
trainpath = torch.load('/home/zxw/DeepGrasp/cache/ow/RN_ow_paths_'..FF..'_Train.t7')
testgt = torch.load('/home/zxw/DeepGrasp/cache/ow/RN_ow_cornell_'..FF..'_Test.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/ow/RN_ow_paths_'..FF..'_test.t7')

--train_RGBD = torch.load('/home/zxw/DeepGrasp/cache/train_RGBD.t7')
--test_RGBD = torch.load('/home/zxw/DeepGrasp/cache/test_RGBD.t7')

--print(train_RGBD['pcd0100r.png']:size())
print(#trainpath)

train()



