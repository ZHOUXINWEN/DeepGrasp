model_path = '/home/zxw/DeepGrasp/output/320b_12DGssdResAng448_v501/180epoch.t7'
img_dir ='/home/zxw/DeepGrasp/data/cornell_dataset/image'
torch.setdefaulttensortype('torch.FloatTensor')
-- _=dofile('ssd.lua')
--[[
require 'cudnn'
require 'cunn'
require 'nn'
--]]
dofile('cfg320.lua')
dofile('utils320.lua')
require 'image'

testgt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Test.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Test.t7')
--print(testpath[1])

result = torch.load(model_path)
model = result.model:cuda()
print(result['TestResult']['total'])

evaluate(model,testpath,'/home/zxw/DeepGrasp/data/cornell_dataset/image',180, cfg)
