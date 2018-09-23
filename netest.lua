require 'cudnn'
require 'cutorch'
require 'cunn'
require 'image'
require 'optim'
dofile('cfg.lua')

createModel = require 'ResNet.lua'
torch.setdefaulttensortype('torch.FloatTensor')
local cfg = {
  scale = 54,
  nmap = 1, -- number of feature maps
  msize = {10}, -- feature matp size
  bpc = {6}, -- number of default boxes per cell
  aratio = {1, '1', 2, 1/2, 3, 1/3},  -- aspect retio
  variance = {0.1, 0.1, 0.2, 0.2, 0.2},
  steps = {32},
  imgshape = 320, -- input image size
  classes = 2
}
net = createModel(cfg.classes, cfg)

param, gparam = net:getParameters()

inputs = torch.randn(2,3,336,336)
local imgs = torch.FloatTensor(2, 3, 336, 336)
im1 = image.load('/home/zxw/DeepGrasp/data/cornell_dataset/image/pcd0101r.png')
im2 = image.load('/home/zxw/DeepGrasp/data/cornell_dataset/image/pcd0102r.png')
im1 = image.crop(im1,0,0,336,336)
im2 = image.crop(im2,0,0,336,336)
imgs[1] = im1
imgs[2] = im2
--print(imgs)
loc_grad = torch.randn(2,2646,5)
conf_grad = torch.randn(2,2646,2)
net:forward(imgs:cuda())

net:backward(imgs:cuda(),{loc_grad, conf_grad})
--print(torch.type(loc_grad),loc_grad:size())
--print(torch.type(conf_grad),conf_grad:size())
--[[local function feval() return loss, gparam end
      -- parameter update
optim.sgd(feval, param, opt_conf)--]]
