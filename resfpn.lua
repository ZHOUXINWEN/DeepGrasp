require 'nn'
require 'nngraph'
require 'cunn';
require 'cudnn';
optnet = require 'optnet'
nninit = require 'nninit'
torch.setdefaulttensortype('torch.CudaTensor')

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
--[[
con = {
  bpc = {6, 6, 6}, -- number of default boxes per cell
  imgshape = 320, -- input image size
 }

classe = 2

pretai = nil
--]]
local function createModel(classes, conf, pretrain)
  local depth = 50
  local shortcutType = 'B'
  --print(classes,conf)
  local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end
    
   local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

   assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
   local def, nFeatures, block = table.unpack(cfg[depth])
   iChannels = 64
   print(' | ResNet-' .. depth .. ' ImageNet')

   local main
   local branch2

   if pretrain ~= nil then
      main = torch.load(pretrain[1])
      branch2 = torch.load(pretrain[2])
      --print(main)
      --print(branch2)
   else
      main = nn.Sequential()
      main:add(Convolution(4,64,7,7,2,2,3,3))   -- 224*224 -> 112*112    2
      main:add(SBatchNorm(64))
      main:add(ReLU(true))
      main:add(Max(3,3,2,2,1,1))
      main:add(layer(block, 64, def[1], 2))        -- 112*112 -> 56*56      4  
      main:add(layer(block, 128, def[2], 2))    -- 56*56 -> 28*28*128        8
      --main:add(layer(block, 128, def[2], 2)) -- 128 -> 64
      branch2 = nn.Sequential()
      branch2:add(layer(block, 256, def[3], 2))    -- 28*28 -> 14*14*256        16
   end

    local branch3    
    local deconv1
    local deconv2
    local subbranch1 = nn.Sequential()
    local subbranch2 = nn.Sequential()
    local subbranch3 = nn.Sequential()
    local D1 = nn.SpatialFullConvolution(2048,1024,2,2,2,2)
    local D2 = nn.SpatialFullConvolution(1024,512,2,2,2,2)
    local Add = nn.CAddTable()   


    branch3 = nn.Sequential()
    branch3:add(layer(block, 512, def[4], 2))            -- 7x7x512    32    
   

    subbranch1:add(nn.ConcatTable()
    :add(nn.Sequential():add(Convolution(512, 5*conf.bpc[1], 3, 3, 1, 1, 1, 1))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, 4)))
    :add(nn.Sequential():add(Convolution(512, classes*conf.bpc[1], 3, 3, 1, 1, 1, 1))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, classes))))


    subbranch2:add(nn.ConcatTable()
    :add(nn.Sequential():add(Convolution(1024,5*conf.bpc[2], 3, 3, 1, 1, 1, 1))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, 4)))
    :add(nn.Sequential():add(Convolution(1024, classes*conf.bpc[2], 3, 3, 1, 1, 1, 1))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, classes))))


    subbranch3:add(nn.ConcatTable()
    :add(nn.Sequential():add(Convolution(2048, 5*conf.bpc[3], 3, 3, 1, 1, 1, 1))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, 4)))
    :add(nn.Sequential():add(Convolution(2048, classes*conf.bpc[3], 3, 3, 1, 1, 1, 1))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, classes))))

    input = nn.Identity()()
    input1 = nn.Identity()(input)
    OutBranch2 = branch2(input)
    OutBranch3 = branch3(OutBranch2)
    OutSub3 = subbranch3(OutBranch3)        
    OutD1 = D1(OutBranch3)
    InSub2 = nn.CAddTable()({OutD1,OutBranch2}) 
    OutD2 = D2(InSub2)
    InSub1 = nn.CAddTable()({OutD2,input1})     
    OutSub2 = subbranch2(InSub2) 
    OutSub1 = subbranch1(InSub1) 
    outputs = {OutSub1, OutSub2, OutSub3}

    fpn = nn.gModule({input},outputs) 

    main:add(fpn):add(nn.FlattenTable())

  -- transform
    main:add(nn.ConcatTable():add(nn.SelectTable(1))
    :add(nn.SelectTable(3))
    :add(nn.SelectTable(5))
    :add(nn.SelectTable(2))
    :add(nn.SelectTable(4))
    :add(nn.SelectTable(6)))

    main:add(nn.ConcatTable()
   :add(nn.Sequential()
   :add(nn.NarrowTable(1, 3))
   :add(nn.JoinTable(2)))
   :add(nn.Sequential()
   :add(nn.NarrowTable(4, 6))
   :add(nn.JoinTable(2))))
  --print(main)
  main = main:cuda()
  local inp = torch.randn(1, 4, conf.imgshape, conf.imgshape):cuda()
  local opts = {inplace=true, mode='training'}
  optnet.optimizeMemory(main, inp, opts)--]]
  return main
end

return createModel
--[[
net = createModel(classe, con, pretrai)
inputs = torch.randn(1,3,320,320)
print(net:forward(inputs))
--print(net:get(6).output)--]]
