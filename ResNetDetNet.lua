require 'nn';
require 'cunn';
require 'cudnn';
require 'cudnnorn'
optnet = require 'optnet'
nninit = require 'nninit'
torch.setdefaulttensortype('torch.CudaTensor')
--[[
classe = 18
con = { 
   bpc = {6},
   imgshape = 336
}--]]


local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = nn.LeakyReLU(0.1)
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(classes, conf)
   local shortcutType = 'B'
   local iChannels

  -- xavier initilization and bias zero
  local function xconv(ic,oc,kw,kh,sw,sh,pw,ph,type,dw,dh,relu)
    local conv
    use_relu = relu
    if type == 'N' then
      conv = cudnn.SpatialConvolution(ic, oc, kw, kh, sw, sh, pw, ph):init('weight', nninit.xavier, {dist='uniform', gain=1.1})
    elseif type == 'D' then
      local karnel = torch.randn(oc, ic, kw, kh)
      conv = nn.SpatialDilatedConvolution(ic, oc, kw, kh, sw, sh, pw, ph, pw, ph)
      nninit.xavier(nn.SpatialConvolution(ic, oc, kw, kh, sw, sh, pw, ph), karnel, {dist='uniform', gain=1.1})
      conv.weight:copy(karnel)
    end
    if cudnn.version >= 4000 then
      conv.bias = nil
      conv.gradBias = nil
    else
      conv.bias:zero()
    end
    if use_relu then
      return nn.Sequential():add(conv):add(nn.SpatialBatchNormalization(oc)):add(nn.LeakyReLU(0.1))
    else
      return conv
    end
  end

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
      s:add(ReLU)
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU)
   end

   local function ADilatedbottleneck()

      local s = nn.Sequential()
      s:add(Convolution(1024, 256, 1, 1, 1, 1, 0, 0))
      s:add(SBatchNorm(256))
      s:add(nn.LeakyReLU(0.1))
      s:add(nn.ORConv(32, 32, 8, 3, 3, 1, 1, 1, 1))
      s:add(SBatchNorm(256))
      s:add(nn.LeakyReLU(0.1))
      s:add(Convolution(256, 1024, 1, 1, 1, 1, 0, 0))
      s:add(SBatchNorm(256 * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
         :add(nn.CAddTable(true))
         :add(nn.LeakyReLU(0.1))
   end

   local function BDilatedbottleneck()

      local s = nn.Sequential()
      s:add(Convolution(1024, 256, 1, 1, 1, 1, 0, 0))
      s:add(SBatchNorm(256))
      s:add(nn.LeakyReLU(0.1))
      s:add(nn.ORConv(32, 32, 8, 3, 3, 1, 1, 1, 1))
      s:add(SBatchNorm(256))
      s:add(nn.LeakyReLU(0.1))
      s:add(Convolution(256, 1024, 1, 1, 1, 1, 0, 0))
      s:add(SBatchNorm(256 * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(nn.Sequential()
            :add(Convolution(1024, 1024, 1, 1, 1, 1))
            :add(SBatchNorm(1024))))
         :add(nn.CAddTable(true))
         :add(nn.LeakyReLU(0.1))
   end

   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

  local main
  local branch = nn.Sequential()
  if pretrain ~= nil then
        main = torch.load(pretrain)
        print('load pretrained net from '..pretrain )
  else   
    iChannels = 64
    main = nn.Sequential()
    main:add(Convolution(3,64,7,7,2,2,3,3))   -- 224*224 -> 112*112    2
    main:add(SBatchNorm(64))
    main:add(ReLU)
    main:add(Max(3,3,2,2,1,1))                -- 112*112 -> 56*56      4 
    main:add(layer(basicblock, 64, 2))         
    main:add(layer(basicblock, 128, 2, 2))    -- 56*56 -> 28*28        8
    main:add(layer(basicblock, 256, 2, 2))    -- 28*28 -> 14*14        16
    --main:add(layer(basicblock, 512, 2, 2))    -- 14*14 -> 7*7
  end

    main:add(BDilatedbottleneck())
    --main:add(ADilatedbottleneck())
    main:add(ADilatedbottleneck())
    main:add(ADilatedbottleneck())

    main:add(BDilatedbottleneck())
    --main:add(ADilatedbottleneck())
    main:add(ADilatedbottleneck())
    main:add(ADilatedbottleneck())

    branch:add(nn.ConcatTable()
    :add(nn.Sequential():add(xconv(1024, 5*conf.bpc[1], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, 5)))
    :add(nn.Sequential():add(xconv(1024, conf.classes*conf.bpc[1], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, 2))))

    main:add(branch)


   local function ConvInit(name)
      for k,v in pairs(main:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(main:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end
if pretrain == nil then
   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')

   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
end
    main = main:cuda()

    local inp = torch.randn(1, 3, conf.imgshape, conf.imgshape):cuda()
    local opts = {inplace=true, mode='training'}
    optnet.optimizeMemory(main, inp, opts)
    return main
end

return createModel
