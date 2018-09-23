require 'nn';
require 'cunn';
require 'cudnn';
optnet = require 'optnet'
nninit = require 'nninit'
torch.setdefaulttensortype('torch.CudaTensor')
--[[
classe = 18
con = { 
   bpc = {6},
   imgshape = 336
}--]]


local function createModel(classes, conf)
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

  local main
  local branch = nn.Sequential()

  if pretrain ~= nil then
        main = torch.load(pretrain)
        print('load pretrained net from '..pretrain )
  else
    main = nn.Sequential() 
    main:add(xconv(3, 64, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(64, 64, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 512-> 256
    -- conv2 (module 6 ~ 10)
    main:add(xconv(64, 128, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(128, 128, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 256 -> 128
    -- conv3 (module 11 ~ 17)
    main:add(xconv(128, 256, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(256, 256, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(256, 256, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 128 -> 64
    -- conv4 (module 18 ~ 23)
    main:add(xconv(256, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(512, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    main:add(xconv(512, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    -- conv5 (module 24 ~ 31)
    main:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 64 -> 32
  end

    local resblock2 = nn.Sequential()
    local s = nn.Sequential()
    s:add(xconv(512, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    s:add(xconv(512, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))
    s:add(xconv(512, 512, 3, 3, 1, 1, 1, 1, 'N', 0, 0, true))

    resblock2:add(nn.ConcatTable()
              :add(s)
              :add(nn.Identity()))
            :add(nn.CAddTable(true))
            :add(nn.LeakyReLU(0.1)):add(cudnn.SpatialMaxPooling(2, 2, 2, 2)):add(nn.SpatialFullConvolution(512,512,2,2,2,2))

    local fpn = nn.Sequential()
    fpn:add(nn.ConcatTable()
              :add(resblock2)
              :add(nn.Identity()))
            :add(nn.CAddTable(true))
            :add(nn.LeakyReLU(0.1)):add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

    main:add(fpn)
    --main -- 64 -> 32
    branch:add(nn.ConcatTable()
    :add(nn.Sequential():add(xconv(512, 5*conf.bpc[1], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, 5)))
    :add(nn.Sequential():add(xconv(512, conf.classes*conf.bpc[1], 3, 3, 1, 1, 1, 1, 'N', 0, 0, false))
    :add(nn.Transpose({2,3},{3,4}))
    :add(nn.Reshape(-1, 2))))

    main:add(branch)
   
    main = main:cuda()
    local inp = torch.randn(1, 3, conf.imgshape, conf.imgshape):cuda()
    local opts = {inplace=true, mode='training'}
    optnet.optimizeMemory(main, inp, opts)
    return main
end

return createModel

--[[
net = createModel(classe, con)
inputs = torch.randn(2,3,336,336)
print(net:forward(inputs))--]]

