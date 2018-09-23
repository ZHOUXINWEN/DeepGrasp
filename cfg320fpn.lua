lapp = require 'pl.lapp'
-- network config
cfg = {
  scale = {27, 54, 108},
  nmap = 3, -- number of feature maps
  msize = {20, 10, 5}, -- feature matp size
  bpc = {6, 6, 6}, -- number of default boxes per cell
  aratio = {1, '1', 2, 1/2, 3, 1/3},  -- aspect retio
  variance = {0.1, 0.1, 0.1, 0.1, 0.1},
  steps = {16, 32, 64},
  imgshape = 320, -- input image size
  classes = 2,
  NegRatio = 3
}

opts = {}

-- parameters and others config
function opts.parse(arg)
  opt = lapp [[
    Command line options:
    Training Related:
    --lr         (default 1e-3)                    learning rate
    --momentum   (default 0.9)                     momentum
    --lrd        (default 0.0002)                  learning rate decay
    --wd         (default 0.0005)                  weight decay
    --snap       (default 2000)                   snapshot
    --iter       (default 120000)                   iterations
    --batchsize  (default 12)                      mini-batch size
    --test       (default 10000)                   test span
    --disp       (default 59)                     display span
    --output     (default ./output/ssd_pretrain_RGD_10x10x6)                 output directory
    --root       (default data)          dataset root directory
    --cache      (default ./cache)                 cache file directory
    --gpu        (default 3)                       gpu id
    --pretrain   (default nil)                     pretrain model root directory
    --seed       (default 1)
  ]]
  return opt
end
