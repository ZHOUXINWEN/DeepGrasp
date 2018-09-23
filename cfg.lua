lapp = require 'pl.lapp'
-- network config
cfg = {
  scale = 54,
  nmap = 1, -- number of feature maps
  msize = {21}, -- feature matp size
  bpc = {6}, -- number of default boxes per cell
  aratio = {1, '1', 2, 1/2, 3, 1/3},  -- aspect retio
  variance = {0.1, 0.1, 0.2, 0.2, 0.2},
  steps = {16},
  imgshape = 336, -- input image size
  classes = 2
}

opts = {}
-- parameters and others config
function opts.parse(arg)
  opt = lapp [[
    Command line options:
    Training Related:
    --lr         (default 1e-5)                    learning rate
    --momentum   (default 0.9)                     momentum
    --wd         (default 0.0005)                  weight decay
    --snap       (default 2000)                   snapshot
    --iter       (default 120000)                   iterations
    --batchsize  (default 16)                      mini-batch size
    --test       (default 10000)                   test span
    --disp       (default 100)                     display span
    --output     (default ./output)                output directory
    --root       (default data)          dataset root directory
    --cache      (default ./cache)                 cache file directory
    --gpu        (default 1)                       gpu id
    --pretrain   (default nil)                     pretrain model root directory
    --seed       (default 1)
  ]]
  return opt
end
