function train()
  model:training()
  local index = 1
  local mean_loss = 0
  local mean_loc_loss = 0
  local mean_conf_loss = 0
  local imgs = torch.FloatTensor(opt.batchsize, 3, cfg.imgshape, cfg.imgshape)
  local gt_bboxes = {}
  local shuffle = torch.randperm(#trainpath)
  -- training
  for i = 1, opt.iter * opt.batchsize do
    --local img = image.load(paths.concat(opt.root, 'cornell_dataset', 'image', trainpath[shuffle[index]]))
    --[[local Yran=math.random(0,60)
    Yran=Yran-30    
    local Xran=math.random(0,60)
    Xran=Xran-30      
    --img = image.scale(img, cfg.imgshape, cfg.imgshape):reshape(1, 3, cfg.imgshape, cfg.imgshape)
    -- channel RGB to BGR
    -- get gt
    local points  = traingt[trainpath[shuffle[index]]--]
    --[[local croppedPoints =   pointCrop(points, Xran, Yran)  
    if croppedPoints ~= false then
        --print(croppedPoints:size(1))
        img = ImageCrop(img,Xran,Yran)
        --img:reshape(1, 3,cfg.imgshape, cfg.imgshape)
        imgs[(index-1) % opt.batchsize + 1] = img 
        local box = pointsToLabel(croppedPoints) 
        table.insert(gt_bboxes, box)
    else
        print('out of bound')
        --print(croppedPoints:size(1))
        croppedPoints = pointCrop(points, 0, 0)
        img = ImageCrop(img, 0, 0)
        --img:reshape(1, 3,cfg.imgshape, cfg.imgshape)
        imgs[(index-1) % opt.batchsize + 1] = img 
        local box = pointsToLabel(croppedPoints) 
        table.insert(gt_bboxes, box)
    end--]]
    -- print(gt_bboxes)
    -- batch forward and backward
        local img = torch.randn(3,336,336)
        imgs[(index-1) % opt.batchsize + 1] = img 
    if index % opt.batchsize == 0 then
      gparam:zero()
      local outputs = model:forward(imgs:cuda())
      local loc_preds = outputs[1]
      local conf_preds = outputs[2]
      local loss = 0
      local sum_loc_loss = 0
      local sum_conf_loss = 0
      local loc_grads = torch.Tensor(loc_preds:size())
      local conf_grads = torch.Tensor(conf_preds:size())
      -- calc gradient
     --[[ for j = 1, opt.batchsize do
        local loc_grad, conf_grad, loc_loss, conf_loss = MultiBoxLoss(loc_preds[j], conf_preds[j], gt_bboxes[j], cfg)
        loss = loss + (loc_loss + conf_loss)
        sum_loc_loss = sum_loc_loss + loc_loss
        sum_conf_loss = sum_conf_loss + conf_loss
        loc_grads[j] = loc_grad
        conf_grads[j] = conf_grad
      end
      loss = loss / opt.batchsize
      sum_loc_loss = sum_loc_loss / opt.batchsize
      sum_conf_loss = sum_conf_loss / opt.batchsize
      mean_loss = mean_loss + loss
      mean_loc_loss = mean_loc_loss + sum_loc_loss
      mean_conf_loss = mean_conf_loss + sum_conf_loss
      -- backward
      --print(model:get(15):get(1):get(1).out)imgs:cuda()
      loc_grads = torch.randn(1,2646,5)
      conf_grads = torch.randn(1,2646,2)
      loc_grads:cuda()
      conf_grads:cuda()
      print(loc_grads:size(),conf_grads)
      model:backward(imgs:cuda(), {loc_grads, conf_grads})
      gparam:div(opt.batchsize)
      local function feval() return loss, gparam end
      -- parameter update
      optim.sgd(feval, param, opt_conf)
      gt_bboxes = {}
      collectgarbage()
    end--]]

      -- calc gradient
      --[[for j = 1, opt.batchsize do
        local loc_grad, conf_grad, loc_loss, conf_loss = MultiBoxLoss(loc_preds[j], conf_preds[j], gt_bboxes[j], cfg)
        --print(loc_loss,conf_loss)--,loc_grad,conf_grad loc_loss,torch.type(conf_loss))
        loss = loss + (loc_loss + conf_loss)
        sum_loc_loss = sum_loc_loss + loc_loss
        sum_conf_loss = sum_conf_loss + conf_loss
        loc_grads[j] = loc_grad
        conf_grads[j] = conf_grad
      end
      loss = loss / opt.batchsize
      sum_loc_loss = sum_loc_loss / opt.batchsize
      sum_conf_loss = sum_conf_loss / opt.batchsize
      mean_loss = mean_loss + loss
      mean_loc_loss = mean_loc_loss + sum_loc_loss
      mean_conf_loss = mean_conf_loss + sum_conf_loss--]]
      -- backward
      --print(torch.type(loc_grads),loc_grads:size())
      --print(torch.type(conf_grads),conf_grads:size())
      loc_grads = torch.randn(2,2646,5)
      conf_grads = torch.randn(2,2646,2)
      model:backward(imgs:cuda(), {loc_grads:cuda(), conf_grads:cuda()})
      gparam:div(opt.batchsize)
      local function feval() return loss, gparam end
      -- parameter update
      optim.sgd(feval, param, opt_conf)
      gt_bboxes = {}
      gt_labels = {}
      collectgarbage()
    end

    -- save model
    if i % (opt.snap * opt.batchsize) == 0 then
      torch.save(paths.concat(opt.output, 'model'..(i/opt.batchsize)..'iter.t7'), model)
    end
    -- learning rate decay
    if cfg.year == '2007' and (i / opt.batchsize == 80000 or i / opt.batchsize == 100000) then
      opt.lr = opt.lr * 0.1
      opt_conf.learningRate = opt.lr
    elseif cfg.year == '2012' and i / opt.batchsize == 60000 then
      opt.lr = opt.lr * 0.1
      opt_conf.learningRate = opt.lr
    end
    -- index reset
    if index == #trainpath then
      print('1 epoch finish')
      index = 0
      shuffle = torch.randperm(#trainpath)
    end
    -- next index
    index = index + 1
    -- display
    if i % (opt.disp * opt.batchsize) == 0 then
      print('iter : '..i/opt.batchsize..'   mean error : '..mean_loss/opt.disp)
      print('mean l1 loss : '..mean_loc_loss/opt.disp)
      print('mean cross entropy : '..mean_conf_loss/opt.disp)
      mean_loss = 0
      mean_loc_loss = 0
      mean_conf_loss = 0
      print(conf_mat)
      local dataNum = conf_mat.mat:sum(2)
      local tp = dataNum:narrow(1,2,cfg.classes - 1):sum()
      local fp = dataNum[1]:sum()
      print('TP : '..tp..'  FP : '..fp)
      conf_mat:zero()
    end
  end
end
