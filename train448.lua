function train()
  model:training()
  local index = 1
  local mean_loss = 0
  local mean_loc_loss = 0
  local mean_conf_loss = 0

  local Totallosses = {}
  local Reglosses = {}
  local Conflosses = {}
  local accuracy = {}
  -- local ND_num = {} 

  local epoch = 0
  local imgs = torch.FloatTensor(opt.batchsize, 3, cfg.imgshape, cfg.imgshape)
  local gt_bboxes = {}
  local shuffle = torch.randperm(#trainpath)
  -- training
  for i = 1, opt.iter * opt.batchsize do
    local img = image.load(paths.concat(opt.root, 'cornell_dataset', 'image', trainpath[shuffle[index]]))
    local Yran=math.random(0,20)
    Yran=Yran-10   
    local Xran=math.random(0,20)
    Xran=Xran-10      
    -- get gt
    local points  = traingt[trainpath[shuffle[index]]]
    local croppedPoints =   pointCrop(points, Xran, Yran)  
    if croppedPoints ~= false then
        --print(croppedPoints:size(1))
        img = ImageCrop(img,Xran,Yran)
        --img:reshape(1, 3,cfg.imgshape, cfg.imgshape)
        img = ImageColorAug(img)
        img = ImageSharpnessAug(img)
        imgs[(index-1) % opt.batchsize + 1] = img 
        local box = pointsToLabel(croppedPoints) 
        table.insert(gt_bboxes, box)
    else
        print('out of bound')
        --print(croppedPoints:size(1))
        croppedPoints = pointCrop(points, 0, 0)
        img = ImageCrop(img, 0, 0)
        --img:reshape(1, 3,cfg.imgshape, cfg.imgshape)
        img = ImageColorAug(img)
        img = ImageSharpnessAug(img)
        imgs[(index-1) % opt.batchsize + 1] = img 
        local box = pointsToLabel(croppedPoints) 
        table.insert(gt_bboxes, box)
    end--]]
    -- batch forward and backward
    if i % opt.batchsize == 0 then
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
      for j = 1, opt.batchsize do
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
      mean_conf_loss = mean_conf_loss + sum_conf_loss--]]
      -- backward
      model:backward(imgs:cuda(), {loc_grads:cuda(), conf_grads:cuda()})
      gparam:div(opt.batchsize)
      local function feval() return loss, gparam end
      -- parameter update
      optim.sgd(feval, param, opt_conf)
      gt_bboxes = {}
      gt_labels = {}
      collectgarbage()
    end

    -- 

    -- learning rate decay
    if i / opt.batchsize == 30000 or i / opt.batchsize == 120000 then
      opt_conf.learningRate = opt_conf.learningRate * 0.1
    end

    -- index reset,evaluate and save model
    if index == #trainpath then
      epoch = epoch + 1
      print(epoch..'th epoch finish')

      if epoch%20 == 0 then
          local per_img_acc = evaluate(model,testpath,'/home/zxw/DeepGrasp/data/cornell_dataset/image',epoch)
          table.insert(Totallosses, mean_loss/opt.disp)
          table.insert(Reglosses, mean_loc_loss/opt.disp)
          table.insert(Conflosses, mean_conf_loss/opt.disp)
          table.insert(accuracy, per_img_acc['total'])

          model:clearState()           
          local saved = {}

          saved['model'] = model
          saved['TestResult'] = per_img_acc
          saved['cfg'] = cfg
          saved['opt'] = opt

          torch.save(paths.concat(opt.output, epoch..'epoch.t7'), saved)
      end
      if epoch %120 == 0 then

          gnuplot.pngfigure(paths.concat(opt.output, epoch..'test.png'))
          gnuplot.plot({'Reg Loss',torch.Tensor(Reglosses)},{'Ang Loss',torch.Tensor(Conflosses)},{'Accuracy',torch.Tensor(accuracy)})
          gnuplot.close()

          local curves ={}
          curves['Totallosses'] = Totallosses
          curves['Reglosses'] = Reglosses 
          curves['Conflosses'] = Conflosses
          curves['accuracy'] = accuracy
          torch.save(paths.concat(opt.output, epoch..'Result.t7'), curves)
      end
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
      conf_mat:zero()--]]
    end
  end
end

function ImageColorAug(img)
    local randR = torch.rand(1)*0.06+0.97
    local randG = torch.rand(1)*0.06+0.97                                                
    local randB = torch.rand(1)*0.06+0.97
    img[1]:mul(randR:float()[1])
    img[2]:mul(randG:float()[1])                              
    img[3]:mul(randB:float()[1])
    return img
end

function ImageSharpnessAug(img)
    local blurK = torch.FloatTensor(5,5):fill(1/25)
    local Cur_im_blurred = image.convolve(img,blurK,'same')
    local cur_im_residue = torch.add(img,-1,Cur_im_blurred)
    local ranSh = torch.rand(1)*1.5
    img:add(ranSh:float()[1],cur_im_residue)
    return img
end
