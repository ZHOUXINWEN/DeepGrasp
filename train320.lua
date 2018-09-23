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
  local ND_num = {} 

  local rand_flip = 0
  local rand_r = 1
  local center = { x = 160,
                   y = 160 }
  local epoch = 0
  local imgs = torch.FloatTensor(opt.batchsize, 3, cfg.imgshape, cfg.imgshape)
  local gt_bboxes = {}
  local shuffle = torch.randperm(#trainpath)
  -- training
  for i = 1, opt.iter * opt.batchsize do
    local img = image.load(paths.concat(opt.root, 'cornell_dataset', 'NewCrgd', trainpath[shuffle[index]]))

    local Yran=math.random(0,100)
    Yran=Yran-50  
    local Xran=math.random(0,100)
    Xran=Xran-50      
    -- get gt
    local points  = traingt[trainpath[shuffle[index]]]

    local goodC, croppedPoints = pointCrop(points, Xran, Yran,cfg)  

    while goodC == false do  
        Yran=math.random(0,100)
        Yran=Yran-50 
 
        Xran=math.random(0,100)
        Xran=Xran-50

        goodC, croppedPoints = pointCrop(points, Xran, Yran, cfg)            
    end
    img = ImageCrop(img, Xran, Yran, cfg)
    box = pointsToLabel(croppedPoints) 

    local r = (rand_r - 0.5)*math.pi/3

    img = image.rotate(img, r)
    goodR, box = LabelRotateTensor(box, r, center )

    if goodR == false then
           goodR, box = LabelRotateTensor(box, -r, center )
    end

    img = ImageColorAug(img)
    img = ImageSharpnessAug(img)

    if  rand_flip == 1 then
           img , box = ImageVFilp(img, box)  
    end--]] 

    if  rand_flip == 2 then
           img , box = ImageHFilp(img, box)
    end

    --[[if  rand_flip == 3 then
           img , box = ImageVFilp(img, box)  
           img , box = ImageHFilp(img, box)
    end--]]


    imgs[(i-1) % opt.batchsize + 1] = img           
    table.insert(gt_bboxes, box)

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
      rand_r = math.random()
    end

    -- index reset,evaluate and save model
    if index == #trainpath then
      epoch = epoch + 1
      print(epoch..'th epoch finish')

      if epoch%10 == 0 then
          local per_img_acc = evaluate(model,testpath,'/home/zxw/DeepGrasp/data/cornell_dataset/NewCrgd', epoch, cfg)
          table.insert(Totallosses, mean_loss/opt.disp)
          table.insert(Reglosses, mean_loc_loss/opt.disp)
          table.insert(Conflosses, mean_conf_loss/opt.disp)
          table.insert(accuracy, per_img_acc['total']) 
          table.insert(ND_num, per_img_acc['pic_num_ND'])
          model:clearState()           
          local saved = {}

          saved['model'] = model
          saved['TestResult'] = per_img_acc
          saved['cfg'] = cfg
          opt.lr = opt_conf.learningRate
          saved['opt'] = opt

          torch.save(paths.concat(opt.output, epoch..'epoch.t7'), saved)
          collectgarbage()
          --[[local save_times = epoch/5
          if save_times > 1 then
         	 if accuracy[save_times-1] > accuracy[save_times] and ND_num[save_times-1] < ND_num[save_times] then    
                     print('------------------------------')
                     print('sign of overfitting') 
                     print('------------------------------')
                     --opt_conf.learningRate = opt_conf.learningRate/math.sqrt(10)
                 end
          end--]]
	  if opt_conf.learningRate < 1e-5 then
              opt_conf.learningRateDecay = 0
          end
      end
      if epoch %100 == 0 then

          gnuplot.pngfigure(paths.concat(opt.output, epoch..'test.png'))
          gnuplot.plot('Accuracy',torch.Tensor(accuracy))
          gnuplot.close()

          --[[gnuplot.pngfigure(paths.concat(opt.output, epoch..'num_NDtest.png'))
          gnuplot.plot('ND_num',torch.Tensor(ND_num))
          gnuplot.close()
          --]]

          local curves ={}
          curves['Totallosses'] = Totallosses
          curves['Reglosses'] = Reglosses 
          curves['Conflosses'] = Conflosses
          curves['accuracy'] = accuracy 
          curves['ND_num'] = ND_num
          torch.save(paths.concat(opt.output, epoch..'Result.t7'), curves)

      end


      index = 0
      shuffle = torch.randperm(#trainpath)
      rand_flip = math.random(1, 3)
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

