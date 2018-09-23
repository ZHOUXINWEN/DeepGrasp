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
  local imgs = torch.FloatTensor(opt.batchsize, 4, cfg.imgshape, cfg.imgshape)
  local gt_bboxes = {}
  local shuffle = torch.randperm(#trainpath)
  -- training
  for i = 1, opt.iter * opt.batchsize do
    
    local img  = torch.FloatTensor(4,480,640)
    local loadedimg = image.load(paths.concat(opt.root, 'cornell_dataset', 'image', trainpath[shuffle[index]]))    --load RGB img  
    img[{{1},{},{}}] = loadedimg[{{1},{},{}}]  
    img[{{2},{},{}}] = loadedimg[{{2},{},{}}]  
    img[{{3},{},{}}] = loadedimg[{{3},{},{}}]  
    --print(shuffle[index])
    --local img = train_RGBD[trainpath[shuffle[index]]]
    local Dname = string.gsub(trainpath[shuffle[index]],'r','d')
    img[{{4},{},{}}] = image.load(paths.concat(opt.root, 'cornell_dataset', 'depth', Dname)) 
    


    local Yran=math.random(0,60)
    Yran=Yran-30   
    local Xran=math.random(0,80)
    Xran=Xran-40      
    -- get gt
    local points  = traingt[trainpath[shuffle[index]]]
    local croppedPoints =   pointCrop(points, Xran, Yran,cfg)  

    if croppedPoints ~= false then
        --print(croppedPoints:size(1))
        img = ImageCrop(img, Xran, Yran, cfg)
        box = pointsToLabel(croppedPoints) 
    else
        print('out of bound')
        if  Yran < 0 then
        	croppedPoints = pointCrop(points, Xran, 30, cfg)
        	img = ImageCrop(img, Xran, 30, cfg)
        else
            	croppedPoints = pointCrop(points, Xran, -30, cfg)
        	img = ImageCrop(img, Xran, -30, cfg)           
        end
        box = pointsToLabel(croppedPoints) 
    end

    local r = (rand_r - 0.5)*math.pi/6

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

    -- 

    -- learning rate decay
    --[[if i / opt.batchsize == 21000 or i / opt.batchsize == 120000 then
      opt_conf.learningRate = opt_conf.learningRate * 0.1
    end--]]

    -- index reset,evaluate and save model
    if index == #trainpath then
      epoch = epoch + 1
      print(epoch..'th epoch finish')

      if epoch%5 == 0 then
          local per_img_acc = evaluate(model,testpath,'/home/svc3/DeepGrasp/data/cornell_dataset/image', epoch, cfg)
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
          saved['opt'] = opt

          torch.save(paths.concat(opt.output, epoch..'epoch.t7'), saved)
          collectgarbage()
          local save_times = epoch/5
          if save_times > 1 then
         	 if accuracy[save_times-1] > accuracy[save_times] and ND_num[save_times-1] < ND_num[save_times] then    
                     print('------------------------------')
                     print('sign of overfitting') 
                     print('------------------------------')
                     --opt_conf.learningRate = opt_conf.learningRate/math.sqrt(10)
                 end
          end
      end
      if epoch %30 == 0 then

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

      --[[if epoch %60 == 0 and epoch < 181 then
          opt_conf.learningRate = opt_conf.learningRate/math.sqrt(10)
      end--]]

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

function ImageVFilp(img,label)
    local resimg = image.vflip(img)
    label[{{},{5}}] = - label[{{},{5}}]
    label[{{},{2}}] = 320 - label[{{},{2}}]

    return resimg,label
end

function ImageHFilp(img,label)
    local resimg = image.hflip(img)
    label[{{},{5}}] = - label[{{},{5}}]
    label[{{},{1}}] = 320 - label[{{},{1}}]

    return resimg,label
end
