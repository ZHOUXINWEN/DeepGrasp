require 'nn'
require 'cudnn'
require 'cutorch'
require 'torch'
require 'cunn'--]]
loc_loss_func = nn.SmoothL1Criterion():cuda()
loc_loss_func.sizeAverage = false
conf_loss_func = nn.CrossEntropyCriterion():cuda()
conf_loss_func.nll.sizeAverage = false
------------------
--po = trainGT['pcd1021r.png']

function pointsToLabel(prior_bboxes)    --utils
  local label = torch.Tensor(prior_bboxes:size(1),5)
  label[{{}, {1}}] = torch.add(prior_bboxes[{{}, {1}}], prior_bboxes[{{}, {5}}]):div(2)    -- center x
  label[{{}, {2}}] = torch.add(prior_bboxes[{{}, {2}}], prior_bboxes[{{}, {6}}]):div(2)    -- center y
  label[{{}, {3}}] = torch.sqrt(torch.add(torch.pow(torch.csub(prior_bboxes[{{}, {3}}],prior_bboxes[{{}, {5}}]),2),torch.pow(torch.csub(prior_bboxes[{{}, {4}}],prior_bboxes[{{}, {6}}]),2)))--h
  label[{{}, {4}}] = torch.sqrt(torch.add(torch.pow(torch.csub(prior_bboxes[{{}, {7}}],prior_bboxes[{{}, {5}}]),2),torch.pow(torch.csub(prior_bboxes[{{}, {8}}],prior_bboxes[{{}, {6}}]),2)))--w
  label[{{}, {5}}] = -math.deg(torch.atan(torch.cdiv(torch.csub(prior_bboxes[{{}, {6}}],prior_bboxes[{{}, {8}}]),torch.csub(prior_bboxes[{{}, {5}}],prior_bboxes[{{}, {7}}]))))
  return label
end

function pointCrop(prior_bboxes,Xran,Yran,cfg)   --utils
  local num_boxes = prior_bboxes:size(1)
  local num_points = prior_bboxes:size(2)
  local Croped = torch.Tensor(num_boxes,num_points):copy(prior_bboxes)
  local XOriN=110+Xran
  local YOriN=110+Yran
     
  Croped[{{},{1}}]:csub(XOriN)
  Croped[{{},{3}}]:csub(XOriN)
  Croped[{{},{5}}]:csub(XOriN)
  Croped[{{},{7}}]:csub(XOriN)

  Croped[{{},{2}}]:csub(YOriN)
  Croped[{{},{4}}]:csub(YOriN)
  Croped[{{},{6}}]:csub(YOriN)
  Croped[{{},{8}}]:csub(YOriN)
   
  local row_mask = torch.gt(torch.lt(Croped,0)+torch.gt(Croped,cfg.imgshape),0):sum(2)    -- the rows  zero remain byteTensor
  local remain_num = torch.squeeze(row_mask,2):eq(0):nonzero()                                    -- longTensor of mask
  --print(row_mask:eq(0))
  --print(remain_num)
  local inside_num = row_mask:eq(0):sum()
  --print(inside_num)
  if inside_num == 0 then
         print('out of bound')         
         return false, prior_bboxes
  else
	 local tmp_points = Croped:index(1,torch.squeeze(remain_num,2))
         return true, tmp_points
  end

end

function LabelRotateTensor(label, r, center)
  local labelR = torch.Tensor(label:size()):copy(label)
  local radius = torch.sqrt(torch.pow((label[{{},{1}}]-center.x), 2)+torch.pow((label[{{},{2}}]-center.y), 2) )
  --print(radius)
  cosor = torch.cdiv((label[{{},{1}}]-center.x), radius)
  sinor = torch.cdiv((label[{{},{2}}]-center.y), radius)
  cosr = math.cos(r)
  sinr = math.sin(r)

  labelR[{{},{5}}] = labelR[{{},{5}}] + r*180/math.pi

  cosRot = cosor*cosr+sinor*sinr
  sinRot = sinor*cosr-cosor*sinr

  local lowerB = torch.lt(labelR[{{},{5}}], -90)
  local higherB = torch.gt(labelR[{{},{5}}], 90)
  labelR[{{},{5}}] = torch.csub(torch.add(labelR[{{},{5}}]:float(), 180*lowerB:float()), 180*higherB:float())

  --print(cosRot,sinRot)

  labelR[{{},{1}}]=torch.cmul(radius,cosRot)+center.x
  labelR[{{},{2}}]=torch.cmul(radius,sinRot)+center.y

  local points_rotated = labelToPointsTensor(labelR)

  local row_mask = torch.gt(torch.lt(points_rotated, 0)+torch.gt(points_rotated, cfg.imgshape),0):sum(2)    -- the rows  zero remain byteTensor, if one of the coordinates across the boundary note 1
  local remain_num = torch.squeeze(row_mask,2):eq(0):nonzero()                                    -- longTensor of mask
  --print(row_mask:eq(0))
  --print(remain_num)
  local inside_num = row_mask:eq(0):sum()
  --print(inside_num)
  if inside_num == 0 then
         print('over rotated')
         return false, label
  else
         local tmp_label = labelR:index(1,torch.squeeze(remain_num, 2))
         return true, tmp_label
  end
end


function labelToPoints(label)
  local w = label[4]
  local h = label[3]
  local uw = {}
  uw.x = math.cos(label[5])
  uw.y = math.sin(label[5])
  local uh = {}
  uh.x = math.sin(label[5])
  uh.y = -math.cos(label[5])
  local w1 = {x=uw.x*w/2,y=uw.y*w/2}
  local w2 = {x=-uw.x*w/2,y=-uw.y*w/2}
  local h1 = {x=uh.x*h/2,y=uh.y*h/2}
  local h2 = {x=-uh.x*h/2,y=-uh.y*h/2}
  local center = {x=label[1],y=label[2]}
  local points={}
  points.x = torch.Tensor(4)
  points.y = torch.Tensor(4)
  points.x=torch.Tensor({center.x+w1.x+h1.x,center.x+w1.x+h2.x,center.x+w2.x+h1.x,center.x+w2.x+h2.x})
  points.y=torch.Tensor({center.y+w1.y+h1.y,center.y+w1.y+h2.y,center.y+w2.y+h1.y,center.y+w2.y+h2.y})
  return points
end

function labelToPointsTensor(label)
  local points = torch.Tensor(label:size(1), 8)
  local center_x = label[{{},{1}}]
  local center_y = label[{{},{2}}]
  local h = label[{{},{3}}]
  local w = label[{{},{4}}]
  local theta = label[{{},{5}}]*math.pi/180     --  degree to radian
  -- x<0
  points[{{},{1}}] = center_x + torch.cmul(torch.cos(theta), w)/2 + torch.cmul(torch.sin(theta), h)/2
  points[{{},{2}}] = center_y - torch.cmul(torch.sin(theta), w)/2 + torch.cmul(torch.cos(theta), h)/2

  points[{{},{3}}] = center_x - torch.cmul(torch.cos(theta), w)/2 + torch.cmul(torch.sin(theta), h)/2
  points[{{},{4}}] = center_y + torch.cmul(torch.sin(theta), w)/2 + torch.cmul(torch.cos(theta), h)/2

  points[{{},{5}}] = center_x - torch.cmul(torch.cos(theta), w)/2 - torch.cmul(torch.sin(theta), h)/2
  points[{{},{6}}] = center_y + torch.cmul(torch.sin(theta), w)/2 - torch.cmul(torch.cos(theta), h)/2

  points[{{},{7}}] = center_x + torch.cmul(torch.cos(theta), w)/2 - torch.cmul(torch.sin(theta), h)/2
  points[{{},{8}}] = center_y - torch.cmul(torch.sin(theta), w)/2 - torch.cmul(torch.cos(theta), h)/2

  return points
end

function TensorToPoints(tensor)
        local tmp_tensor = tensor:view(4,2)
	local points = {}
        points.x = torch.Tensor(4):copy(torch.squeeze(tmp_tensor[{{},{1}}]))
        points.y = torch.Tensor(4):copy(torch.squeeze(tmp_tensor[{{},{2}}]))
	return points
end


function ImageCrop(img,Xran,Yran,cfg)
  local XOriN=110+Xran
  local YOriN=110+Yran
  local ImageCroped=image.crop(img,XOriN,YOriN,XOriN+cfg.imgshape,YOriN+cfg.imgshape) --the box of crop should be lower than 350 pixels
  return ImageCroped
end
---------------


function EncodeBBox(bbox, prior_bboxes, variance)                            --adapted to grasp
  local gt_offset = torch.Tensor(prior_bboxes:size())

  local prior_center_x = prior_bboxes[{{}, {1}}]
  local prior_center_y = prior_bboxes[{{}, {2}}]
  local prior_height = prior_bboxes[{{}, {3}}]  --h
  local prior_width = prior_bboxes[{{}, {4}}]   --w
  local prior_angle = prior_bboxes[{{}, {5}}]

  local bbox_center_x = bbox[{{}, {1}}]                                      --gt
  local bbox_center_y = bbox[{{}, {2}}]
  local bbox_height = bbox[{{}, {3}}]
  local bbox_width = bbox[{{}, {4}}]
  local bbox_angle = bbox[{{}, {5}}]

  local encode_bbox_xmin, encode_bbox_ymin, encode_bbox_xmax, encode_bbox_ymax

  if variance == nil then
    gt_offset[{{},{1}}] = torch.cdiv(torch.csub(bbox_center_x, prior_center_x), prior_width)
    gt_offset[{{},{2}}] = torch.cdiv(torch.csub(bbox_center_y, prior_center_y), prior_height)
    gt_offset[{{},{3}}] = torch.log(torch.cdiv(bbox_height, prior_height))
    gt_offset[{{},{4}}] = torch.log(torch.cdiv(bbox_width, prior_width))
    gt_offset[{{},{5}}] = torch.div(torch.csub(bbox_angle, prior_angle), 30)
  else
    gt_offset[{{},{1}}] = torch.cdiv(torch.csub(bbox_center_x, prior_center_x), prior_width) / variance[1]
    gt_offset[{{},{2}}] = torch.cdiv(torch.csub(bbox_center_y, prior_center_y), prior_height) / variance[2]
    gt_offset[{{},{3}}] = torch.log(torch.cdiv(bbox_height, prior_height)) / variance[3]
    gt_offset[{{},{4}}] = torch.log(torch.cdiv(bbox_width, prior_width)) / variance[4]
    gt_offset[{{},{5}}] = torch.div(torch.csub(bbox_angle, prior_angle), 30) / variance[5]
  end
  return gt_offset
end


function DecodeBBox(bbox, prior_bboxes, variance)    --adapted to grasp 
  local decode_xyhw = torch.Tensor(prior_bboxes:size())
  local prior_center_x = prior_bboxes[{{}, {1}}]
  local prior_center_y = prior_bboxes[{{}, {2}}]
  local prior_height = prior_bboxes[{{}, {3}}]
  local prior_width = prior_bboxes[{{}, {4}}]
  local prior_angle = prior_bboxes[{{}, {5}}]

  local decode_bbox_center_x, decode_bbox_center_y, decode_bbox_width, decode_bbox_height
  if variance == nil then
    decode_xyhw[{{},{1}}] = torch.add(torch.cmul(bbox[{{}, {1}}], prior_width), prior_center_x)
    decode_xyhw[{{},{2}}] = torch.add(torch.cmul(bbox[{{}, {2}}], prior_height), prior_center_y)
    decode_xyhw[{{},{3}}] = torch.cmul(torch.exp(bbox[{{}, {3}}]), prior_height)
    decode_xyhw[{{},{4}}] = torch.cmul(torch.exp(bbox[{{}, {4}}]), prior_width)
    decode_xyhw[{{},{5}}] = torch.add(torch.mul(bbox[{{}, {5}}], 30), prior_angle)
  else
    decode_xyhw[{{},{1}}] = torch.add(torch.cmul(bbox[{{}, {1}}], prior_width* variance[1]), prior_center_x)
    decode_xyhw[{{},{2}}] = torch.add(torch.cmul(bbox[{{}, {2}}], prior_height* variance[2]), prior_center_y)
    decode_xyhw[{{},{3}}] = torch.cmul(torch.exp(bbox[{{}, {3}}]* variance[3]), prior_height)
    decode_xyhw[{{},{4}}] = torch.cmul(torch.exp(bbox[{{}, {4}}]* variance[4]), prior_width )
    decode_xyhw[{{},{5}}] = torch.add(torch.mul(bbox[{{}, {5}}], 30)* variance[5], prior_angle)
  end
  return decode_xyhw
end

function GetPriorBBoxes(cfg)
   local x_centers = torch.range(0, cfg.msize[1]-1)
   x_centers:mul(cfg.steps[1]):add(cfg.steps[1]/2)


   local y_centers = torch.range(0, cfg.msize[1]-1)
   y_centers:mul(cfg.steps[1]):add(cfg.steps[1]/2)

   local a_centers = torch.range(0, cfg.bpc[1]-1)
   a_centers:mul(60):add(-60)   

   local output_view = torch.Tensor(cfg.bpc[1], 5, cfg.msize[1], cfg.msize[1])

   local xc = output_view:select(2, 1)	
   local yc = output_view:select(2, 2)
   local w = output_view:select(2, 3)
   local h = output_view:select(2, 4)
   local a = output_view:select(2, 5)

   xc:copy(x_centers:view(1, 1, cfg.msize[1]):expand(cfg.bpc[1], cfg.msize[1], cfg.msize[1]))
   yc:copy(y_centers:view(1, cfg.msize[1], 1):expand(cfg.bpc[1], cfg.msize[1], cfg.msize[1]))
   w:fill(cfg.scale)
   h:fill(cfg.scale)
   a:copy(a_centers:view(cfg.bpc[1], 1, 1):expand(cfg.bpc[1], cfg.msize[1], cfg.msize[1]))
    --print(output_view)
   local tmp_prior_box = output_view:view(-1,cfg.msize[1], cfg.msize[1]):transpose(1,2):transpose(2,3)
   
   tmp_prior_box = tmp_prior_box:contiguous()
   --tmp_prior_box = tmp_prior_box
   --print(   tmp_prior_box:isContiguous() )
   --local prior_box = 
   --print(tmp_prior_box:view(torch.LongStorage{600, 5}))
   return tmp_prior_box:view(-1, 5)
end


function EncodeLocPrediction(loc_preds, prior_bboxes, gt_locs, match_indices, cfg)
  local loc_gt_data = EncodeBBox(gt_locs:index(1, match_indices:nonzero():view(-1)), prior_bboxes:index(1, match_indices:nonzero():view(-1)), cfg.variance)
  local loc_pred_data = loc_preds:index(1, match_indices:nonzero():view(-1))
  -- if cfg.variance ~= nil then
  --   loc_pred_data[{{}, {1}}]:div(cfg.variance[1])
  --   loc_pred_data[{{}, {2}}]:div(cfg.variance[2])
  --   loc_pred_data[{{}, {3}}]:div(cfg.variance[3])
  --   loc_pred_data[{{}, {4}}]:div(cfg.variance[4])
  -- end
  return loc_gt_data, loc_pred_data
end

function EncodeConfPrediction(conf_preds, match_indices, neg_indices)
  local num_matches = match_indices:nonzero():size(1)
  local num_samples = num_matches + neg_indices:size(1)
  local conf_gt_data = torch.zeros(num_samples)
  conf_gt_data[{{1,num_matches}}] = match_indices[match_indices:ne(0)]
  conf_gt_data[{{num_matches+1,-1}}] = 1
  --print(conf_gt_data)
  local match_preds = conf_preds:index(1, match_indices:nonzero():view(-1))
  local neg_preds = conf_preds:index(1, match_indices:eq(0):nonzero():view(-1)):index(1, neg_indices)
  local conf_pred_data = torch.cat(match_preds, neg_preds, 1)
  return conf_gt_data, conf_pred_data
end

function BBoxSize(bbox)
  local sizes = torch.zeros(bbox:size(1))
  local idx = torch.cmul(bbox[{{}, {3}}]:gt(bbox[{{}, {1}}]), bbox[{{}, {4}}]:gt(bbox[{{}, {2}}])):view(-1)
  local width = torch.csub(bbox[{{}, {3}}][idx], bbox[{{}, {1}}][idx])
  local height = torch.csub(bbox[{{}, {4}}][idx], bbox[{{}, {2}}][idx])
  sizes[idx] = torch.cmul(width, height):float()
  return sizes
end
   
function MatchingBBoxes(prior_bboxes, gt_bboxes)                       --adapted to grasp
  local match_indices = torch.zeros(prior_bboxes:size(1))
  local match_overlaps = torch.zeros(prior_bboxes:size(1))
  local gt_locs = torch.zeros(prior_bboxes:size())
  --print(prior_bboxes:size())

  for i = 1, gt_bboxes:size(1) do
      local W 
      if gt_bboxes[i][1] <= 0 then
            W = 1
      elseif gt_bboxes[i][1] >= 320 then
            W = 20
      else
            W = math.ceil(gt_bboxes[i][1]/16)
      end

      local H
      if gt_bboxes[i][2] <= 0 then
            H = 1
      elseif gt_bboxes[i][2] >= 320 then
            H = 20
      else
            H = math.ceil(gt_bboxes[i][2]/16)
      end

      local A
      if gt_bboxes[i][5] == -90 then
             A = 1
      elseif gt_bboxes[i][5] == 90 then
             A = 3
      else  
             A = math.ceil((gt_bboxes[i][5]+90)/60)
      end
      --print(gt_bboxes[i][5],A)
      --print(H,W,A)
      local BT = torch.zeros(20,20,3)
      if H == H and W == W and A == A then
      	BT[H][W][A] = 1
      	local idx =torch.gt(BT,0)
        idx:view(prior_bboxes:size(1))
        match_indices[idx] = 2       
        gt_locs[{{}, {1}}][idx] = gt_bboxes[i][1]  --idx = (H-1)*21*6+ (W-1)*6 +A
        gt_locs[{{}, {2}}][idx] = gt_bboxes[i][2]
        gt_locs[{{}, {3}}][idx] = gt_bboxes[i][3]
        gt_locs[{{}, {4}}][idx] = gt_bboxes[i][4]
        gt_locs[{{}, {5}}][idx] = gt_bboxes[i][5]
      end
  end
  return match_indices, gt_locs
end


function MineHardExamples(conf_preds, match_indices,cfg)     -- adapted to grasp
  local num_matches = match_indices:nonzero():size(1)                    -- number of prior boxes which has matched with gt
  local num_sel = math.min(num_matches * cfg.NegRatio, match_indices:eq(0):sum())   -- set the num of selected
  -- calc loss
  -- print(match_indices:eq(0):sum())
  -- print(conf_preds[{{},{1}}]:size())
  local neg_loss = -torch.log(torch.cdiv(
  torch.exp(conf_preds[{{},{1}}][match_indices:eq(0)]),
  torch.exp(conf_preds):sum(2)[match_indices:eq(0)]))          -- calculate negative log likelyhood for negative examples which are not matched with gt(the likelyhood of being background)
  -- get topk
  -- print(match_indices:eq(0)) 
  --print(neg_loss:size())
  local topk, neg_indices = neg_loss:topk(num_sel, true) 
  --print(conf_preds[{{},{1}}][match_indices:eq(0)])  
  return neg_indices
end


function GetGradient(loc_gt_data, loc_pred_data, conf_gt_data, conf_pred_data, match_indices, neg_indices, prior_bboxes_shape, cfg)
  local num_matches = match_indices:nonzero():size(1)
  local match_indices_tensor = match_indices:nonzero():view(-1)
  local not_match_indices_tensor = match_indices:eq(0):nonzero():view(-1)
  local loc_loss = loc_loss_func:forward(loc_pred_data:cuda(), loc_gt_data:cuda()) / num_matches
  local loc_grad = (loc_loss_func:backward(loc_pred_data:cuda(), loc_gt_data:cuda()) / num_matches):float()
  local conf_loss = conf_loss_func:forward(conf_pred_data:cuda(), conf_gt_data:cuda()) / num_matches
  local conf_grad = (conf_loss_func:backward(conf_pred_data:cuda(), conf_gt_data:cuda()) / num_matches):float()
  if conf_mat ~= nil then
    conf_mat:batchAdd(conf_pred_data, conf_gt_data)
  end
  local loc_dE_do = torch.zeros(prior_bboxes_shape)
  local conf_dE_do = torch.zeros(prior_bboxes_shape[1], cfg.classes)
  for i = 1, num_matches do
    loc_dE_do[match_indices_tensor[i]] = loc_grad[i]:float()
    conf_dE_do[match_indices_tensor[i]] = conf_grad[i]:float()
  end
  for i = num_matches + 1, num_matches + neg_indices:size(1) do
    local neg_index = not_match_indices_tensor[neg_indices[i-num_matches]]
    conf_dE_do[neg_index] = conf_grad[i]
  end
  return loc_dE_do, conf_dE_do, loc_loss, conf_loss
end

function MultiBoxLoss(loc_preds, conf_preds, gt_bboxes, cfg)
  local prior_bboxes = GetPriorBBoxes(cfg)
  local match_indices, gt_locs = MatchingBBoxes(prior_bboxes, gt_bboxes, cfg)
  local neg_indices = MineHardExamples(conf_preds:float(), match_indices:float(), cfg)
  local loc_gt_data, loc_pred_data = EncodeLocPrediction(loc_preds:float(), prior_bboxes, gt_locs, match_indices, cfg)
  local conf_gt_data, conf_pred_data = EncodeConfPrediction(conf_preds:float(), match_indices, neg_indices)
  return GetGradient(loc_gt_data, loc_pred_data, conf_gt_data, conf_pred_data, match_indices, neg_indices, prior_bboxes:size(), cfg)
end

function NMS(original_bboxes, original_conf, original_classes, threshold)
  local pick = {}
  local bboxes = original_bboxes:clone()
  local classes = original_classes:clone()
  local sorted_score, i = original_conf:sort(true)
  classes = classes:index(1, i)
  bboxes = bboxes:index(1, i)
  while i:dim() ~= 0 do
    local idx = i[1]
    table.insert(pick, idx)
    local overlaps = JaccardOverlap(bboxes, original_bboxes[idx]):view(-1)
    local diff_bboxes = torch.add(overlaps:lt(threshold), classes:ne(original_classes[idx])):ne(0)
    i = i[diff_bboxes]
    if i:dim() == 0 then
      break
    end
    classes = classes[diff_bboxes]
    local non_zero = diff_bboxes:nonzero()
    bboxes = bboxes:index(1, non_zero:reshape(non_zero:size(1)))
  end
  pick = torch.LongTensor{pick}:reshape(#pick)
  return original_bboxes:index(1, pick), original_classes:index(1, pick), original_conf:index(1, pick)
end

function Detect(model, imgs, nms_threshold, conf_threshold, cfg)
  if imgs:dim() ~= 4 then
    imgs = imgs:reshape(1, imgs:size(1), imgs:size(2), imgs:size(3))
  end
  local outputs = model:forward(imgs:cuda())
  local conf_preds = outputs[2]:float()
  local all_bboxes = {}
  local all_classes = {}
  local all_scores = {}
  local prior_bboxes = GetPriorBBoxes(cfg)
  for i = 1, imgs:size(1) do
    local loc_preds = DecodeBBox(outputs[1][i]:float(), prior_bboxes, cfg.variance)

    local softmax_conf = nn.SoftMax():forward(conf_preds[i]):view(-1,cfg.classes)
    local conf, cls = softmax_conf:narrow(2,2,cfg.classes-1):max(2)
    conf = conf:view(-1)
    local idx = conf:ge(conf_threshold)
    if idx:sum() ~= 0 then
      cls = cls[idx]
      conf = conf[idx]
      local non_zero = idx:nonzero()
      loc_preds = loc_preds:index(1, non_zero:reshape(non_zero:size(1)))
      local bboxes, classes, score = NMS(loc_preds, conf:view(-1), cls:view(-1), nms_threshold)
      bboxes[{{}, {1}}] = torch.cmax(bboxes[{{}, {1}}], 0)
      bboxes[{{}, {2}}] = torch.cmax(bboxes[{{}, {2}}], 0)
      bboxes[{{}, {3}}] = torch.cmin(bboxes[{{}, {3}}], 1)
      bboxes[{{}, {4}}] = torch.cmin(bboxes[{{}, {4}}], 1)
      table.insert(all_bboxes, bboxes)
      table.insert(all_classes, classes)
      table.insert(all_scores, score)
    else
      local dammy_tensor = torch.Tensor{0}
      table.insert(all_bboxes, dammy_tensor)
      table.insert(all_classes, dammy_tensor)
      table.insert(all_scores, dammy_tensor)
    end
  end
  return all_bboxes, all_classes, all_scores
end

function DrawRect(img, box, cls, index2class)
  for i = 1, box:size(1) do
    img = image.drawRect(img, box[i][1], box[i][2], box[i][3], box[i][4])
    img = image.drawText(img, index2class[cls[i]], box[i][1], box[i][2], {color={255,255,255}, bg={0,0,255}, size=1})
  end
  return img
end

function VDetect(model, imgs, cfg)
  if imgs:dim() ~= 4 then
    imgs = imgs:reshape(1, imgs:size(1), imgs:size(2), imgs:size(3))
  end
  --local timer = torch.Timer()
  local outputs = model:forward(imgs:cuda())
  --print('Time elapsed for detection ' .. timer:time().real .. ' seconds')
  local conf_preds = outputs[2]:float()
  local all_bboxes = {}
  local all_scores = {}
  local prior_bboxes = GetPriorBBoxes(cfg)
  local conf 
  local loc_preds
  for i = 1, imgs:size(1) do
    --print(outputs[1])
    loc_preds = DecodeBBox(outputs[1][i]:float(), prior_bboxes, cfg.variance)
    local softmax_conf = nn.SoftMax():forward(conf_preds[i]):view(-1,cfg.classes)
    conf = softmax_conf[{{},{2}}]
    --print(torch.squeeze(softmax_conf[{{},{2}}]:gt(0.5),2))--  -- the max confidence prediction in every bbox 
    --print(softmax_conf[{{},{2}}]:gt(0.5):nonzero())
    local idx = torch.squeeze(softmax_conf[{{},{2}}]:gt(0.5),2):nonzero()

    if idx:sum() ~= 0 then
      conf = conf:index(1,torch.squeeze(idx,2))
      local non_zero = idx:nonzero()
      loc_preds = loc_preds:index(1,torch.squeeze(idx,2))
      return conf,loc_preds
      --print(loc_preds, conf)
    else
      --print('the match prediction does not exist')
      return false, false
    end
  end  -- print(conf,loc_preds)
end

function Top1Detect(model, imgs, cfg)
  if imgs:dim() ~= 4 then
    imgs = imgs:reshape(1, imgs:size(1), imgs:size(2), imgs:size(3))
  end
  -- local timer = torch.Timer()
  local outputs = model:forward(imgs:cuda())
  -- print('Time elapsed for detection ' .. timer:time().real .. ' seconds')
  local conf_preds = outputs[2]:float()
  local prior_bboxes = GetPriorBBoxes(cfg)
  local conf 
  local loc_preds

  for i = 1, imgs:size(1) do
    -- print(outputs[1])
    loc_preds = DecodeBBox(outputs[1][i]:float(), prior_bboxes, cfg.variance)
    local softmax_conf = nn.SoftMax():forward(conf_preds[i]):view(-1,cfg.classes)
    conf = softmax_conf[{{},{2}}]
    --print(conf)
    local idx = torch.squeeze(softmax_conf[{{},{2}}]:gt(0.5),2):nonzero()       --
    --      maxconf ,maxid = torch.squeeze(softmax_conf[{{},{2}}]:gt(0.5),2)
    local maxconf ,maxid = torch.max(softmax_conf[{{},{2}}],1)
    --print(maxconf, maxid)


    conf = conf:index(1,torch.squeeze(maxid,2))
    loc_preds = loc_preds:index(1,torch.squeeze(maxid,2))

    return conf, loc_preds
  end
end

-------------

function isInRect(point,vertex)
  local isVertical = false
  for t=1,3 do
    if vertex.x[t] == vertex.x[t+1] then
      isVertical = t
    end
  end
  if not isVertical then     -- if the box is not vertical
    local k = {}
    for t = 1,3 do
      k[t] = (vertex.y[t+1]-vertex.y[t])/(vertex.x[t+1]-vertex.x[t]); 
    end
    k[4] = (vertex.y[1]-vertex.y[4])/(vertex.x[1]-vertex.x[4]);
    local isUp = {}
    for t = 1,4 do
      isUp[t] = point.y - vertex.y[t] -k[t]*(point.x - vertex.x[t]);
    end
    if isUp[1]*isUp[3]<0 and isUp[2]*isUp[4]<0 then
      return true
    else
      return false
    end
  else
    --print(isVertical,(isVertical+2)%4)
    local next_idx
    if isVertical == 2 then
          next_idx = 4
    else    
          next_idx = (isVertical+2)%4
    end
    if (point.x - vertex.x[isVertical])*(point.x - vertex.x[next_idx])<0 then
      if (point.y - vertex.y[isVertical])*(point.y - vertex.y[next_idx])<0 then
        return true
      end
      return false
    end
    return false
  end
end

function computeIOU(rec1,rec2)
  local xmin = math.min(rec1.x[1],rec1.x[2],rec1.x[3],rec1.x[4],rec2.x[1],rec2.x[2],rec2.x[3],rec2.x[4])
  local ymin = math.min(rec1.y[1],rec1.y[2],rec1.y[3],rec1.y[4],rec2.y[1],rec2.y[2],rec2.y[3],rec2.y[4])
  local xmax = math.max(rec1.x[1],rec1.x[2],rec1.x[3],rec1.x[4],rec2.x[1],rec2.x[2],rec2.x[3],rec2.x[4])
  local ymax = math.max(rec1.y[1],rec1.y[2],rec1.y[3],rec1.y[4],rec2.y[1],rec2.y[2],rec2.y[3],rec2.y[4])
  local mask = torch.Tensor(math.ceil(ymax)-math.ceil(ymin)+1,math.ceil(xmax)-math.ceil(xmin)+1):zero()
  local countMerge = 0
  local countInteraction = 0 
  for w = math.ceil(xmin),math.ceil(xmax) do
    for h = math.ceil(ymin),math.ceil(ymax) do
      local point = {x=w,y=h}
      if isInRect(point,rec1) then
        mask[{{h-math.ceil(ymin)+1},{w-math.ceil(xmin)+1}}] = mask[{{h-math.ceil(ymin)+1},{w-math.ceil(xmin)+1}}]+1
        countMerge = countMerge+1
      end
      if isInRect(point,rec2) then
        mask[{{h-math.ceil(ymin)+1},{w-math.ceil(xmin)+1}}] = mask[{{h-math.ceil(ymin)+1},{w-math.ceil(xmin)+1}}]+1
        if mask[{h-math.ceil(ymin)+1,w-math.ceil(xmin)+1}] == 2 then
          countInteraction = countInteraction+1
        else
          countMerge = countMerge+1
        end
      end
    end
  end
  return countInteraction/countMerge
end


function evaluate(model, testpath, img_dir, epoch, cfg)
	model:evaluate()
	local per_img_acc = {}
	local pred_num = 0
	local succ_pred = 0
        local pic_num_ND = 0  -- picture number of zero prediction
	local run_time = 0
	local timer = torch.Timer()
        local PPPI = 0
        local NO_MATCH = 0
	for  i,img_name in pairs(testpath) do
		--img_path = paths.concat(img_dir,img_name) --
		--img = image.load(img_path)
		--[[local img  = torch.FloatTensor(3,480,640)
		local loadedimg = image.load(paths.concat(opt.root, 'cornell_dataset', 'image', img_name))    --load RGB img  
		img[{{1},{},{}}] = loadedimg[{{1},{},{}}]  
		img[{{2},{},{}}] = loadedimg[{{2},{},{}}]  
		--img[{{3},{},{}}] = loadedimg[{{3},{},{}}]  
		--print(shuffle[index])
		--local img = train_RGBD[trainpath[shuffle[index]]--]
		--[[local Dname = string.gsub(img_name,'r','d')
		img[{{3},{},{}}] = image.load(paths.concat(opt.root, 'cornell_dataset', 'depth', Dname)) 

		res = img:clone()
		res = ImageCrop(res, 0, 0,cfg)
	        local scores, boxes  = Top1Detect(model, res, cfg)
               --timer:reset()--]]
		img_path = paths.concat(img_dir,img_name)
	        
      		local starttime = os.clock()

		img = image.load(img_path)

                --print(test_po - starttime)
		res = img:clone()
		res = ImageCrop(res, 0, 0,cfg)
                --res = res
	        local scores, boxes  = Top1Detect(model, res, cfg)
                --print(boxes)
		local endtime = os.clock()
	        local tmp_time = endtime - starttime
                --print(scores, boxes)
	        if boxes == false then
                         pic_num_ND = 1 + pic_num_ND
			--print(img_name,'does not have gt bbox to match with')
		else
	        --print(boxes,scores)
                        local test_1 = os.clock()

		        gt = testgt[img_name]   -- 8 colunm
		        goodC, gt = pointCrop(gt, 0, 0,cfg)

		        gt_label = pointsToLabel(gt)
		        --print(gt_label)
		        overlap_matrix = torch.zeros(boxes:size(1),gt:size(1))
	
		        pred_ang = torch.expand(boxes[{{},{5}}],boxes:size(1),gt_label:size(1))
	        	--print(pred_ang)
		        gt_ang = torch.expand(gt_label[{{},{5}}],gt_label:size(1),boxes:size(1)):transpose(1, 2)

			--print(gt_ang)
		        local ang_matched = torch.abs(pred_ang-gt_ang):le(30)
                        local ang_matched2 = torch.abs(pred_ang-gt_ang):gt(150)
	                local pred_Tensor = labelToPointsTensor(boxes)
                        --print(pred_Tensor)
			for j =1 , boxes:size(1) do       
		        	    --
				--print(points)
				for k=1, gt:size(1) do
	  				local gt_points = TensorToPoints(gt[k])
                                        local pred_points = TensorToPoints(pred_Tensor[j])
					overlap_matrix[j][k] = computeIOU(pred_points,gt_points)
				end
			end
                       -- local test_2 = os.clock()
                        --print(test_2 - test_1)
	        	--print(overlap_matrix:ge(0.20))
	       		--print(ang_matched)	                                                  --  eq(2) means statisfy  IoU and  angle
	        	local correct_matches = (overlap_matrix:ge(0.25) + ang_matched + ang_matched2):eq(2)  -- if one pred bbox match a gt bbox its corresponding index equal to 1
			--print(correct_matches) 
                        local gt_match_numbers = torch.sum(torch.sum(correct_matches,1):ge(1))
	        	local match_numbers = torch.sum(torch.sum(correct_matches,2):ge(1))   -- how many pred bbox has its match gt bbox
	        	pred_num = pred_num + boxes:size(1)
	        	succ_pred =  succ_pred + match_numbers
                        --print(img_name,match_numbers)
                        no_match_number = gt:size(1) - gt_match_numbers
	        	per_img_acc[img_name] = match_numbers/boxes:size(1)
                        pppi = match_numbers/boxes:size(1)
		end
                NO_MATCH = NO_MATCH +no_match_number
                PPPI = PPPI + pppi
	        run_time = run_time + tmp_time
	end
        --print('PPPI is', PPPI/177)
        --print('Miss Rate is',NO_MATCH/177)
	print('Total accuracy is',succ_pred/pred_num)
	print('average run time:',run_time/#testpath)
        print('amount of pictures that do not have prediction', pic_num_ND)
	if pred_num ==0 then
        	per_img_acc['total'] = 0
	else
        	per_img_acc['total'] = succ_pred/pred_num
	end
	per_img_acc['Avgtime'] = run_time/#testpath
        per_img_acc['pic_num_ND'] = pic_num_ND
	--torch.save(paths.concat('./output/', 'testresult'..epoch..'epoch.t7'), per_img_acc)
	model:training()
        return per_img_acc
end

