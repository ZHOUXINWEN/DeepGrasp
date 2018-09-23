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
  label[{{}, {5}}] = math.deg(torch.atan(torch.cdiv(torch.csub(prior_bboxes[{{}, {6}}],prior_bboxes[{{}, {8}}]),torch.csub(prior_bboxes[{{}, {5}}],prior_bboxes[{{}, {7}}]))))
  return label
end


function pointCrop(prior_bboxes,Xran,Yran)   --utils
  local num_boxes = prior_bboxes:size(1)
  local num_points = prior_bboxes:size(2)
  --local low_bound = torch.Tensor(num_boxes,num_points):fill(0)
  --local high_bound = torch.Tensor(num_boxes,num_points):fill(336)
  local Croped = torch.Tensor(num_boxes,num_points):copy(prior_bboxes)
  local XOriN=150+Xran
  local YOriN=100+Yran
     
  Croped[{{},{1}}]:csub(XOriN)
  Croped[{{},{3}}]:csub(XOriN)
  Croped[{{},{5}}]:csub(XOriN)
  Croped[{{},{7}}]:csub(XOriN)

  Croped[{{},{2}}]:csub(YOriN)
  Croped[{{},{4}}]:csub(YOriN)
  Croped[{{},{6}}]:csub(YOriN)
  Croped[{{},{8}}]:csub(YOriN)
   
  local row_mask = torch.gt(torch.lt(Croped,0)+torch.gt(Croped,336),0):sum(2)    -- the rows  zero remain byteTensor
  local remain_num = torch.squeeze(row_mask,2):eq(0):nonzero()                                    -- longTensor of mask
  --print(row_mask:eq(0))
  --print(remain_num)
  local inside_num = row_mask:eq(0):sum()
  --print(inside_num)
  if inside_num == 0 then
         return false
  else
	 local tmp_points = Croped:index(1,torch.squeeze(remain_num,2))
         return tmp_points
  end
end

function ImageCrop(img,Xran,Yran)
  local XOriN=150+Xran
  local YOriN=100+Yran
  local ImageCroped=image.crop(img,XOriN,YOriN,XOriN+336,YOriN+336) --the box of crop should be lower than 350 pixels
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
    gt_offset[{{},{5}}] = torch.div(torch.csub(bbox_angle, prior_angle), 15)
  else
    gt_offset[{{},{1}}] = torch.cdiv(torch.csub(bbox_center_x, prior_center_x), prior_width) / variance[1]
    gt_offset[{{},{2}}] = torch.cdiv(torch.csub(bbox_center_y, prior_center_y), prior_height) / variance[2]
    gt_offset[{{},{3}}] = torch.log(torch.cdiv(bbox_height, prior_height)) / variance[3]
    gt_offset[{{},{4}}] = torch.log(torch.cdiv(bbox_width, prior_width)) / variance[4]
    gt_offset[{{},{5}}] = torch.div(torch.csub(bbox_angle, prior_angle), 15) / variance[5]
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
    decode_xyhw[{{},{5}}] = torch.add(torch.mul(bbox[{{}, {5}}], 15), prior_angle)
  else
    decode_xyhw[{{},{1}}] = torch.add(torch.cmul(bbox[{{}, {1}}], prior_width* variance[1]), prior_center_x)
    decode_xyhw[{{},{2}}] = torch.add(torch.cmul(bbox[{{}, {2}}], prior_height* variance[2]), prior_center_y)
    decode_xyhw[{{},{3}}] = torch.cmul(torch.exp(bbox[{{}, {3}}]* variance[3]), prior_height)
    decode_xyhw[{{},{4}}] = torch.cmul(torch.exp(bbox[{{}, {4}}]* variance[4]), prior_width )
    decode_xyhw[{{},{5}}] = torch.add(torch.mul(bbox[{{}, {5}}], 15)* variance[5], prior_angle)
  end
  return decode_xyhw
end


function GetPriorBBoxes(cfg) --adapted to grasp
  local scale = cfg.scale or 54
  local map_num = cfg.nmap or 1
  local map_size = cfg.msize or {21}
  local img_size = cfg.imgshape or 336
  local box_per_cell = cfg.bpc or {6}
  local ar = cfg.aratio or {1, '1', 2, 1/2, 3, 1/3}
  local steps = cfg.steps or {16}
  local prior_bboxes = {}
  for k = 1, map_num do
    local step_w = steps[k]
    local step_h = steps[k]
    local tmp_prior_bboxes = torch.zeros(map_size[k], map_size[k], box_per_cell[k]*5)
    for h = 1, map_size[k] do
      for w = 1, map_size[k] do
        local center_x = ((w-1) + 0.5) * step_w
        local center_y = ((h-1) + 0.5) * step_h
        for b = 1, box_per_cell[k] do
          local box_width = scale * math.sqrt(ar[1])
          local box_height = scale / math.sqrt(ar[1])
          tmp_prior_bboxes[h][w][(b-1)*5+1] = center_x
          tmp_prior_bboxes[h][w][(b-1)*5+2] = center_y
          tmp_prior_bboxes[h][w][(b-1)*5+3] = box_height
          tmp_prior_bboxes[h][w][(b-1)*5+4] = box_width
          tmp_prior_bboxes[h][w][(b-1)*5+5] = -75+30*(b-1)
        end
      end
    end
    table.insert(prior_bboxes, tmp_prior_bboxes:view(-1, 5))
  end
  return nn.JoinTable(1):forward(prior_bboxes)
end   -- output : torch.doubletensor (21*21*6)x5


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
      local W = math.ceil(gt_bboxes[i][1]/16)
      local H = math.ceil(gt_bboxes[i][2]/16)
      local A
      if gt_bboxes[i][5] == -90 then
             A = 1
      elseif gt_bboxes[i][5] == 90 then
             A =6
      else  
             A = math.ceil(gt_bboxes[i][5]/30)+3
      end
      --print(gt_bboxes[i][5],A)
      --print(H,W,A)
      local BT = torch.zeros(21,21,6)  
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
  return match_indices, gt_locs
end


function MineHardExamples(conf_preds, match_indices)     -- adapted to grasp
  local num_matches = match_indices:nonzero():size(1)                    -- number of prior boxes which has matched with gt
  local num_sel = math.min(num_matches * 1, match_indices:eq(0):sum())   -- set the num of selected
  -- calc loss
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
  local neg_indices = MineHardExamples(conf_preds:float(), match_indices:float())
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
    local conf, cls = softmax_conf:narrow(2,2,cfg.classes-1):max(2)   -- the max confidence prediction in every bbox 
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

function VDetect(model, imgs, conf_threshold, cfg)
  if imgs:dim() ~= 4 then
    imgs = imgs:reshape(1, imgs:size(1), imgs:size(2), imgs:size(3))
  end
  local outputs = model:forward(imgs:cuda())
  local conf_preds = outputs[2]:float()
  local all_bboxes = {}
  local all_scores = {}
  local prior_bboxes = GetPriorBBoxes(cfg)
  local conf 
  local loc_preds
  for i = 1, imgs:size(1) do
    loc_preds = DecodeBBox(outputs[1][i]:float(), prior_bboxes, cfg.variance)
    local softmax_conf = nn.SoftMax():forward(conf_preds[i]):view(-1,cfg.classes)
    conf = softmax_conf[{{},{2}}]:view(-1)   -- the max confidence prediction in every bbox 
    local idx = conf:ge(conf_threshold)
    if idx:sum() ~= 0 then
      cls = cls[idx]
      conf = conf[idx]
      local non_zero = idx:nonzero()
      loc_preds = loc_preds:index(1, non_zero:reshape(non_zero:size(1)))
    else
      print('the match prediction does not exist')
    end
  end
  return conf,loc_preds
end



function DrawRect(img, box, cls, index2class)
  for i = 1, box:size(1) do
    img = image.drawRect(img, box[i][1], box[i][2], box[i][3], box[i][4])
    img = image.drawText(img, index2class[cls[i]], box[i][1], box[i][2], {color={255,255,255}, bg={0,0,255}, size=1})
  end
  return img
end

--[[
local label = pointsToLabel(pointCrop(po,20,20))
print(label)
local prior_bboxes = GetPriorBBoxes(cfg)
local match_indice, gt_locs = MatchingBBoxes(prior_bboxes,label)
--print(match_indice,match_indice:nonzero():view(-1))
conf_preds = torch.randn(2646,2)
loc_preds = torch.randn(2646,5)
local neg_indices = MineHardExamples(conf_preds, match_indice)
--local loc_gt_data, loc_pred_data = EncodeLocPrediction(loc_preds:float(), prior_bboxes, gt_locs, match_indice, cfg)
local loc_gt_data = EncodeBBox(gt_locs:index(1, match_indice:nonzero():view(-1)), prior_bboxes:index(1, match_indice:nonzero():view(-1)), cfg.variance)
print(loc_gt_data)
print('--------------------')
print(DecodeBBox(loc_gt_data, prior_bboxes:index(1, match_indice:nonzero():view(-1)), cfg.variance))

print(MultiBoxLoss(loc_preds, conf_preds, label, cfg))--]]

