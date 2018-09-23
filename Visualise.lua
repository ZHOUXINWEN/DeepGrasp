model_path = '/home/zxw/DeepGrasp/output/320b_12DGssdResAng448_v501/180epoch.t7'
img_dir ='/home/zxw/DeepGrasp/data/cornell_dataset/image'
torch.setdefaulttensortype('torch.FloatTensor')
-- _=dofile('ssd.lua')
--[[
require 'cudnn'
require 'cunn'
require 'nn'
--]]
dofile('cfg320.lua')
dofile('utils320.lua')
require 'image'

testgt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Test.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Test.t7')
--print(testpath[1])

result = torch.load(model_path)
model = result.model:cuda()
print(result['TestResult']['total'])

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

function TensorToPoints(tensor)
        local tmp_tensor = tensor:view(4,2)
	local points = {}
        points.x = torch.Tensor(4):copy(torch.squeeze(tmp_tensor[{{},{1}}]))
        points.y = torch.Tensor(4):copy(torch.squeeze(tmp_tensor[{{},{2}}]))
	return points
end

local per_img_acc = {}
local pred_num = 0
local succ_pred = 0
--[[local bizaard = {11,18,50,67,80,87,157,159,169}
for i ,idx in pairs(bizaard) do
     print(idx,testpath[idx])
     --image.display(image.load('/home/zxw/DeepGrasp/data/cornell_dataset/image/'..testpath[idx]))
end--]]

local run_time = 0
local timer = torch.Timer()


for  i,img_name in pairs(testpath) do
	img_path = paths.concat(img_dir,img_name) --
        timer:reset()
	img = image.load(img_path)
	res = img:clone()
	res = ImageCrop(res, 0, 0)
        local scores, boxes  = VDetect(model, res, 0.6, cfg)
        timer:stop()
        local tmp_time = timer:time().real
        run_time = run_time + tmp_time
        if boxes == false then
		print(img_name,'does not have gt bbox to match with')
	else
	        --print(boxes,scores)
	        gt = testgt[img_name]
	        gt = pointCrop(gt, 0, 0)
	        gt_label = pointsToLabel(gt)
	        --print(gt_label)
	        overlap_matrix = torch.zeros(boxes:size(1),gt:size(1))

	        pred_ang = torch.expand(boxes[{{},{5}}],boxes:size(1),gt_label:size(1))
	        --print(pred_ang)
	        gt_ang = torch.expand(gt_label[{{},{5}}],gt_label:size(1),boxes:size(1)):transpose(1, 2)
	
		--print(gt_ang)
	        local ang_matched = torch.abs(pred_ang-gt_ang):le(15)
	        --print(ang_matched)	
	
		for j =1 , boxes:size(1) do
	        	boxes[j][5] = math.rad(boxes[j][5])
	        	local pred_points = labelToPoints(boxes[j])
			--print(points)
			for k=1, gt:size(1) do
	  			local gt_points = TensorToPoints(gt[k])
				overlap_matrix[j][k] = computeIOU(pred_points,gt_points)
			  	--res = drawLine(res,{points.x[1], points.y[1]},{points.x[2], points.y[2]}, 2)
	  			--res = drawLine(res,{1, 255},{350, 255}, 20)
			end
		end
	        --print(overlap_matrix)                                                  --  eq(2) means statisfy  IoU and  angle
	        local correct_matches = (overlap_matrix:ge(0.25) + ang_matched):eq(2)  -- if one pred bbox match a gt bbox its corresponding index equal to 1
		--print(correct_matches) 
	        local match_numbers = torch.sum(torch.sum(correct_matches,2):ge(1))   -- how many pred bbox has its match gt bbox
	        pred_num = pred_num + boxes:size(1)
	        succ_pred =  succ_pred + match_numbers
	        per_img_acc[img_name] = match_numbers/boxes:size(1)
	end
        print(i,per_img_acc[img_name])
end
print('Total accuracy is',succ_pred/pred_num)
print('average run time:',run_time/177)
per_img_acc['total'] = succ_pred/pred_num
per_img_acc['Avgtime'] = run_time/177
torch.save(paths.concat('./output/', 'testresult50kiter.t7'), per_img_acc)--]]
--[[image.savePNG('pcd0479r.png', res)
--]]

