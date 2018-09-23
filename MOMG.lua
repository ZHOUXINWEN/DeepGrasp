require 'pl'
require 'image'
require 'optim'
require 'cutorch'
require 'gnuplot'
require 'nn'
require 'cudnn'
local matio = require 'matio'
cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')
dofile('utils320_ang45.lua')

testgt = torch.load('/home/home/DeepGrasp/cache/MOMGannotation.t7')
 testpath= torch.load('/home/home/DeepGrasp/cache/MOMGimagepath.t7') --testpath = {'pcd0188.png', 'pcd0412.png', 'pcd0449.png', 'pcd0890.png'}
img_dir = '/home/zxw/DeepGrasp/data/rgd_cropped320/'
result = torch.load('/home/zxw/DeepGrasp/output/Resnet101_RGD_20x20x4_HV_b16_lrd2_RN_leaky01_22222_FF4/830epoch.t7')
cfg = result['cfg']
--print(cfg)
model = result['model']:cuda()

model:evaluate()

local per_img_acc = {}
local pred_num = 0
local succ_pred = 0
local pic_num_ND = 0  -- picture number of zero prediction
local run_time = 0


local TP_total = 0
local Total_GT = 0

local FPPI = {}
local MissRate = {}

local Detect_Result = {}

--[[
for  i,img_name in pairs(testpath) do
		img_path = paths.concat(img_dir,img_name)

		Detect_Result[img_name] = {}
		local starttime = os.clock()

		img = image.load(img_path)

                --print(test_po - starttime)
		res = img:clone()
		--res = ImageCrop(res, 0, 0,cfg)

	        local scores, boxes  = VDetect(model, res, cfg)
		local endtime = os.clock()

                Detect_Result[img_name]['scores'] = scores
                Detect_Result[img_name]['boxes'] = boxes
end
torch.save('Detect_Result.t7',Detect_Result)--]]
		--print(#testpath)
Detect_Result = torch.load('/home/svc3/DeepGrasp/Detect_Result.t7')
f = assert(io.open('FPPIMissRate.txt',a))
for threshold = 0.05, 0.95, 0.05 do
	local TP_total = 0
	local Total_GT = 0
	local FP_total = 0
	for  i,img_name in pairs(testpath) do
		img_path = paths.concat(img_dir,img_name)


		local starttime = os.clock()

		img = image.load(img_path)


		res = img:clone()

	        --local scores, boxes  = VDetect(model, res, cfg)

		local allscores, allboxes = Detect_Result[img_name]['scores'], Detect_Result[img_name]['boxes']
                local UNdscores =torch.squeeze(allscores)
		local endtime = os.clock()
		--print(allscores, allboxes )
                local num_box = UNdscores:ge(threshold):sum()
                --print(num_box)
                --local boxes = allboxes:index(1, torch.squeeze(UNdscores:ge(0.8):nonzero()))
                --print(inddd)--,boxes:index(1,UNdscores:ge(0.8):nonzero()))
                --print(boxes)

		--local SelectedBoxes = boxes:maskedSelect(scores)
                --Detect_Result[img_name]['scores'] = scores
                --Detect_Result[img_name]['boxes'] = boxes

	        if num_box == 0 then
                        pic_num_ND = 1 + pic_num_ND
			Total_GT = Total_GT + testgt[img_name]:size(1)
		else
                        local test_1 = os.clock()
			--print(torch.squeeze(UNdscores:ge(0.8):nonzero()))
                        --print(UNdscores:ge(0.8):nonzero():view(-1))
               		local boxes = allboxes:index(1, UNdscores:ge(threshold):nonzero():view(-1))

		        gt = testgt[img_name]   -- 8 colunm
		        --goodC, gt = pointCrop(gt, 0, 0,cfg)

		        gt_label = pointsToLabel(gt)
		        --print(gt_label)
		        overlap_matrix = torch.zeros(boxes:size(1),gt:size(1))
			print(boxes:size(1),gt:size(1))
		        pred_ang = torch.expand(boxes[{{},{5}}],boxes:size(1),gt_label:size(1))
	        	--print(pred_ang)
		        gt_ang = torch.expand(gt_label[{{},{5}}],gt_label:size(1),boxes:size(1)):transpose(1, 2)

			--print(gt_ang)
		        local ang_matched = torch.abs(pred_ang-gt_ang):le(30)
                        local ang_matched2 = torch.abs(pred_ang-gt_ang):gt(150)
	                local pred_Tensor = labelToPointsTensor(boxes)
                        --print(pred_Tensor)
			for j =1 , boxes:size(1) do       
				--print(points)
				for k=1, gt:size(1) do
	  				local gt_points = TensorToPoints(gt[k])
                                        local pred_points = TensorToPoints(pred_Tensor[j])
					overlap_matrix[j][k] = computeIOU(pred_points,gt_points)
				end
			end
			--local ang_mask = ang_matched+ ang_matched2
                        local matched_overlap = torch.zeros(boxes:size(1),gt:size(1))
	                
                                                                                         --  eq(2) means statisfy  IoU and  angle
	        	local correct_matches = (overlap_matrix:ge(0.25) + ang_matched+ ang_matched2):eq(2)  -- if one pred bbox match a gt bbox its corresponding index equal to 1
			--print('correct_matches',correct_matches)

                        --print(overlap_matrix)
			--print(correct_matches)
			matched_overlap = torch.cmul(correct_matches:float(), overlap_matrix)
			local thres, index = torch.topk(matched_overlap, 1, true)     --find the best match GT for each box
                        local GT_flag = torch.zeros(gt:size(1))
			--print(matched_overlap)

                        for j = 1, thres:size(1) do--]]
				if thres[j][1] >= 0.25	and GT_flag[index[j][1]] == 0 then
					GT_flag[index[j][1]] = 1
				end					
                        end
                        --print(GT_flag:nonzero())
                        local TP = GT_flag:sum()
			local FP = boxes:size(1) - TP
                        print(TP,FP)
                        TP_total = TP_total +TP
                        FP_total = TP_total +FP
			Total_GT = Total_GT + gt:size(1)

		end
                  
	        local tmp_time = endtime - starttime
	        run_time = run_time + tmp_time
	end
	table.insert(FPPI, FP_total/#testpath)
        table.insert(MissRate, 1-(TP_total/Total_GT))
        print('for threshold '..threshold..':')
        print('FPPI',FP_total/#testpath,'MissRate',1-(TP_total/Total_GT))
        local tmp_result = threshold..','..FP_total/#testpath..','..(1-(TP_total/Total_GT))..'\n'
        f:write(tmp_result)
end
torch.save('FPPI.t7',FPPI)
torch.save('MissRate.t7',MissRate)
print(FP/96)
--print(No_MATCH/Total_GT)--]]
