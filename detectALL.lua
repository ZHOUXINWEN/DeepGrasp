require 'pl'
require 'image'
require 'optim'
require 'cutorch'
require 'gnuplot'
require 'nn'
require 'cudnn'
local matio = require 'matio'

torch.setdefaulttensortype('torch.FloatTensor')
dofile('utils320.lua')

testgt = torch.load('/home/zxw/DeepGrasp/cache/RNcornell_5_Test.t7')
testpath = {'pcd0188.png', 'pcd0412.png', 'pcd0449.png', 'pcd0890.png'} --= torch.load('/home/zxw/DeepGrasp/cache/RNpaths_5_Test.t7')
img_dir = '/home/zxw/DeepGrasp/data/cornell_dataset/image'
result = torch.load('/home/zxw/DeepGrasp/output/Resnet101_RGB_10x10x6_HV_R15deg_b16_lrd1_RN/1615epoch.t7')
cfg = result['cfg']
--print(cfg)
model = result['model']:cuda()

model:evaluate()

local per_img_acc = {}
local pred_num = 0
local succ_pred = 0
local pic_num_ND = 0  -- picture number of zero prediction
local run_time = 0
local timer = torch.Timer()

for  i,img_name in pairs(testpath) do
	img_path = paths.concat(img_dir,img_name)
      
	local starttime = os.clock()

		img = image.load(img_path)

                --print(test_po - starttime)
		res = img:clone()
		res = ImageCrop(res, 0, 0,cfg)

	        local scores, boxes  = VDetect(model, res, cfg)
                --print(boxes)
		local endtime = os.clock()

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
				--print(points)
				for k=1, gt:size(1) do
	  				local gt_points = TensorToPoints(gt[k])
                                        local pred_points = TensorToPoints(pred_Tensor[j])
					overlap_matrix[j][k] = computeIOU(pred_points,gt_points)
				end
			end
	                                                  --  eq(2) means statisfy  IoU and  angle
	        	local correct_matches = (overlap_matrix:ge(0.25) + ang_matched+ ang_matched2):eq(2)  -- if one pred bbox match a gt bbox its corresponding index equal to 1
			--print(correct_matches) 
	        	local match_numbers = torch.sum(torch.sum(correct_matches,2):ge(1))   -- how many pred bbox has its match gt bbox
                        if match_numbers > 0 then
                               local matname = img_name ..'.mat'
                               local matpath = '/home/zxw/DeepGrasp/output/TrueMDetecrtion/'..matname
                               matio.save(matpath, {coord = pred_Tensor, conf = scores})
                        else
                               local matname = img_name ..'.mat'
                               local matpath = '/home/zxw/DeepGrasp/output/FlaseMDetection/'..matname
                               matio.save(matpath, {coord = pred_Tensor, conf = scores})
                        end
	        	pred_num = pred_num + boxes:size(1)
	        	succ_pred =  succ_pred + match_numbers
                        --print(img_name,match_numbers)
	        	per_img_acc[img_name] = match_numbers/boxes:size(1)
		end
                  
	        local tmp_time = endtime - starttime
	        run_time = run_time + tmp_time
end

