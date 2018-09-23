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

img_dir = '/home/zxw/DeepGrasp/data/MultiObject'
result = torch.load('/home/zxw/DeepGrasp/output/Resnet101_RGB_10x10x6_HV_R15deg_b16_lrd1_RN/1950epoch.t7')
cfg = result['cfg']
--print(cfg)
model = result['model']:cuda()

model:evaluate()
img_name = 'test.png'
--for img_name in io.lines('/home/zxw/DeepGrasp/data/MultiObject/imgname.txt')  do 
        img_path = paths.concat(img_dir,img_name)
     
	img = image.load(img_path)
	--res = img:clone()
	--res = ImageCrop(res, 0, 0,cfg)

	local scores, boxes  = VDetect(model, img, cfg)
        local pred_Tensor = labelToPointsTensor(boxes)
        local matpath = '/home/zxw/DeepGrasp/data/MultiObject/'..img_name..'.mat'
        matio.save(matpath, {coord = pred_Tensor, conf = scores})
--end
