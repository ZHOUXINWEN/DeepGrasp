require 'image'
require 'cutorch'
require 'gnuplot'
require 'nn'
require 'cunn'
require 'cudnn'
torch.setdefaulttensortype('torch.FloatTensor')
dofile('cfg320.lua')
dofile('utils320.lua')

--[[img = image.load('/home/zxw/DeepGrasp/test.jpg')
img = image.crop(img,520, 0, 3640, 3120)
img = image.scale(img,320,320)
image.display(img)
image.save('test.png', img)--]]
result = torch.load('/home/zxw/DeepGrasp/output/320b_12DGssdResAng448_v501/180epoch.t7')

model = result['model']:cuda()
accuracy = result['TestResult']
--print(accuracy)
img = image.load('/home/zxw/DeepGrasp/test.png')
img:cuda()
 scores, boxes = VDetect(model, img, 0.5, cfg)


for j =1 , boxes:size(1) do
	boxes[j][5] = math.rad(boxes[j][5])
	local  points = labelToPoints(boxes[j])
				--print(points)
        for t=1,4,1 do
                  img = image.drawText(img, "x", points.x[t], points.y[t])
        end
end
image.save('ge.png', img)
print( scores, boxes )
