require 'image'
require 'gnuplot'

testgt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Test.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Test.t7')
traingt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Train.t7')
trainpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Train.t7')

function TensorToPoints(tensor)
        local tmp_tensor = tensor:view(4,2)
	local points = {}
        points.x = torch.Tensor(4):copy(torch.squeeze(tmp_tensor[{{},{1}}]))
        points.y = torch.Tensor(4):copy(torch.squeeze(tmp_tensor[{{},{2}}]))
	return points
end
 img_dir = '/home/zxw/DeepGrasp/data/cornell_dataset/image'
for  i,img_name in pairs(trainpath) do
        local img_path = paths.concat(img_dir,img_name)
	local img = image.load(img_path)
        local boxes = traingt[img_name]

	for j =1 , boxes:size(1) do
		local  points = TensorToPoints(boxes[j])
        	for t=1,4,1 do
        	          img = image.drawText(img, "x", points.x[t], points.y[t])
        	end
	end
        image.save('/home/zxw/DeepGrasp/data/labeledimg/'..img_name, img)
end
