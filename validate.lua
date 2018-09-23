require 'image'
require 'gnuplot'
testgt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Test.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Test.t7')

function TensorToPoints(tensor)
        local tmp_tensor = tensor:view(4,2)
	local points = {}
        points.x = torch.Tensor(4):copy(torch.squeeze(tmp_tensor[{{},{1}}]))
        points.y = torch.Tensor(4):copy(torch.squeeze(tmp_tensor[{{},{2}}]))
	return points
end

function pointsToLabel(prior_bboxes)    --utils
  local label = torch.Tensor(prior_bboxes:size(1),5)
  label[{{}, {1}}] = torch.add(prior_bboxes[{{}, {1}}], prior_bboxes[{{}, {5}}]):div(2)    -- center x
  label[{{}, {2}}] = torch.add(prior_bboxes[{{}, {2}}], prior_bboxes[{{}, {6}}]):div(2)    -- center y
  label[{{}, {3}}] = torch.sqrt(torch.add(torch.pow(torch.csub(prior_bboxes[{{}, {3}}],prior_bboxes[{{}, {5}}]),2),torch.pow(torch.csub(prior_bboxes[{{}, {4}}],prior_bboxes[{{}, {6}}]),2)))--h
  label[{{}, {4}}] = torch.sqrt(torch.add(torch.pow(torch.csub(prior_bboxes[{{}, {7}}],prior_bboxes[{{}, {5}}]),2),torch.pow(torch.csub(prior_bboxes[{{}, {8}}],prior_bboxes[{{}, {6}}]),2)))--w
  label[{{}, {5}}] = math.deg(torch.atan(torch.cdiv(torch.csub(prior_bboxes[{{}, {6}}],prior_bboxes[{{}, {8}}]),torch.csub(prior_bboxes[{{}, {5}}],prior_bboxes[{{}, {7}}]))))
  return label
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

function ImageVFilp(img,label)
    local resimg = image.vflip(img)
    label[{{},{5}}] = - label[{{},{5}}]
    label[{{},{2}}] = 480 - label[{{},{2}}]

    return resimg,label
end

function ImageHFilp(img,label)
    local resimg = image.hflip(img)
    label[{{},{5}}] = - label[{{},{5}}]
    label[{{},{1}}] = 640 - label[{{},{1}}]

    return resimg,label
end
imggt = testgt['pcd0404r.png']
img_name = '/home/zxw/DeepGrasp/data/cornell_dataset/image/pcd0404r.png'
img = image.load(img_name)
flip = image.load(img_name)
print(imggt)


	for j =1 , imggt:size(1) do
		local  points = TensorToPoints(imggt[{{j},{}}])
        	for t=1,4,1 do
        	          img = image.drawText(img, "x", points.x[t], points.y[t])
        	end
	end
image.save('0404.png',img)

label = pointsToLabel(imggt)

fliped, newlabel = ImageHFilp(flip,label)
--fliped, newlabel = ImageVFilp(fliped,newlabel)
   -- image.display(fliped)

	for j =1 , newlabel:size(1) do
	        newlabel[j][5] = math.rad(newlabel[j][5])
		local  points = labelToPoints(newlabel[j])
        	for t=1,4,1 do
        	          fliped = image.drawText(fliped, "x", points.x[t], points.y[t])
        	end
	end
image.save('H0404.png',fliped)

