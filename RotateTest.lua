require 'image'


testgt = torch.load('/home/svc3/DeepGrasp/cache/cornell_5_Test.t7')
testpath = torch.load('/home/svc3/DeepGrasp/cache/paths_5_Test.t7')

function pointsToLabel(prior_bboxes)    --utils
  local label = torch.Tensor(prior_bboxes:size(1),5)
  label[{{}, {1}}] = torch.add(prior_bboxes[{{}, {1}}], prior_bboxes[{{}, {5}}]):div(2)    -- center x
  label[{{}, {2}}] = torch.add(prior_bboxes[{{}, {2}}], prior_bboxes[{{}, {6}}]):div(2)    -- center y
  label[{{}, {3}}] = torch.sqrt(torch.add(torch.pow(torch.csub(prior_bboxes[{{}, {3}}],prior_bboxes[{{}, {5}}]),2),torch.pow(torch.csub(prior_bboxes[{{}, {4}}],prior_bboxes[{{}, {6}}]),2)))--h
  label[{{}, {4}}] = torch.sqrt(torch.add(torch.pow(torch.csub(prior_bboxes[{{}, {7}}],prior_bboxes[{{}, {5}}]),2),torch.pow(torch.csub(prior_bboxes[{{}, {8}}],prior_bboxes[{{}, {6}}]),2)))--w
  label[{{}, {5}}] = -torch.atan(torch.cdiv(torch.csub(prior_bboxes[{{}, {6}}],prior_bboxes[{{}, {8}}]),torch.csub(prior_bboxes[{{}, {5}}],prior_bboxes[{{}, {7}}])))
  return label
end
--[[
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
  if label[5] < 0 then
	  points.x=torch.Tensor({center.x+w1.x+h2.x,center.x+w2.x+h2.x,center.x+w2.x+h1.x,center.x+w1.x+h1.x})
	  points.y=torch.Tensor({center.y+w1.y+h2.y,center.y+w2.y+h2.y,center.y+w2.y+h1.y,center.y+w1.y+h1.y})
  else
	  points.x=torch.Tensor({center.x+w2.x+h2.x,center.x+w1.x+h1.x,center.x+w1.x+h1.x,center.x+w2.x+h2.x})
	  points.y=torch.Tensor({center.y+w2.y+h2.y,center.y+w1.y+h1.y,center.y+w1.y+h1.y,center.y+w2.y+h2.y})
  end
  return points
end
--]]
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

--  local uw = {}
  local uwx = torch.cos(label[{{},{5}}])
  local uwy = torch.sin(label[{{},{5}}])
--  local uh = {}
  local uhx = torch.sin(label[{{},{5}}])
  local uhy = -torch.cos(label[{{},{5}}])
--  local w1 = {x=uw.x*w/2,y=uw.y*w/2}
  local w1x = torch.cmul(uwx, w)/2
  local w1y = torch.cmul(uwy, w)/2

--  local w2 = {x=-uw.x*w/2,y=-uw.y*w/2}
  local w2x = -torch.cmul(uwx, w)/2
  local w2y = -torch.cmul(uwy, w)/2
--  local h1 = {x=uh.x*h/2,y=uh.y*h/2}
  local h1x = torch.cmul(uhx, h)/2  
  local h1y = torch.cmul(uhy, h)/2  

--  local h2 = {x=-uh.x*h/2,y=-uh.y*h/2}
  local h2x = -torch.cmul(uhx, h)/2
  local h2y = -torch.cmul(uhy, h)/2


  points[{{},{1}}] = center_x+w1x+h1x
  points[{{},{2}}] = center_y+w1y+h1y
  points[{{},{3}}] = center_x+w1x+h2x
  points[{{},{4}}] = center_y+w1y+h2y
  points[{{},{5}}] = center_x+w2x+h2x
  points[{{},{6}}] = center_y+w2y+h2y
  points[{{},{7}}] = center_x+w2x+h1x
  points[{{},{8}}] = center_y+w2y+h1y

  return points
end

function labelToPointsTensor2(label)
  local points = torch.Tensor(label:size(1), 8)
  local center_x = label[{{},{1}}]
  local center_y = label[{{},{2}}]
  local h = label[{{},{3}}]
  local w = label[{{},{4}}]
  local theta = label[{{},{5}}]
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

function ImageRotate(img,r)    
  local ImageR=image.rotate(img,r)
  return ImageR
end

function LabelRotate(label,r,center)
  local labelR=label:clone()
  local radius=math.sqrt((label[1]-center.x)^2+(label[2]-center.y)^2)
  print(radius)
  cosor = (label[1]-center.x)/radius
  sinor = (label[2]-center.y)/radius
  cosr = math.cos(r)
  sinr = math.sin(r)
  cosRot = cosor*cosr+sinor*sinr
  sinRot = sinor*cosr-cosor*sinr
  labelR[5]=label[5]-r*180/math.pi

  if labelR[5] < -90 then
    labelR[5] = labelR[5] + 180
  elseif labelR[5] > 90 then
    labelR[5] = labelR[5] - 180
  end
  --print(cosRot,sinRot)
  labelR[1]=radius*cosRot+center.x
  labelR[2]=radius*sinRot+center.y
  return labelR
end

function LabelRotateTensor(label, r, center)
  local labelR=label:clone()
  local radius=torch.sqrt(torch.pow((label[{{},{1}}]-center.x), 2)+torch.pow((label[{{},{2}}]-center.y), 2) )
  print(radius)
  cosor = torch.cdiv((label[{{},{1}}]-center.x), radius)
  sinor = torch.cdiv((label[{{},{2}}]-center.y), radius)
  cosr = math.cos(r)
  sinr = math.sin(r)

  labelR[{{},{5}}] = labelR[{{},{5}}]-r*180/math.pi

  cosRot = cosor*cosr+sinor*sinr
  sinRot = sinor*cosr-cosor*sinr
  local lowerB = torch.lt(labelR[{{},{5}}], -90)
  local higherB = torch.gt(labelR[{{},{5}}], 90)
  labelR[{{},{5}}] = torch.csub(torch.add(labelR[{{},{5}}]:float(), 180*lowerB:float()), 180*higherB:float())

  --print(cosRot,sinRot)
  labelR[{{},{1}}]=torch.cmul(radius,cosRot)+center.x
  labelR[{{},{2}}]=torch.cmul(radius,sinRot)+center.y
  return labelR
end
--[[
for data in io.lines('/home/svc3/DeepGrasp/data/labeledimg/49/imgpath.txt') do
	gt_points = testgt[data]
	--img = image.load('/home/svc3/DeepGrasp/data/cornell_dataset/image/'..data)
	gt_label = pointsToLabel(gt_points)
	--print(gt_points)
	hh = torch.Tensor(gt_points:size())
	--print(hh)
	for i = 1, gt_label:size(1) do
	    points =labelToPoints(gt_label[i])
	    --print(points.x[1], points.y[1], points.x[2], points.y[2], points.x[3], points.y[3], points.x[4], points.y[4])
	    hh[i] = torch.Tensor({points.x[1], points.y[1], points.x[2], points.y[2], points.x[3], points.y[3], points.x[4], points.y[4]})
	end
	--print(hh)

	hh_label = pointsToLabel(hh)
	--print(gt_points)
	--print(hh)
	--print(labelToPointsTensor(gt_label))
	--print(torch.sum((gt_points- hh),2))  
	--print(gt_label)
        --print(hh_label)
	print(torch.sum((hh_label-gt_label), 1))
	--print(pointsToLabel(points)- hh_label)
end
--]]


for data in io.lines('/home/svc3/DeepGrasp/data/labeledimg/49/imgpath.txt') do
        print(data)
	gt_points = testgt[data]
        print(gt_points)
        gt_label = pointsToLabel(gt_points)
        print(gt_label)
	re_points = labelToPointsTensor2(gt_label)
        print(re_points)
        --print(torch.sum((re_points-gt_points),1))
        re_label = pointsToLabel(re_points)
	print(re_label)
        print((re_label-gt_label))
        print(torch.sum((re_label-gt_label),1))
end



--[[print(gt_label)
r = 0.314
center = { x = 320,
           y = 240 }

for i = 1, gt_label:size(1) do
    print(LabelRotate(gt_label[i], r, center))
end

labelR = LabelRotateTensor(gt_label, r, center)
print(labelR)
rotated = ImageRotate(img, r)

	for j =1 , labelR:size(1) do
	        labelR[j][5] = math.rad(labelR[j][5])
		local  points = labelToPoints(labelR[j])
        	for t=1,4,1 do
        	          rotated = image.drawText(rotated, "x", points.x[t], points.y[t])
        	end
	end
image.save('rotated.png',rotated)
--]]
