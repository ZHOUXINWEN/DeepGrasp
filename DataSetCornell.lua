-- This file is borrowed from https://github.com/fmassa/object-detection.torch
local lfs = require"lfs"
require 'image'
require 'math'
require 'torch'
require 'os'

local DataSetCornell = torch.class('DataSetCornell')
DataSetCornell._isDataSet = true

function DataSetCornell:__init(path)
  self.datasetpath = path
  self.datasetimagelist = self:getFileList(path..'image');
  table.sort(self.datasetimagelist)
  self.datasetsize = #self.datasetimagelist
  self.datasetposannolist = self:getFileList(path..'posanno');
  table.sort(self.datasetposannolist)
  self.datasetnegannolist = self:getFileList(path..'neganno');
  table.sort(self.datasetnegannolist)
  self.datasetcloudlist = self:getFileList(path..'cloud');
  table.sort(self.datasetcloudlist)
  self.gridNum = 7
  self.mean = {0.5,0.5,0.5}
end

function DataSetCornell:size()
  return self.datasetsize
end

function DataSetCornell:getFileList (path)
    local list = {}
    local counter = 0;
    for file in lfs.dir(path) do
        if file ~= "." and file ~= ".." then
            local f = path..'/'..file
            local attr = lfs.attributes (f)
            assert (type(attr) == "table")
            counter = counter+1;
            list[counter] = f;
            local a,b = math.modf(counter/10000)
            if b==0 then
                print ((a*10000)..'image')
            end
        end
    end
    return list;
end

function DataSetCornell:getImagePath(i)
  return self.datasetimagelist[i]
end

function DataSetCornell:getImage(i)
  local filename = self:getImagePath(i)
  local image = image.load(filename);
  return image
end

function DataSetCornell:getPosAnnotationPath(i)
  return self.datasetposannolist[i]
end

function DataSetCornell:getNegAnnotationPath(i)
  return self.datasetnegannolist[i]
end

function DataSetCornell:getrandomGraspBoxes(i,isPos)
    local filename = nil
    if isPos then
      filename = self:getPosAnnotationPath(i)
    else
      filename = self:getNegAnnotationPath(i)
    end
    local rfile=io.open(filename, "r")                    --read the file 
    assert(rfile)                                         --打开时验证是否出错   
    local lx,ly={},{}
    local i=1              
    for x,y in rfile:lines("*n","*n") do     --以数字格式一行一行的读取  
        --print(x,y)                         --显示在屏幕上  
        lx[i], ly[i]=x,y                     --put the coordinates in table labxy
        i=i+1
    end
    local alea=math.floor(i/4)               --there are alea groups of annotation in this picture
    math.randomseed(os.time())
    local choosen=math.random(1,alea)              --create random number to choose Rect GT
    --print(choosen)
    local inde=(choosen-1)*4
    local points = {}
    points.x = torch.Tensor(4)
    points.y = torch.Tensor(4)
    for t=1,4,1 do
      points.x[t] = lx[inde+t]
      points.y[t] = ly[inde+t]
    end
    rfile:close()                                    --调用结束后记得关闭  
    return points
end

function DataSetCornell:getAllGraspBoxes(i,isPos)
    local filename = nil
    if isPos then
      filename = self:getPosAnnotationPath(i)
    else
      filename = self:getNegAnnotationPath(i)
    end
    local rfile=io.open(filename, "r")                    --read the file 
    assert(rfile)                                         --打开时验证是否出错   
    local lx,ly={},{}
    local i=1              
    for x,y in rfile:lines("*n","*n") do     --以数字格式一行一行的读取  
        --print(x,y)                         --显示在屏幕上  
        lx[i], ly[i]=x,y                     --put the coordinates in table labxy
        i=i+1
    end
    local points = {}
    points.num=math.floor(i/4)
    points.x = lx
    points.y = ly
    rfile:close()                                    --调用结束后记得关闭  
    return points
end

function DataSetCornell:pointsToLabel(p)
  local label = torch.Tensor(5)
  label[1]=(p.x[3]+p.x[1])/2
  label[2]=(p.y[3]+p.y[1])/2
  label[3]=math.sqrt((p.x[2]-p.x[3])^2+(p.y[2]-p.y[3])^2)   --h
  label[4]=math.sqrt((p.x[4]-p.x[3])^2+(p.y[4]-p.y[3])^2)   --w
  label[5]=math.atan((p.y[3]-p.y[4])/(p.x[3]-p.x[4]))
  return label
end

function DataSetCornell:labelToPoints(label)
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

function DataSetCornell:ImageRotate(img,r)
  local ImageR=image.rotate(img,r)
  return ImageR
end

function DataSetCornell:LabelRotate(label,r,center)
  local labelR=label:clone()
  local radius=math.sqrt((label[1]-center.x)^2+(label[2]-center.y)^2)
  cosor = (label[1]-center.x)/radius
  sinor = (label[2]-center.y)/radius
  cosr = math.cos(r)
  sinr = math.sin(r)
  cosRot = cosor*cosr+sinor*sinr
  sinRot = sinor*cosr-cosor*sinr
  labelR[5]=label[5]-r
  labelR[1]=radius*cosRot+center.x
  labelR[2]=radius*sinRot+center.y
  return labelR
end

function DataSetCornell:ImageCrop(img,Xran,Yran)
  local XOriN=120+Xran
  local YOriN=90+Yran
  local ImageCroped=image.crop(img,XOriN,YOriN,XOriN+336,YOriN+336) --the box of crop should be lower than 350 pixels
  return ImageCroped
end

function DataSetCornell:LabelCrop(label,Xran,Yran)  
  local XOriN=120+Xran
  local YOriN=90+Yran
  label[1]=label[1]-XOriN
  label[2]=label[2]-YOriN
  return label
end

function DataSetCornell:ImageResize(img,scaler)
  local resizeW = img:size(3)*scaler
  local resizeH = img:size(2)*scaler
  local ImageResized=image.scale(img,resizeW,resizeH) 
  return ImageResized
end

function DataSetCornell:LabelResize(label,scaler)
  label:sub(1,4):mul(scaler)
  return label
end

function DataSetCornell:dataPreprocess(imgID)
  local r = (math.random(1,7)-4)/10
  local img = self:getImage(imgID)
  local pts = self:getAllGraspBoxes(imgID,true)
  local targetMat = torch.Tensor(6,self.gridNum,self.gridNum):zero()
  local labels = {}
  --rotate
  img = self:ImageRotate(img,r)
  for ptsNum = 1,pts.num,1 do
    local points = {}
    points.x = {pts.x[ptsNum*4-3],pts.x[ptsNum*4-2],pts.x[ptsNum*4-1],pts.x[ptsNum*4]}
    points.y = {pts.y[ptsNum*4-3],pts.y[ptsNum*4-2],pts.y[ptsNum*4-1],pts.y[ptsNum*4]}
    local label = self:pointsToLabel(points)
    label = self:LabelRotate(label,r,{x=img:size(3)/2,y=img:size(2)/2})
    labels[ptsNum] = label
  end
  --crop
  local Yran=math.random(0,40)
  Yran=Yran-20    
  local Xran=math.random(0,40)
  Xran=Xran-20       
  img=self:ImageCrop(img,Xran,Yran) --the box of crop should be lower than 350 pixels
  for ptsNum = 1,pts.num,1 do
    labels[ptsNum] = self:LabelCrop(labels[ptsNum],Xran,Yran)
  end
  for ptsNum = 1,pts.num,1 do
    local w = math.ceil(labels[ptsNum][1]*self.gridNum/336)
    local h = math.ceil(labels[ptsNum][2]*self.gridNum/336)
    if w<=7 and w>=1 and h<=7 and h>=1 then
      targetMat[{6,h,w}]=1;
      targetMat[{1,h,w}]=labels[ptsNum][1] - (w-1)*(336/self.gridNum)
      targetMat[{2,h,w}]=labels[ptsNum][2] - (h-1)*(336/self.gridNum)
      for labelNum = 3,5,1 do
        targetMat[{labelNum,h,w}]=labels[ptsNum][labelNum]
      end
    end
  end
  return targetMat,img
end

function DataSetCornell:getCloudPath(i)
  return self.datasetcloudlist[i]
end

function DataSetCornell:readDepthMap(imgID)
  local filename = self:getCloudPath(imgID)
  --filename = '/home/zxw/DeepGrasp/data/cornell_dataset/cloud/pcd0170.txt'
  --print(filename)
  local rfile=io.open(filename, "r")                    --read the file 
  assert(rfile)                                         --打开时验证是否出错   
  local depth = torch.Tensor(480,640)
  local x,y,z,c,i
  local filereader = rfile:lines()
  for t=1,10 do --first 10 lines are not useful
    filereader()
  end
  for a in filereader do
    local num = self:pickNumsFromStr(a)
    x = num[1]
    y = num[2]
    z = num[3]
    c = num[4]
    i = num[5]
    local row = math.floor(i/ 640) + 1
    local col = (i % 640) + 1
    local depthValue = math.sqrt(x^2+y^2+z^2)/4096
    depth[{{row},{col}}] = depthValue;
  end
  rfile:close()                                    --调用结束后记得关闭  
  return depth
end

function DataSetCornell:combineRGD(imgID)
  local RGB = self:getImage(imgID)
  local D = self:readDepthMap(imgID)
  local imgRGD = torch.Tensor(3,480,640)
  imgRGD[{1}] = RGB[{1}]
  imgRGD[{2}] = RGB[{2}]
  imgRGD[{3}] = D
  return imgRGD
end

function DataSetCornell:combineRGBD(imgID)
  local RGB = self:getImage(imgID)
  local D = self:readDepthMap(imgID)
  local imgRGBD = torch.Tensor(4,480,640)
  imgRGBD[{1}] = RGB[{1}]
  imgRGBD[{2}] = RGB[{2}]
  imgRGBD[{3}] = RGB[{3}]
  imgRGBD[{4}] = D
  return imgRGBD
end

function DataSetCornell:computeIOU(rec1,rec2)
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
      if self:isInRect(point,rec1) then
        mask[{{h-math.ceil(ymin)+1},{w-math.ceil(xmin)+1}}] = mask[{{h-math.ceil(ymin)+1},{w-math.ceil(xmin)+1}}]+1
        countMerge = countMerge+1
      end
      if self:isInRect(point,rec2) then
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
-- other functions
function DataSetCornell:pickNumsFromStr(str)
  local numtbl={}
  local count = 1
  local i1=0
  local i2=1
  local c = string.sub(str,i2,i2)
  local printflag = 0;
  local asc = string.byte;
  while ((asc(c)<asc('0') or asc(c)>asc('9')) and asc(c)~=asc('.') and asc(c)~=asc('-') and asc(c)~=asc('e') and c~='') do
    print('loop1')
    i1 = i1+1;
    i2 = i2+1;
    c = string.sub(str,i2,i2)
  end
  while (true) do
    if (c~='' and (asc(c)>=asc('0') and asc(c)<=asc('9')) or asc(c)==asc('.') or asc(c)==asc('-') or asc(c)==asc('e')) then -- c is a component of number
      printflag=0;
      i2=i2+1;
    elseif (tonumber(string.sub(str,i1+1,i2-1))~=nil and printflag == 0) then -- c is not a component of number and string.sub(str,i1+1,i2-1) is a number
      numtbl[count]=tonumber(string.sub(str,i1+1,i2-1))
      printflag =1;
      i1=i2;
      i2=i2+1;
      count=count+1;
    elseif (c=='') then  -- end of string
      break
    else -- c is not a component of number and string.sub(str,i1+1,i2-1) is not a number
      i1=i2;
      i2=i2+1;
    end
    c=string.sub(str,i2,i2)
  end
  return numtbl
end

function DataSetCornell:isInRect(point,vertex)
  local isVertical = false
  for t=1,3 do
    if vertex.x[t] == vertex.x[t+1] then
      isVertical = t
    end
  end
  if not isVertical then
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
    if (point.x - vertex.x[isVertical])*(point.x - vertex.x[(isVertical+2)%4])<0 then
      if (point.y - vertex.y[isVertical])*(point.y - vertex.y[(isVertical+2)%4])<0 then
        return true
      end
      return false
    end
    return false
  end
end
