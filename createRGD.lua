require 'DataSetCornell'
require 'image'
require 'paths'
--[[
datapath = '/home/zxw/DeepGrasp/data/cornell_dataset/'

d01 = DataSetCornell(datapath)

if (not paths.dirp(datapath..'depth')) then
  os.execute('mkdir '..'depth')
end

imgname_RGD = {}

for imgID = 1,d01:size() do
  --local depthImage =d01:readDepthMap(imgID)
  local RGBName =  paths.basename(d01:getImagePath(imgID))
  local Dname = string.gsub(RGBName,'r','d')
  --print(RGBName)
  --print(Dname)
  imgname_RGD[RGBName] = d01:combineRGD(imgID)
  --image.save(datapath..'depth/'..Dname,depthImage )
end

torch.save('./cache/RGD.t7', imgname_RGD)

print(torch.load('/home/zxw/DeepGrasp/cache/paths_5_Test.t7'))--]] 

imgs = torch.load('/home/zxw/DeepGrasp/cache/RGD.t7')--[[
img = imgs['pcd0104r.png']
img =image.crop(img, 0, 0, 320, 320)
img = image.vflip(img)
print(img:size()) 
--]]
trainpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Train.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Test.t7')

train_RGBD = {}
test_RGBD = {}

for  i,img_name in pairs(trainpath) do
    train_RGBD[img_name] = imgs[img_name]        
end


for  i,img_name in pairs(testpath) do
    test_RGBD[img_name] = imgs[img_name]        
end


torch.save('./cache/train_RGD.t7', train_RGBD)
torch.save('./cache/test_RGD.t7', test_RGBD)
--]]
