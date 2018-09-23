require 'DataSetCornell'
require 'image'
require 'paths'

datapath = '/home/zxw/DeepGrasp/data/cornell_dataset/'

d01 = DataSetCornell(datapath)

if (not paths.dirp(datapath..'depth')) then
  os.execute('mkdir '..'depth')
end

imgname_RGBD = {}

for imgID = 1,d01:size() do
  --local depthImage =d01:readDepthMap(imgID)
  local RGBName =  paths.basename(d01:getImagePath(imgID))
  local Dname = string.gsub(RGBName,'r','d')
  --print(RGBName)
  --print(Dname)
  imgname_RGBD[RGBName] = d01:combineRGBD(imgID)
  --image.save(datapath..'depth/'..Dname,depthImage )
end

torch.save('./cache/RGBD.t7', imgname_RGBD)
  
