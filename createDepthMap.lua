require 'DataSetCornell'
require 'image'
require 'paths'

datapath = '/home/zxw/DeepGrasp/data/cornell_dataset/'

d01 = DataSetCornell(datapath)

if (not paths.dirp(datapath..'depth')) then
  os.execute('mkdir '..'depth')
end

for imgID = 1,d01:size() do
  local depthImage =d01:readDepthMap(imgID)
  local RGBName =  paths.basename(d01:getImagePath(imgID))
  local Dname = string.gsub(RGBName,'r','d')
  image.save(datapath..'depth/'..Dname,depthImage )
end
  
