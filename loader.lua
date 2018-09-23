require 'paths'
require 'nn'
--require 'image'     
local trainPath  = {}
local testPath = {}
local trainGT = {}
local testGT = {}   
counter = 1
cache_path = './cache'

local cfg = {
  scale = 54,
  nmap = 1, -- number of feature maps
  msize = {21}, -- feature matp size
  bpc = {6}, -- number of default boxes per cell
  aratio = {1, '1', 2, 1/2, 3, 1/3},  -- aspect retio
  variance = {0.1, 0.1, 0.2, 0.2, 0.2},
  steps = {16},
  imgshape = 336, -- input image size
  classes = 2
}

for data in io.lines('/home/zxw/SSD_cornell/cornell_dataset/imagepath.txt') do

    if counter%5 == 0 then
       table.insert(testPath, data)
    else 
       table.insert(trainPath, data)
    end
    --print(data)
    counter = counter + 1
end


    --print(prefix..'cpos.txt')
for i, data in pairs(trainPath) do
    local annopath = string.sub(data,1,7)..'cpos.txt'
    local filename = '/home/zxw/SSD_cornell/cornell_dataset/posanno/'..annopath
    local rfile=io.open(filename, "r")                    --read the file 
    assert(rfile)                                         --打开时验证是否出错   
    local tmp_data = {}
    local points = {}
    
    local i=1              
    for x,y in rfile:lines("*n","*n") do     --以数字格式一行一行的读取  
       table.insert(points,x)                        --显示在屏幕上  
       table.insert(points,y)    
        if i%4 == 0 then
        	--table.insert(points,1,2)
                table.insert(tmp_data, torch.Tensor(points):reshape(1,8))
		points ={}		
        end  
	i = i+1      
    end
    trainGT[data] = nn.JoinTable(1):forward(tmp_data)  -- combination of torch.DoubleTensor of size *x8
    --print(trainGT)	
end
for i, data in pairs(testPath) do
    local annopath = string.sub(data,1,7)..'cpos.txt'
    local filename = '/home/zxw/SSD_cornell/cornell_dataset/posanno/'..annopath
    local rfile=io.open(filename, "r")                    --read the file 
    assert(rfile)                                         --打开时验证是否出错   
    local tmp_data = {}
    local points = {}
    
    local i=1              
    for x,y in rfile:lines("*n","*n") do     --以数字格式一行一行的读取  
       table.insert(points,x)                        --显示在屏幕上  
       table.insert(points,y)    
        if i%4 == 0 then
        	--table.insert(points,1,2)
                table.insert(tmp_data, torch.Tensor(points):reshape(1,8))
		points ={}		
        end  
	i = i+1      
    end
    testGT[data] = nn.JoinTable(1):forward(tmp_data)  -- combination of torch.DoubleTensor of size *x8
    --print(trainGT)	
end
    torch.save(paths.concat(cache_path, 'cornell_5_Train.t7'), trainGT)
    torch.save(paths.concat(cache_path, 'cornell_5_Test.t7'), testGT)
    torch.save(paths.concat(cache_path, 'paths_5_Train.t7'), trainPath)
    torch.save(paths.concat(cache_path, 'paths_5_Test.t7'), testPath)

