require 'paths'
require 'nn'
local matio = require 'matio'
--require 'image'     
local trainPath  = torch.load('/home/zxw/DeepGrasp/cache/RNpaths_5_Test.t7')
local trainGT = {}



for i, data in pairs(trainPath) do
print(data)
    local annopath = string.sub(data,1,7)..'cpos.txt'
    local filename = '/home/zxw/DeepGrasp/data/cornell_dataset/posanno/'..annopath
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
    local matname = data ..'.mat'
    local matpath = '/home/zxw/DeepGrasp/output/GT/'..matname
    matio.save(matpath, {coord = trainGT[data], conf = 1})
    --print(trainGT)	
end
