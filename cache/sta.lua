
traingt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Train.t7')
trainpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Train.t7')
testgt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Test.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Test.t7')

number = 0

for  i,box in pairs(traingt) do
          number = number + box:size(1)
	  print(box)
end

print(number)
