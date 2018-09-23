trainpath = torch.load('/home/zxw/DeepGrasp/cache/FiveFold/RNpaths_4_Train.t7')

testpath = torch.load('/home/zxw/DeepGrasp/cache/FiveFold/RNpaths_4_Test.t7')
for i =1,177 do
                      print(testpath[i])
end 
print('----------------------------------')
	for j= 1,708 do
		--if trainpath[j] == testpath[i] then
                      print(trainpath[j])
		--end
	end
