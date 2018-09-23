result = torch.load('/home/zxw/DeepGrasp/output/10x10x6_OW/Resnet50_RGD_20x20x6_HV_b16_lrd2_OW_RN_stop22222_FF3/2200Result.t7')
accuracy = result['accuracy']

AccuracyTensor = torch.Tensor(accuracy)--[{{31,90}}]
print('Average',torch.mean(AccuracyTensor))
score, index = torch.topk(AccuracyTensor,5,1,true)
print(score, index*10)

--[[for i, v in ipairs(accuracy) do
   if v > 0.98 then
   print(i*10, v)
   end
end--]]
