require 'pl'
require 'image'
require 'optim'
require 'cutorch'
require 'gnuplot'

torch.setdefaulttensortype('torch.FloatTensor')

dofile('utils320.lua')


function Top1Detect(model, imgs, cfg)
  if imgs:dim() ~= 4 then
    imgs = imgs:reshape(1, imgs:size(1), imgs:size(2), imgs:size(3))
  end
  -- local timer = torch.Timer()
  local outputs = model:forward(imgs:cuda())
  -- print('Time elapsed for detection ' .. timer:time().real .. ' seconds')
  local conf_preds = outputs[2]:float()
  local prior_bboxes = GetPriorBBoxes(cfg)
  local conf 
  local loc_preds
  for i = 1, imgs:size(1) do
    -- print(outputs[1])
    loc_preds = DecodeBBox(outputs[1][i]:float(), prior_bboxes, cfg.variance)
    local softmax_conf = nn.SoftMax():forward(conf_preds[i]):view(-1,cfg.classes)
    conf = softmax_conf[{{},{2}}]
    print(conf)
    local idx = torch.squeeze(softmax_conf[{{},{2}}]:gt(0.5),2):nonzero()       --
    --      maxconf ,maxid = torch.squeeze(softmax_conf[{{},{2}}]:gt(0.5),2)
    local maxconf ,maxid = torch.max(softmax_conf[{{},{2}}],1)
    print(maxconf, maxid)


    conf = conf:index(1,torch.squeeze(maxid,2))
    loc_preds = loc_preds:index(1,torch.squeeze(maxid,2))

    return conf, loc_preds
  end
end

traingt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Train.t7')
trainpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Train.t7')
testgt = torch.load('/home/zxw/DeepGrasp/cache/cornell_5_Test.t7')
testpath = torch.load('/home/zxw/DeepGrasp/cache/paths_5_Test.t7')

result = torch.load('/home/zxw/DeepGrasp/output/320b_12DGssdResAng448_v501_resblock/100epoch.t7')

cfg = result['cfg']
--print(cfg)
opt = result['opt']
model = result['model']:cuda()

--po = trainGT['pcd1021r.png']

img_path = '/home/zxw/DeepGrasp/data/cornell_dataset/image/pcd1021r.png'

img = image.load(img_path)

res = img:clone()
res = ImageCrop(res, 0, 0,cfg)
conf, loc_preds = Top1Detect(model, res, cfg)
print(conf, loc_preds)
