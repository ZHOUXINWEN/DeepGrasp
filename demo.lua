model_path = '/home/zxw/ssd-torch7/model120000iteration.t7'
img_path ='/home/zxw/ssd-torch7/LVUsine.jpg'
torch.setdefaulttensortype('torch.FloatTensor')
_=dofile('ssd.lua')
dofile('cfg.lua')
dofile('utils.lua')
require 'image'

model = torch.load('/home/zxw/DeepGrasp/output/model12_1e-3_Aug44832000iter.t7')

model = torch.load(model_path):cuda()
img = image.load(img_path)
res = img:clone()
img = image.scale(img, 300, 300)

class_list = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}

img_r = img[{{1},{},{}}]
img_b = img[{{3},{},{}}]
img[{{1},{},{}}] = img_b
img[{{3},{},{}}] = img_r
boxes, classes, scores = Detect(model, img, 0.45, 0.6, cfg)
for i = 1, boxes[1]:size(1) do
  boxes[1][i][1] = boxes[1][i][1] * res:size(3)
  boxes[1][i][2] = boxes[1][i][2] * res:size(2)
  boxes[1][i][3] = boxes[1][i][3] * res:size(3)
  boxes[1][i][4] = boxes[1][i][4] * res:size(2)
end
res = DrawRect(res, boxes[1], classes[1], class_list)

image.savePNG('LVUsine.png', res)
