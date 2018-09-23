require 'nn'
require 'cudnn'

res = torch.load('/home/zxw/DeepGrasp/output/320b_12DGssdResAng448_v501_resblock/100epoch.t7')
print(res['cfg'])
print(res['opt'])
print(res['TestResult']['pic_num_ND'])
print(res['TestResult']['total'])
