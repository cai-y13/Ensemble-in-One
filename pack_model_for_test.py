import torch

model = dict()

model_0 = torch.load('save_temp/model.pth')

model['model_0'] = model_0

torch.save(model, 'model.pth')
