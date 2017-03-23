require 'torch'
require 'xlua'
require 'image'
require 'paths'
require 'nn'
require 'cutorch'
require 'cudnn'
require 'cunn'
require 'optim'

opt = lapp[[
  --max_epoch       (default 30)            Maximum number of epochs
  --use_cuda        (default true)          Using CUDA or not
  --backend         (default cudnnn)        Backend for model
  --batch_size      (default 10)            Batch size
  --learning_rate   (default 0.1)           Learning rate
  --decay_rate      (default 0.5)           Decay rate
  --decay_epoch     (default 10)            Starting decay epoch
]]
-- Load data
function load_data()
  train_data = torch.load('data/train_32x32.t7', 'ascii')
  test_data = torch.load('data/test_32x32.t7', 'ascii')
end

-- Setup model
function setup_model()
  local model = nn.Sequential()

  model:add(nn.SpatialConvolution(1, 32, 5, 5))
  model:add(nn.Tanh())
  model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
 
  model:add(nn.SpatialConvolution(32, 64, 5, 5))
  model:add(nn.Tanh())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  model:add(nn.Reshape(576))
  model:add(nn.Linear(576, 200))
  model:add(nn.Tanh())
  model:add(nn.Linear(200, 10))
  model:add(nn.LogSoftMax())

  return model
end

function init()
  torch.manualSeed(1)

  net = setup_model()

  -- Loss function
  criterion = nn.ClassNLLCriterion()

  if (opt.backend == 'cudnn') then
    cudnn.convert(net, cudnn)
  end

  if (opt.use_cuda == true) then
    net:cuda()
    criterion:cuda()
  end

  confusion = optim.ConfusionMatrix({'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'})

  collectgarbage()
  collectgarbage()
end

function evaluate()
  print('Now evaluating the test size')
  local test_size = test_data.data:size(1)
  local batch_size = opt.batch_size
  confusion:zero()

  for i = 1, test_size, batch_size do
    xlua.progress(i, test_size)
    -- Prepare data
    local data = test_data.data[{{i, i + batch_size - 1}}]
    local labels = test_data.labels[{{i, i + batch_size - 1}}]
    -- labels = torch.reshape(labels, 1, batch_size)

    if (opt.use_cuda == true) then
      input = torch.CudaTensor()
      target = torch.CudaTensor()
      input:resize(data:size()):copy(data)
      target:resize(labels:size()):copy(labels)
    else
      input = torch.Tensor()
      target = torch.FloatTensor()
      input:resize(data:size()):copy(data)
      target:resize(labels:size()):copy(labels)
    end

    local output = net:forward(input)

    for j = 1, batch_size do
      confusion:add(output[j], target[j])
    end
  end
  print('\n')
  print(confusion)
end

function train()
  local max_epoch = opt.max_epoch
  local train_size = train_data.data:size(1)
  local batch_size = opt.batch_size
  local learning_rate = opt.learning_rate

  for epoch = 1, max_epoch do
    local loss = 0
    for i = 1, train_size, batch_size do
      xlua.progress(i, train_size)
      -- Prepare data
      local data = train_data.data[{{i, i + batch_size - 1}}]
      local labels = train_data.labels[{{i, i + batch_size - 1}}]
      -- labels = torch.reshape(labels, 1, batch_size)

      if (opt.use_cuda == true) then
        input = torch.CudaTensor()
        target = torch.CudaTensor()
        input:resize(data:size()):copy(data)
        target:resize(labels:size()):copy(labels)
      else
        input = torch.Tensor()
        target = torch.FloatTensor()
        input:resize(data:size()):copy(data)
        target:resize(labels:size()):copy(labels)
      end

      local output = net:forward(input)
      local f = criterion:forward(output, target)
      loss = loss + (f / batch_size)

      net:zeroGradParameters()

      local gradients = criterion:backward(output, target)
      net:backward(input, gradients)
      net:updateParameters(learning_rate)
    end
    print('\n epoch = ' .. epoch .. ' | loss = ' .. loss)
    if (epoch >= opt.decay_epoch and epoch % 5 == 0) then
      learning_rate = learning_rate * opt.decay_rate
    end
    if (epoch % 5 == 0) then
      evaluate()
    end
  end
end

load_data()
init()
train()
