require 'image'
require 'optim'
require 'nn'
require 'autoencoder'
require 'L1Criterion'
require 'L1overL2Criterion'
require 'FormalCriterion'

----------------------------------------------------------------------
-- load data
dofile 'data.lua'

filename = 'tr-berkeley-N5K-M56x56-lcn.ascii'
rundir = './'
-- for l2 criterion and normalized decoder
-- alpha = 0.005
alpha = 0.02
beta = 0.02
inputsize = 12
codesize = 100
maxiter = 100000

sgdconf = {
  learningRate = 0.1,
  learningRateDecay = 1e-5,
  momentum = 0
  }

statinterval = 1000
display = true
dataset = getdata(filename, inputsize)

if display then
    displayData(dataset, 100, 10, 2)
end

-----------------------------------------------------------------------
print '==> constructing model'

-- params
inputSize = inputsize*inputsize
outputSize = codesize

-- auto encoder
encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize, outputSize))
encoder:add(nn.SoftShrink())
encoder:add(nn.Linear(outputSize, outputSize))
encoder:add(nn.Tanh())
-- encoder:add(nn.Tanh())
decoder = nn.Linear(outputSize, inputSize)
cost = nn.MSECriterion()
-- reg = nn.L1Criterion(alpha)
-- reg = nn.L1overL2Criterion(alpha, beta)
reg = nn.FormalCriterion(alpha, makeS(outputSize))
module = nn.regautoencoder(encoder, decoder, cost, reg)

-- verbose
print('==> constructed linear auto-encoder')

-----------------------------------------------------------------------
-- get all parameters
x,dl_dx = module:getParameters()

function go(maxiter)
  -- training errors
  local err = 0
  local errec = 0
  local erreg = 0
  local iter = 0

  for t = 1,maxiter do
    local sample = dataset[t]
    local input = sample[1]:clone()
    local target = input

    --------------------------------------------------------------------
    -- define eval closure
    local feval = function()

      -- reset gradient/f
      local f = 0
      dl_dx:zero()

      -- f
      f = module:updateOutput(input)

      -- gradients
      module:accGradParameters(input)
      -- module:normalize()

      -- return f and df/dx
      return f,dl_dx
    end

    _,fs = optim.sgd(feval, x, sgdconf)
    err = err + fs[1]
    errec = errec + module.recenergy
    erreg = erreg + module.regenergy

    --------------------------------------------------------------------
    -- compute statistics and report error
    if math.fmod(t, statinterval) == 0 then

      -- report
      print('==> iteration = ' .. t .. ', loss = ' .. err/statinterval)
      print('    recloss = ' .. errec/statinterval .. ', regloss = ' .. erreg/statinterval)

      torch.save(rundir .. '/outputs/model_' .. t .. '.bin', module)

      dd,de = module:displayweights(20,inputsize,inputsize)
      -- save filter images
      image.save(rundir .. '/img/filters_dec_' .. t .. '.jpg', dd)
      image.save(rundir .. '/img/filters_enc_' .. t .. '.jpg', de)
      -- live display
      if display then      
        _win1_ = image.display{image=dd,win=_win1_,legend='Decoder',zoom=2}
        _win2_ = image.display{image=de,win=_win2_,legend='Encoder',zoom=2}
      end

      -- reset counters
      err = 0; errec = 0; erreg = 0; iter = 0
    end
  end
end

go(100000)

