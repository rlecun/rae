----------------------------------------------------------------------
-- This script downloads and loads the MNIST dataset
-- http://yann.lecun.com/exdb/mnist/
----------------------------------------------------------------------
print '==> downloading dataset'

-- Here we download dataset files. 

-- Note: files were converted from their original LUSH format
-- to Torch's internal format.

-- The SVHN dataset contains 3 files:
--    + train: training data
--    + test:  test data

tar = 'http://data.neuflow.org/data/mnist.t7.tgz'

datadir = datadir or './'

if not paths.dirp(datadir .. 'mnist.t7') then
   os.execute('cd ' .. datadir .. '; wget ' .. tar)
   os.execute('cd ' .. datadir .. '; tar xvf ' .. paths.basename(tar))
end

train_file = datadir .. 'mnist.t7/train_32x32.t7'
test_file = datadir .. 'mnist.t7/test_32x32.t7'

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, it's straightforward

trainData = torch.load(train_file,'ascii')
testData = torch.load(test_file,'ascii')

setmetatable(trainData, {__index = function(self, index)
                                       index = math.mod(index - 1, trainData.data:size(1)) + 1
                                       local sample = trainData.data:select(1, index)
                                       local size = sample:size(1) * sample:size(2) * sample:size(3)
                                       dsample = (torch.Tensor(size):copy(sample:resize(size))):mul(1/255)
                                       return {dsample,dsample,dsample}
                                   end})

print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

