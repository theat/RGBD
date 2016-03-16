--require 'mobdebug'
require 'paths'
require 'optim'
require 'cutorch'
require 'cunn'
require 'torch'

--[[files]]--
paths.dofile('load.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')
paths.dofile('weight-init.lua')
paths.dofile('classicNet.lua')
paths.dofile('ourNet.lua')


--[[Hyper-parameters]]--
local useDepth = false							-- true if using the 4th depth channel
local net = ourNet(useDepth)					-- the network architecture to use
local stepSize = 0.05							-- the gradient descent step size
local nbOfEpoch = 200							-- maximum allowed number of epochs for training
local criterion = nn.ClassNLLCriterion():cuda()	-- loss criterion
local maxUselessEpochs = 50						-- max allowed #epochs without validation imporvements
local maxFixedStepUselessEpochs = 10			-- max allowed #epochs without imporvements before reducing the gradient step size
local stepSizeDecay = 2							-- gradient descent step size is divided by this number if no improvement is observed for a while
local epsilon = 1e-10							-- precision used to observe 'no imporvement'
local batchSize = 32							-- batch size to calculate the gradient estimation
local initMethod = 'kaiming'					-- method to initialize network weights
local normalizeData = false						-- true if normalizing all the datasets by the training dataset mean / std
local N = 10 									-- number of experiments
local testBatchSize = 0							-- batchs size when testing / validation. 0 means using all in a single batch.



--[[Load datasets]]--
local dataPath = '../data/rgbd-dataset/'
local trainFile = './config/rgbd_train_names.txt'
local validationFile = './config/rgbd_valid_names.txt'
local testFile = './config/rgbd_test_names.txt'

local trainSet, trainLabels, nbOfLabels = load(trainFile, dataPath, useDepth)
local validationSet, validationLabels, _ = load(validationFile, dataPath, useDepth)
local testSet, testLabels, _ = load(testFile, dataPath, useDepth)

--[[Normalize datasets]]--
if normalizeData then
	local trainMeans, trainStds = normalize(trainSet)
	normalize(validationSet, trainMeans, trainStds)
	normalize(testSet, trainMeans, trainStds)
end


--[[Run experiments]]--
local train_errors = torch.Tensor(N)
local test_errors = torch.Tensor(N)
for i = 1, N do
	print("EXPERIMENT " .. i .. ":")
	--net:reset()
	net = w_init(net, initMethod)
	net, train_errors[i] = train(trainSet,
		trainLabels,
		nbOfLabels,
		validationSet,
		validationLabels,
		maxUselessEpochs,
		maxFixedStepUselessEpochs,
		stepSizeDecay,
		epsilon,
		net,
		criterion,
		stepSize,
		nbOfEpoch,
		batchSize,
		testBatchSize)
	test_errors[i] = test(testSet, testLabels, nbOfLabels, net, testBatchSize, false) -- false = not validating, but testing.
end


--[[Print results]]--
print("Train errors \t Test errors")
for i = 1, N do
	print(string.format('%f \t\t %f', train_errors[i], test_errors[i]))
end
print("\n average:")
print(string.format('%f \t\t %f', torch.mean(train_errors), torch.mean(test_errors)))
print("\n std:")
print(string.format('%f \t\t %f', torch.std(train_errors), torch.std(test_errors)))
