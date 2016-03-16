require 'image'
require 'io'


-- Load a list of files into a Tensor of size #samples * #channels * X * Y.
-- Create an appropriately ordered Tensor of labels.
-- Dataset used: http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/
function load(listFile, dataPath, useDepth)
	--load list of filenames
	local timer = torch.Timer()
	local namesFile = io.open(listFile)
	if namesFile then
		local N = 0
		local lines = {}
		for line in namesFile:lines() do
			N = N + 1
			lines[N] = line
		end
		local K = useDepth and 4 or 3
		local X, Y = 64, 64

		local dataSet = torch.Tensor(N, K, X, Y)
		local labels, labelsToInd = {}, {}
		local nbOfLabels = 0

		for n = 1, N do
			local term1, term2, term3 = unpack(lines[n]:split("_"))
			local class, classInstance
			if tonumber(term2) ~= nil then
			   --it's a number
				class = term1
				classInstance = term2
			else
				class = term1..'_'..term2
				classInstance = term3
			end
			local label = class..'_'..classInstance
			if labelsToInd[label] == nil then
				nbOfLabels = nbOfLabels+1
				labelsToInd[label] = nbOfLabels
			end
			local imageName = dataPath ..class..'/'..label..'/'..lines[n]
			local depthName = string.gsub(imageName, "crop", "depthcrop", 1)
			dataSet[n][{{1, 3}, {}, {}}] = image.scale(image.load(imageName), X, Y)
			if useDepth then
				dataSet[n][4] =  image.scale(image.load(depthName), X, Y)
			end
			labels[n] = labelsToInd[label]
		end
		namesFile:close()

		print('----------------------------------------------------------------------------------------------------')
		print(string.format("From %s loaded %d examples with %d labels in %f seconds.", listFile, N, nbOfLabels, timer:time().real))

		return dataSet, torch.Tensor(labels), nbOfLabels
	else
		error('error opening file with names')
	end
end


-- Normalize the dataset by a given mean / std.
-- If no values provided, compute them based on the dataset.
-- Based on https://github.com/torch/demos/blob/master/person-detector/preprocessing.lua
function normalize(dataSet, means, stds)
	local K = dataSet:size(2)

	if means == nil or stds == nil then
		means = torch.CudaTensor(K)
		stds = torch.CudaTensor(K)

		for k = 1, K do
			local channelSlice = dataSet[{{}, k, {}, {}}]
			means[k] = channelSlice:mean()
			stds[k] = channelSlice:std()
		end
	end

	for k = 1, K do
		local channelSlice = dataSet[{{}, k, {}, {}}]
		channelSlice:add(-means[k])
		channelSlice:div(stds[k])
	end

	return means, stds
end