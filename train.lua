-- Train a neural network on a dataset.
function train(trainSet,
                trainLabels,
                nbOfLabels,
                validationSet,
                validationLabels,
                maxUselessEpochs,
                maxFixedStepUselessEpochs,
                stepSizeDecay,
                epsilon,
                rgbdNet,
                criterion,
                stepSize,
                nbOfEpoch,
                batchSize,
                testBatchSize)
    print('----------------------TRAINING--------------------------------')

    local trainConfusion = optim.ConfusionMatrix(nbOfLabels)

    local N = trainSet:size(1)
    local K = trainSet:size(2)
    local X, Y = trainSet:size(3), trainSet:size(4)

    local bestValidationAccuracy = 0
    local bestNet = rgbdNet
    local bestTrainAccuracy = 0
    local uselessEpochs = 0
    local fixedStepUselessEpochs = 0


    --[[Training cycle]]--
    local timer = torch.Timer()
    for epoch = 1, nbOfEpoch do
        trainConfusion:zero()

        local shuffle = torch.randperm(N)
        for t = 1, N, batchSize do
            -- create mini batch
            local currentBatchSize = math.min(batchSize, N - t + 1)
            local inputs = torch.CudaTensor(currentBatchSize, K, X, Y)
            local targets = torch.CudaTensor(currentBatchSize)
            for b_n = 1, currentBatchSize do
                inputs[b_n] = trainSet[shuffle[b_n + t - 1]]
                targets[b_n] = trainLabels[shuffle[b_n + t - 1]]
            end

            -- reset gradients
            rgbdNet:zeroGradParameters()

            --calculate gradient for the current batch
            local outputs = rgbdNet:forward(inputs)
            criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            rgbdNet:backward(inputs, df_do)

            local _, predictedLabels = outputs:max(2)
            trainConfusion:batchAdd(predictedLabels, targets)

            rgbdNet:updateParameters(stepSize)
        end

        trainConfusion:updateValids()
        local trainAccuracy = trainConfusion.totalValid * 100
        print(string.format("Epoch " ..epoch.. ": mean train accuracy = %f%%", trainConfusion.totalValid * 100))

        -- Validate
        local validationAccuracy = test(validationSet, validationLabels, nbOfLabels, rgbdNet, testBatchSize, true) -- true = validating, not testing.

        -- If no improvements in last maxUselessEpochs epochs - abort and return the best performing net.
        if validationAccuracy <= bestValidationAccuracy + epsilon then
            uselessEpochs = uselessEpochs + 1
            if uselessEpochs == maxUselessEpochs then
                break
            else
                fixedStepUselessEpochs = fixedStepUselessEpochs + 1
                -- If no improvements for the given step size for maxFixedStepUselessEpochs epochs - decrease the step size
                -- and return to the best performing net so far.
                if fixedStepUselessEpochs == maxFixedStepUselessEpochs then
                    stepSize = stepSize / stepSizeDecay
                    print("New step size = " .. stepSize)
                    fixedStepUselessEpochs = 0
                    rgbdNet = bestNet:clone()
                end
            end
        else
            bestValidationAccuracy = validationAccuracy
            bestTrainAccuracy = trainAccuracy
            bestNet = rgbdNet:clone()
            uselessEpochs = 0
        end
    end
    print('Trained network in ' .. timer:time().real .. ' seconds.')
    torch.save('rgbdNet.dat', bestNet)

    return bestNet, bestTrainAccuracy
end