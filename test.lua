-- Test or validate a dataset. If testing, richer text formatting is used.
function test(testSet, testLabels, nbOfLabels, net, batchSize, isValidating)
    if not isValidating then
        print('----------------------TESTING---------------------------------')
    end

    local confusion = optim.ConfusionMatrix(nbOfLabels)
    local timer = torch.Timer()


    local N = testSet:size(1)
    if batchSize == 0 then batchSize = N end

    local K = testSet:size(2)
    local X, Y = testSet:size(3), testSet:size(4)
    for t = 1, N, batchSize do
        -- create mini batch
        local currentBatchSize = math.min(batchSize, N - t + 1)
        local inputs = torch.CudaTensor(currentBatchSize, K, X, Y)
        local targets = torch.CudaTensor(currentBatchSize)
        for b_n = 1, currentBatchSize do
            inputs[b_n] = testSet[b_n + t - 1]
            targets[b_n] = testLabels[b_n + t - 1]
        end
        local outputs = net:forward(inputs)
        local maxScores, predictedLabels = outputs:max(2)
        confusion:batchAdd(predictedLabels, targets)
    end

    confusion:updateValids()


    if isValidating then
        print(string.format("\t mean validation accuracy = %f%%", confusion.totalValid * 100))
    else
        print('Tested data in ' .. timer:time().real .. ' seconds.')
        print(string.format("Mean test accuracy = %f%%", confusion.totalValid * 100))
        print(string.format("Baseline = %f%%", 100 / nbOfLabels))
        print('----------------------TEST END--------------------------------')
    end

    return confusion.totalValid * 100
end