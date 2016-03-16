-- Construct a more VGG-like network with modern architecture decisions.
function ourNet(useDepth)
    local net = nn.Sequential()
    net:add(nn.SpatialBatchNormalization(useDepth and 4 or 3))

    local conv1 = nn.SpatialConvolution(useDepth and 4 or 3, 32, 3, 3, 1, 1, 1, 1)
    net:add(conv1)
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    local conv2 = nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)
    net:add(conv2)
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    local conv3 = nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)
    net:add(conv3)
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

    net:add(nn.View(128 * 8 * 8))
    net:add(nn.Linear(128 * 8 * 8, 128 * 8 * 8))
    net:add(nn.BatchNormalization(128 * 8 * 8))
    net:add(nn.ReLU())

    net:add(nn.Linear(128 * 8 * 8, 48))
    net:add(nn.BatchNormalization(48))

    net:add(nn.LogSoftMax())

    net:cuda()

    return net
end