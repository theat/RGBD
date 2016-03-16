-- Recreate the network from the article.
function classicNet(useDepth)
    local net = nn.Sequential()

    local conv1 = nn.SpatialConvolution(useDepth and 4 or 3, 32, 5, 5, 1, 1, 2, 2)
    net:add(conv1)
    net:add(nn.SpatialMaxPooling(3,3,2,2))
    net:add(nn.ReLU())

    local conv2 = nn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2)
    net:add(conv2)
    net:add(nn.ReLU())
    net:add(nn.SpatialAveragePooling(3,3,2,2))

    local conv3 = nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2)
    net:add(conv3)
    net:add(nn.ReLU())
    net:add(nn.SpatialAveragePooling(3,3,2,2))

    net:add(nn.View(64*7*7))

    local fc48 = nn.Linear(64*7*7, 48)
    net:add(fc48)
    net:add(nn.LogSoftMax())

    net:cuda()
    return net
end