require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require 'datasets/transforms'

-- Arg check
if #arg < 3 then
   io.stderr:write('Usage: th classify.lua [MODEL] [SIZE] [FILE]...\n')
   os.exit(1)
end

-- Get lables
local file = io.open("models/cp.txt", "r");
local labels = {}
for line in file:lines() do
   table.insert(labels, line);
end

-- Load the model
local model = torch.load(arg[1]):cuda()
local softMaxLayer = cudnn.SoftMax():cuda()
-- add Softmax layer
model:add(softMaxLayer)
-- Evaluate mode
model:evaluate()

-- Parse rest of the args
local cropSize = arg[2];

-- Necesary utils
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
   t.ColorNormalize(meanstd)
}




for i=3,#arg do
    local img = image.load(arg[i], 3, 'byte')
    local h, w = img:size(2), img:size(3)

    -- Init results
    local results = {};
    for lab=0, #labels, 1 do
        print(labels)
    end

    for width=0, w, math.floor(cropSize/3) do
        for height=0, h, math.floor(cropSize/3) do
            temp = image.crop(img, width, height, width+cropSize, height+cropSize);
            temp = temp:view(1, table.unpack(temp:size():totable()));

            -- Get the output of the softmax
            local output = model:forward(temp:cuda()):squeeze()
            local probs, predictions = output:topk(1, true, true)
            print(predictions)
            --results[labels[predictions[1]]] = results[labels[predictions[1]]] + 1;

        end
    end

    print(arg[i], results)
end
