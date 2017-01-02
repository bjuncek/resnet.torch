--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'torchx'
require 'paths'
require 'optim'
require 'nn'
local plotting = require 'plotting'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)

-- Workbook logging
-- Feel free to comment these out.
hasWorkbook, labWorkbook = pcall(require, 'lab-workbook')
if hasWorkbook then
  workbook = labWorkbook:newExperiment{}
  lossLog = workbook:newTimeSeriesLog("Training loss",
                                      {"nEpoch", "loss"},
                                      100)
  errorLog = workbook:newTimeSeriesLog("Testing Error",
                                       {"nEpoch", "error"})
else
  print "WARNING: No workbook support. No results will be saved."
end


torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

-- Logger
logger = optim.Logger(paths.concat(opt.save,'training.log'))
logger:setNames{"Training Error", 'Validation Error', "Training Loss", "Validation Loss"}

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

trainingStats = { testLoss={}, trainLoss={}, testError={}, trainError={}}

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
local bestLoss = math.huge
local bestEpoch = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss, trainLossAbs = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5, testLoss, testLossAbs = trainer:test(epoch, valLoader)

   if hasWorkbook then
      lossLog{nEpoch = epoch, loss = trainLoss}
      errorLog{nEpochs = epoch or 0,
                 error = 1.0 - testTop1}
    end

   -- Update training stats
   table.insert(trainingStats.testError, testTop1)
   table.insert(trainingStats.trainError, trainTop1)
   -- table.insert(trainingStats.trainLoss, trainLoss)
   -- table.insert(trainingStats.testLoss, testLoss)
   table.insert(trainingStats.trainLoss, trainLossAbs) -- for regression
   table.insert(trainingStats.testLoss, testLossAbs)

   -- Update logger
   logger:add{trainTop1, testTop1, trainLossAbs, testLossAbs}

   -- Plot learning curves
   plotting.error_curve(trainingStats, opt)
   plotting.loss_curve(trainingStats, opt)

   local bestModel = false
   if testLossAbs < bestLoss then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      bestLoss = testLossAbs
      bestEpoch = epoch
      print(string.format(' * Best Model -- epoch:%i  top1: %6.3f  top5: %6.3f  loss: %6.3f', bestEpoch, bestTop1, bestTop5, bestLoss))

   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Best Model -- epoch:%i  top1: %6.3f  top5: %6.3f  loss: %6.3f', bestEpoch, bestTop1, bestTop5, bestLoss))
