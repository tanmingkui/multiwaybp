--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
local nn = require 'nn'
local optim = require 'optim'
local utils = require 'utils'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState, fcs)
   self.model = model
   self.pivotSet = [8, 13, 18, 23]
   self.pivotWeights = {0.11, 0.3, 0.61, 1}
   table.sort(self.pivotSet, function(a,b) return a<b end)

   self.imsize = (opt.dataset == 'imagenet' or opt.dataset == 'imagenet100') and 224
                  or 32

   self.lrWeights = {}
   local base_lr = 0
   for i=table.getn(self.pivotSet), 1, -1 do
      base_lr = base_lr + self.pivotWeights[i]
      self.lrWeights[i] = base_lr
   end

   self.criterionSet = {}
   self.criterion = criterion
   for i=1, table.getn(self.pivotSet) do
      table.insert(self.criterionSet, self.criterion:clone())
   end
   table.insert(self.criterionSet, self.criterion)

   self.optimStateSet = {}
   if optimState and torch.type(optimState)=='table' then
      self.optimStateSet = optimState
   else
      self.optimState = optimState or {
         learningRate = opt.LR,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         nesterov = true,
         dampening = 0.0,
         weightDecay = opt.weightDecay,
      }
      for i=1, table.getn(self.pivotSet) do
         local optimStateTemp_1 = {}
         for j=table.getn(self.pivotSet)+2-i, 1, -1 do
         	local optimStateTemp = optimState or {
	         learningRate = opt.LR,
	         learningRateDecay = 0.0,
	         momentum = opt.momentum,
	         nesterov = true,
	         dampening = 0.0,
	         weightDecay = opt.weightDecay,
	         }
	         table.insert(optimStateTemp_1, optimStateTemp)
         end
         local optimStateTemp_2 = optimState or {
         learningRate = opt.LR,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         nesterov = true,
         dampening = 0.0,
         weightDecay = opt.weightDecay,
         }
         table.insert(self.optimStateSet, {optimStateTemp_1, optimStateTemp_2})
      end
      table.insert(self.optimStateSet, self.optimState)
   end

   self.opt = opt
-- log setting
   print('Will save at '..opt.save)
   paths.mkdir(opt.save)
   -- trainLogger
   self.trainLog = assert(io.open(paths.concat(opt.save, 'train.log'),'a'))
   -- testLogger
   self.testLog = assert(io.open(paths.concat(opt.save, 'test.log'),'a'))

   local function addFC(m)
      local fc = nn.Sequential()
      local input = torch.randn(1,3,self.imsize,self.imsize):cuda()
      local output = m:forward(input)
      -- fc:add(nn.JoinTable(2, 4))
      fc:add(cudnn.SpatialAveragePooling(output:size(3),output:size(4),1,1))
      fc:add(nn.View(output:size(2)):setNumInputDims(3))
      if self.opt.dataset == 'imagenet' then
         fc:add(nn.Linear(output:size(2),1000))
      elseif self.opt.dataset == 'imagenet100' then
         fc:add(nn.Linear(output:size(2),100))
      elseif self.opt.dataset == 'cifar10' or opt.dataset == 'svhn' then
         fc:add(nn.Linear(output:size(2),10))
      elseif self.opt.dataset == 'cifar100' then
         fc:add(nn.Linear(output:size(2),100))
      end
      for k,v in pairs(fc:findModules('nn.Linear')) do
         v.bias:zero()
      end
      fc:cuda()
      utils.sharedGradInputEnable(self.opt, fc)
      fc = utils.parallelEnable(self.opt, fc)
      return fc
   end

   local function determineLayerIndex(pivot, N)
      local index = 0
      if pivot <= 1 + N then
         index =  2*pivot-1>0 and 2*pivot-1 or 0
      elseif pivot <= 2 + 2 * N then
         index = (pivot-N-2)*2 + 1 + 2*N + 5
      elseif pivot <= 3 + 3 * N then
         index = (pivot-2*N-3)*2 + 1 + 4*N + 10
      end
      return index
   end

   local function netSegmentation()
      local segments = {}
      local depth = self.opt.depth
      local lastPivot = 0
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
         [200] = {{3, 24, 36, 3}, 2048, bottleneck},
      }
      local def, nFeatures, block
      if self.opt.dataset == 'imagenet' or self.opt.dataset == 'imagenet100' then
         def, nFeatures, block = table.unpack(cfg[depth])
      end
      local preface = nil
      if opt.netType == 'resnet' then
         preface = 3
      elseif opt.netType == 'preresnet' then
         preface = 1
      end
      if opt.netType == 'densenet_lite' then
         local N = (depth - 4)/3
         local isBottleNeck = #self.model:get(2):get(2)>4
         if isBottleNeck then N = N/2 end
         local iChannels, iW, iH = 3, self.imsize, self.imsize

         for k, pivotNum in ipairs(self.pivotSet) do
            local segment = nn.Sequential()
            assert(pivotNum<=3*(1+N), 'pivot error')
            -- appending
            local iStart = determineLayerIndex(lastPivot, N) + 1
            local iStop = determineLayerIndex(pivotNum, N)
            for i=iStart, iStop do
               segment:add(self.model:get(i))
            end

            lastPivot = pivotNum
            segment = segment:clone()
            segment:cuda()
            utils.sharedGradInputEnable(self.opt, segment)
            utils.optnetEnable(self.opt, segment, iChannels, iW, iH)
            local sampleInput = torch.randn(1, iChannels, iW, iH):cuda()
            local outSize = segment:forward(sampleInput)
            iChannels = outSize:size(2)
            iW = outSize:size(3)
            iH = outSize:size(4)
            segment = utils.parallelEnable(self.opt, segment)
            table.insert(segments, segment)
         end
         -- last segment
         local segment = nn.Sequential()
         local totalDepth = #(self.model)
         for i=determineLayerIndex(lastPivot, N)+1, totalDepth do
            segment:add(self.model:get(i))
         end
         segment = segment:clone()
         segment:cuda()
         utils.sharedGradInputEnable(self.opt, segment)
         utils.optnetEnable(self.opt, segment, iChannels, iW, iH)
         segment = utils.parallelEnable(self.opt, segment)
         table.insert(segments, segment)
      else
         local iChannels, iW, iH = 3, self.imsize, self.imsize
         for k, pivotNum in ipairs(self.pivotSet) do
            if self.opt.dataset == 'imagenet' or self.opt.dataset == 'imagenet100' then
               -- appending
               local segment = nn.Sequential()
               if pivotNum == 1 then
                  for i=1,4 do
                     segment:add(self.model:get(i))
                  end

               elseif pivotNum <= def[1]+1 then
                  for i=1,4 do
                     segment:add(self.model:get(i))
                  end
                  for i=1,pivotNum-1 do
                     segment:add(self.model:get(5):get(i))
                  end
               elseif pivotNum <= def[1]+def[2]+1 then
                  for i=1,5 do
                     segment:add(self.model:get(i))
                  end
                  for i=1,pivotNum-def[1]-1 do
                     segment:add(self.model:get(6):get(i))
                  end
               elseif pivotNum <= def[1]+def[2]+def[3]+1 then
                  for i=1,6 do
                     segment:add(self.model:get(i))
                  end
                  for i=1,pivotNum-def[1]-def[2]-1 do
                     segment:add(self.model:get(7):get(i))
                  end
               elseif pivotNum <= def[1]+def[2]+def[3]+def[4]+1 then
                  for i=1,7 do
                     segment:add(self.model:get(i))
                  end
                  for i=1,pivotNum-def[1]-def[2]-def[3]-1 do
                     segment:add(self.model:get(8):get(i))
                  end
               end
               -- removing
               if lastPivot > 0 then
               if lastPivot == 1 then
                  for i=1,4 do
                     segment:remove(1)
                  end
               elseif lastPivot < def[1]+1 then
                  for i=1,4 do
                     segment:remove(1)
                  end
                  for i=1,lastPivot-1 do
                     if pivotNum <= def[1]+1 then
                        segment:remove(1)
                     else
                        segment:get(1):remove(1)
                     end
                  end
               elseif lastPivot < def[1]+def[2]+1 then
                  for i=1,5 do
                     segment:remove(1)
                  end
                  for i=1,lastPivot-def[1]-1 do
                     if pivotNum <= def[1]+def[2]+1 then
                        segment:remove(1)
                     else
                        segment:get(1):remove(1)
                     end
                  end
               elseif lastPivot < def[1]+def[2]+def[3]+1 then
                  for i=1,6 do
                     segment:remove(1)
                  end
                  for i=1,lastPivot-def[1]-def[2]-1 do
                     if pivotNum <= def[1]+def[2]+def[3]+1 then
                        segment:remove(1)
                     else
                        segment:get(1):remove(1)
                     end
                  end
               elseif lastPivot < def[1]+def[2]+def[3]+def[4]+1 then
                  for i=1,7 do
                     segment:remove(1)
                  end
                  for i=1,lastPivot-def[1]-def[2]-def[3]-1 do
                     if pivotNum <= def[1]+def[2]+def[3]+def[4]+1 then
                        segment:remove(1)
                     else
                        segment:get(1):remove(1)
                     end
                  end
               end
               end

               lastPivot = pivotNum
               segment = segment:clone()
               segment:cuda()
               utils.sharedGradInputEnable(self.opt, segment)
               utils.optnetEnable(self.opt, segment, iChannels, iW, iH)
               local sampleInput = torch.randn(1, iChannels, iW, iH):cuda()
               local outSize = segment:forward(sampleInput)
               iChannels = outSize:size(2)
               iW = outSize:size(3)
               iH = outSize:size(4)
               segment = utils.parallelEnable(self.opt, segment)
               table.insert(segments, segment)
   --cifar10
            elseif self.opt.dataset == 'cifar10' or self.opt.dataset == 'cifar100' or self.opt.dataset == 'svhn' then
               -- appending
               local segment = nn.Sequential()
               local n = (depth - 2) / 6
               if pivotNum == 1 then
                  for i=1,preface do
                     segment:add(self.model:get(i))
                  end
               elseif pivotNum <= n+1 then
                  for i=1,preface do
                     segment:add(self.model:get(i))
                  end
                  for i=1,pivotNum-1 do
                     segment:add(self.model:get(preface+1):get(i))
                  end
               elseif pivotNum <=n*2+1 then
                  for i=1,preface+1 do
                     segment:add(self.model:get(i))
                  end
                  for i=1,pivotNum-n-1 do
                     segment:add(self.model:get(preface+2):get(i))
                  end
               elseif pivotNum <= n*3+1 then
                  for i=1,preface+2 do
                     segment:add(self.model:get(i))
                  end
                  for i=1,pivotNum-n*2-1 do
                     segment:add(self.model:get(preface+3):get(i))
                  end
               end

               -- removing

               if lastPivot > 0 then
                  if lastPivot == 1 then
                     for i=1,preface do
                        segment:remove(1)
                     end
                  elseif lastPivot < n+1 then
                     for i=1,preface do
                        segment:remove(1)
                     end
                     for i=1,lastPivot-1 do
                        if pivotNum <= n+1 then
                           segment:remove(1)
                        else
                           segment:get(1):remove(1)
                        end
                     end
                  elseif lastPivot <n*2+1 then
                     for i=1,preface+1 do
                        segment:remove(1)
                     end
                     for i=1,lastPivot-n-1 do
                        if pivotNum <= n*2+1 then
                           segment:remove(1)
                        else
                           segment:get(1):remove(1)
                        end
                     end
                  elseif lastPivot < n*3+1 then
                     for i=1,preface+2 do
                        segment:remove(1)
                     end
                     for i=1,lastPivot-n*2-1 do
                        if pivotNum <= n*3+1 then
                           segment:remove(1)
                        else
                           segment:get(1):remove(1)
                        end
                     end
                  end
               end

               lastPivot = pivotNum
               segment = segment:clone()
               segment:cuda()
               utils.sharedGradInputEnable(self.opt, segment)
               utils.optnetEnable(self.opt, segment, iChannels, iW, iH)
               local sampleInput = torch.randn(1, iChannels, iW, iH):cuda()
               local outSize = segment:forward(sampleInput)
               iChannels = outSize:size(2)
               iW = outSize:size(3)
               iH = outSize:size(4)
               segment = utils.parallelEnable(self.opt, segment)
               table.insert(segments, segment)
            else
               error('invalid dataset: ' .. opt.dataset)
            end
         end
               -- last segment
         if self.opt.dataset == 'imagenet' or self.opt.dataset == 'imagenet100' then
            local segment = nn.Sequential()
            local totalDepth = #(self.model)
            if lastPivot == 1 then
               for i=5,totalDepth do
                  segment:add(self.model:get(i))
               end
            elseif lastPivot <= def[1]+1 then
               for i=1,def[1]-lastPivot+1 do
                  segment:add(self.model:get(5):get(lastPivot+i-1))
               end
               for i=6,totalDepth do
                  segment:add(self.model:get(i))
               end
            elseif lastPivot <=def[1]+def[2]+1 then
               for i=1,def[1]+def[2]-lastPivot+1 do
                  segment:add(self.model:get(6):get(lastPivot-def[1]+i-1))
               end
               for i=7,totalDepth do
                  segment:add(self.model:get(i))
               end
            elseif lastPivot <= def[1]+def[2]+def[3]+1 then
               for i=1,def[1]+def[2]+def[3]-lastPivot+1 do
                  segment:add(self.model:get(7):get(lastPivot-def[1]-def[2]+i-1))
               end
               for i=8,totalDepth do
                  segment:add(self.model:get(i))
               end
            elseif lastPivot <= def[1]+def[2]+def[3]+def[4]+1 then
               for i=1,def[1]+def[2]+def[3]+def[4]-lastPivot+1 do
                  segment:add(self.model:get(8):get(lastPivot-def[1]-def[2]-def[3]+i-1))
               end
               for i=9,totalDepth do
                  segment:add(self.model:get(i))
               end
            end
            segment = segment:clone()
            segment:cuda()
            utils.sharedGradInputEnable(self.opt, segment)
            utils.optnetEnable(self.opt, segment, iChannels, iW, iH)
            segment = utils.parallelEnable(self.opt, segment)
            table.insert(segments, segment)
         elseif self.opt.dataset == 'cifar10' or self.opt.dataset == 'cifar100' or self.opt.dataset == 'svhn' then
            local segment = nn.Sequential()
            local n = (depth - 2) / 6
            local totalDepth = #(self.model)
            if lastPivot == 1 then
               for i=preface+1,totalDepth do
                  segment:add(self.model:get(i))
               end
            elseif lastPivot <= n+1 then
               for i=1,n-lastPivot+1 do
                  segment:add(self.model:get(preface+1):get(lastPivot+i-1))
               end
               for i=preface+2,totalDepth do
                  segment:add(self.model:get(i))
               end
            elseif lastPivot <=n*2+1 then
               for i=1,n*2-lastPivot+1 do
                  segment:add(self.model:get(preface+2):get(lastPivot-n+i-1))
               end
               for i=preface+3,totalDepth do
                  segment:add(self.model:get(i))
               end
            elseif lastPivot <= n*3+1 then
               for i=1,n*3-lastPivot+1 do
                  segment:add(self.model:get(preface+3):get(lastPivot-n*2+i-1))
               end
               for i=preface+4,totalDepth do
                  segment:add(self.model:get(i))
               end
            end
            segment = segment:clone()
            segment:cuda()
            utils.sharedGradInputEnable(self.opt, segment)
            utils.optnetEnable(self.opt, segment, iChansnels, iW, iH)
            segment = utils.parallelEnable(self.opt, segment)
            table.insert(segments, segment)
         end
      end
      return segments
   end
   if torch.type(self.model)=='table' then
      self.segments = self.model
      local iChannels, iW, iH = 3, self.imsize, self.imsize
      for i=1, table.getn(self.segments) do
         self.segments[i] = utils.parallelEnable(self.opt, self.segments[i])
         utils.sharedGradInputEnable(self.opt, self.segments[i])
         utils.optnetEnable(self.opt, self.segments[i], iChannels, iW, iH)
         local sampleInput = torch.randn(1, iChannels, iW, iH):cuda()
         local outSize = self.segments[i]:forward(sampleInput)
         if outSize:dim() == 4 then
            iChannels = outSize:size(2)
            iW = outSize:size(3)
            iH = outSize:size(4)
         end
      end
   else
      self.segments = netSegmentation()
   end

   -- fuse features at intermediate layers
   self.fuseSegments = {}
   if self.opt.fuse then
      local depth = self.opt.depth
      local n = (depth - 2) / 6
      table.insert(self.fuseSegments, self.segments[1])
      for i=2, table.getn(self.segments)-1 do
         lastBlock = torch.floor((self.pivotSet[i-1]-1)/n)
         currentBlock = torch.floor((self.pivotSet[i]-1)/n)
         local shortcut = nil
         if currentBlock-lastBlock<=0 then
            shortcut = nn.Identity()
         else
            shortcut = nn.Sequential()
            for i=1, currentBlock-lastBlock do
               shortcut:add(nn.SpatialAveragePooling(1, 1, 2, 2))
            end
         end

         local concat = nn.ConcatTable()
         concat:add(shortcut)
               :add(self.segments[i])
         local tempBlock = nn.Sequential()
         tempBlock:add(concat)
         tempBlock:cuda()
         utils.sharedGradInputEnable(self.opt, tempBlock)
         tempBlock = utils.parallelEnable(self.opt, tempBlock)
         table.insert(self.fuseSegments, tempBlock)
      end
   end

   for i=1, table.getn(self.fuseSegments) do
      print(self.fuseSegments[i])
   end

   self.fcs = {}
   if fcs and torch.type(fcs)=='table' then
      for k, fc in ipairs(fcs) do
         utils.sharedGradInputEnable(self.opt, fcs[k])
         fcs[k] = utils.parallelEnable(self.opt, fcs[k])
      end
      self.fcs = fcs
   else
      local temp_model = nn.Sequential()
      for i=1, table.getn(self.pivotSet) do
         if self.opt.fuse then
            assert(false)
            temp_model:add(self.fuseSegments[i]:get(1))
         else
            temp_model:add(self.segments[i]:get(1))
         end
         local fc = addFC(temp_model)
         table.insert(self.fcs, fc)
      end
   end


end

local function tensorConcat(dst, src)
   local size = dst:size()
   size[1] = size[1] + src:size(1)
   dst:resize(size)
   dst:narrow(1, size[1]-src:size(1)+1, src:size(1)):copy(src)
end

local function correctNaN(grad)
   nan_mask = grad:ne(grad)
   grad[nan_mask] = 0
end

local function paramsCounter(model)
   if type(model)=='table' then
      for i=1, table.getn(model) do
         p, gp = model[i]:getParameters()
         print(#p)
      end
   else
      p, gp = model:getParameters()
      print(#p)
   end
end

function Trainer:train(epoch, dataloader)
   print('-------------------')
   -- Trains the model for a single epoch
   for i=1, table.getn(self.optimStateSet)-1 do
   	  for j=1, table.getn(self.optimStateSet[i][1]) do
   	  	self.optimStateSet[i][1][j].learningRate = self:learningRate(epoch)
   	  end
      self.optimStateSet[i][2].learningRate = self:learningRate(epoch)
   end
   self.optimStateSet[table.getn(self.optimStateSet)].learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval(x)
      correctNaN(self.gradParams)
      return nil, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   for i=1,table.getn(self.segments) do
      self.segments[i]:training()
   end
   for i=1, table.getn(self.fuseSegments) do
      self.fuseSegments[i]:training()
   end

   for i=1,table.getn(self.fcs) do
      self.fcs[i]:training()
   end
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      -- forward
      local tempOutputSet = {}
      local lossSet = {}
      local temp_input = self.input
      local nPivots = table.getn(self.pivotSet)
      for i=1, 2 do
         local temp_output = nil
         if self.opt.fuse then
            temp_output = self.fuseSegments[i]:forward(temp_input)
         else
            temp_output = self.segments[i]:forward(temp_input)
         end
         local temp_fc_output = self.fcs[i]:forward(temp_output)
         local temp_loss = self.criterionSet[i]:forward(temp_fc_output, self.target)
         table.insert(tempOutputSet, temp_output)
         table.insert(lossSet, temp_loss)
         temp_input = temp_output
      end

      local final_output = self.segments[nPivots+1]:forward(tempOutputSet[nPivots])
      local final_loss = self.criterionSet[nPivots+1]:forward(final_output, self.target)

      local function backProp(index, pivot)
         for i=index, 1, -1 do
            self.params, self.gradParams = self.segments[i]:getParameters()
            self.segments[i]:zeroGradParameters()
            if gradInput_1 then
               gradInput_1:resizeAs(self.segments[i+1].gradInputGpu[1]):copy(self.segments[i+1].gradInputGpu[1])
            else
               gradInput_1 = self.segments[i+1].gradInputGpu[1]:clone()
            end
            for k=2,self.opt.nGPU do
               tensorConcat(gradInput_1, self.segments[i+1].gradInputGpu[k])
            end
            if i>1 then
               self.segments[i]:backward(tempOutputSet[i-1], gradInput_1)
            else
               self.segments[i]:backward(self.input, gradInput_1)
            end
            optim.sgd(feval, self.params, self.optimStateSet[i][1][table.getn(self.pivotSet)+2-pivot])
         end
      end

      local function adjustStepSize(currentLoss, pivotIndex)
         for i=1, pivotIndex do
            if currentLoss/final_loss > 2 then
               self.optimStateSet[i][1][nPivots+2-pivotIndex].learningRate = self:learningRate(epoch)*math.pow(final_loss/currentLoss, 1)
            else
               self.optimStateSet[i][1][nPivots+2-pivotIndex].learningRate = self:learningRate(epoch)/self.lrWeights[pivotIndex]
            end
         end
      end

      local dropAO = false
      -- if torch.uniform() < 0 then
      --    dropAO = true
      --    for i=1, table.getn(self.optimStateSet)-1 do
      --       self.optimStateSet[i][1].learningRate = self:learningRate(epoch)
      --    end
      -- else
      --    dropAO = false
      --    for i=1, table.getn(self.optimStateSet)-1 do
      --       self.optimStateSet[i][1].learningRate = self:learningRate(epoch)/self.lrWeights[i]
      --    end
      -- end
-- ibp
      for i=1, nPivots do
         --fc
         self.params, self.gradParams = self.fcs[i]:getParameters()
         self.fcs[i]:zeroGradParameters()
         self.criterionSet[i]:backward(self.fcs[i].output, self.target)
         self.fcs[i]:backward(tempOutputSet[i], self.criterionSet[i].gradInput)
         optim.sgd(feval, self.params, self.optimStateSet[i][2])
         --main
         adjustStepSize(lossSet[i], i)
         if dropAO ~= true then
            self.params, self.gradParams = self.segments[i]:getParameters()
            self.segments[i]:zeroGradParameters()
            if gradInput_1 then
               gradInput_1:resizeAs(self.fcs[i].gradInputGpu[1]):copy(self.fcs[i].gradInputGpu[1])
            else
               gradInput_1 = self.fcs[i].gradInputGpu[1]:clone()
            end
            for k=2,self.opt.nGPU do
               tensorConcat(gradInput_1, self.fcs[i].gradInputGpu[k])
            end
            gradInput_1:mul(self.pivotWeights[i])
            if i>1 then
               self.segments[i]:backward(tempOutputSet[i-1], gradInput_1)
            else
               self.segments[i]:backward(self.input, gradInput_1)
            end
            optim.sgd(feval, self.params, self.optimStateSet[i][1][nPivots+2-i])
            if i>1 then
               backProp(i-1, i)
            end
         end
      end

      self.params, self.gradParams = self.segments[nPivots+1]:getParameters()
      self.segments[nPivots+1]:zeroGradParameters()
      self.criterionSet[nPivots+1]:backward(self.segments[nPivots+1].output, self.target)
      self.segments[nPivots+1]:backward(tempOutputSet[nPivots], self.criterionSet[nPivots+1].gradInput)
      optim.sgd(feval, self.params, self.optimStateSet[nPivots+1])

      backProp(nPivots, nPivots+1)

      local top1, top5 = self:computeScore(final_output:float(), sample.target, 1)
      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, final_loss, top1, top5))

      if self.trainLog then
        local trainLog = self.trainLog:write(('%d \t %d \t %d \t %f \t %f \t %f \n'):format(epoch, n, trainSize, final_loss, top1, top5))
        trainLog:flush()
      end

      top1Sum = top1Sum + top1
      top5Sum = top5Sum + top5
      lossSum = lossSum + final_loss
      N = N + 1

      -- check that the storage didn't get changed do to an unfortunate getParameters call
--      assert(self.params:storage() == self.segments[1]:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end
   train_top1 = top1Sum/N
   train_top5 = top5Sum/N
   train_loss = lossSum/N
   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local N = 0

   for i=1,table.getn(self.segments) do
      self.segments[i]:evaluate()
   end
   for i=1,table.getn(self.fcs) do
      self.fcs[i]:evaluate()
   end

   local nOutputs = table.getn(self.pivotSet) + 1
   local nCropTop1set = {}
   local nCropTop5set = {}
   local singleCropTop1set = {}
   local singleCropTop5set = {}
   for i=1, nOutputs do
      table.insert(nCropTop1set, 0)
      table.insert(nCropTop5set, 0)
      table.insert(singleCropTop1set, 0)
      table.insert(singleCropTop5set, 0)
   end

   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local outputSet = {}
      local fcOutputSet = {}
      local temp_input = self.input
      for i=1, nOutputs-1 do
         local temp_output = self.segments[i]:forward(temp_input)
         local temp_fc_output = self.fcs[i]:forward(temp_output)
         table.insert(outputSet, temp_output)
         table.insert(fcOutputSet, temp_fc_output)
         temp_input = temp_output
      end
      local output = self.segments[nOutputs]:forward(outputSet[nOutputs-1])

      local top1, top5 = 0, 0
      for i=1, nOutputs do
         top1, top5 = self:computeScore((fcOutputSet[i] or output):float(), sample.target, nCrops)
         nCropTop1set[i] = nCropTop1set[i] + top1
         nCropTop5set[i] = nCropTop5set[i] + top5
      end
      if nCrops > 1 then
         for i=1, nOutputs do
            top1, top5 = self:singleCropComputeScore((fcOutputSet[i] or output):float(), sample.target, nCrops)
            singleCropTop1set[i] = singleCropTop1set[i] + top1
            singleCropTop5set[i] = singleCropTop5set[i] + top5
         end
      end
      N = N + 1

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, nCropTop1set[nOutputs]/N, nCropTop5set[nOutputs]/N, singleCropTop1set[nOutputs]/N, singleCropTop5set[nOutputs]/N))

      timer:reset()
      dataTimer:reset()
   end

   for i=1,table.getn(self.segments) do
      self.segments[i]:training()
   end
   for i=1,table.getn(self.fcs) do
      self.fcs[i]:training()
   end

   print((' * Finished epoch # %d     nCropTop1: %7.3f  nCropTop5: %7.3f  singleTop1: %7.3f  singleTop5: %7.3f\n'):format(
      epoch, nCropTop1set[nOutputs]/N, nCropTop5set[nOutputs]/N, singleCropTop1set[nOutputs]/N, singleCropTop5set[nOutputs]/N))

      if self.testLog then
         local stringFormat = ''
         for i=1, nOutputs do
            stringFormat = stringFormat .. '\t %f'
            nCropTop1set[i] = nCropTop1set[i]/N
            nCropTop5set[i] = nCropTop5set[i]/N
            singleCropTop1set[i] = singleCropTop1set[i]/N
            singleCropTop5set[i] = singleCropTop5set[i]/N
         end

         local text = nil
         local stringPrefix = ('%f \t %f \t %f'):format(train_top1, train_top5, train_loss)
         local nCropTop1 = stringFormat:format(table.unpack(nCropTop1set))
         local nCropTop5 = stringFormat:format(table.unpack(nCropTop5set))
         local singleCropTop1 = stringFormat:format(table.unpack(singleCropTop1set))
         local singleCropTop5 = stringFormat:format(table.unpack(singleCropTop5set))
         text = stringPrefix .. nCropTop1 .. nCropTop5 .. singleCropTop1 .. singleCropTop5 .. ' \n'

         local testLog = self.testLog:write(text)
        testLog:flush()
      end

   return nCropTop1set[nOutputs], nCropTop5set[nOutputs]
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:singleCropComputeScore(output, target, nCrops)
   if nCrops > 1 then
      for i=1, target:size(1) do
         output[i] = output[(i-1)*nCrops + 1]
      end
      output:resize(target:size(1), output:size(2))
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' or self.opt.dataset == 'imagenet100' then
      -- decay = math.floor((epoch - 1) / 30)
      -- linear learning rate schedule
      local gamma = self.opt.gamma
      local power = (epoch-1) / self.opt.step
      return self.opt.LR * math.pow(gamma, torch.floor(power))
   elseif self.opt.dataset == 'cifar10' or self.opt.dataset == 'cifar100'or self.opt.dataset == 'svhn' then
--      decay = epoch >= 121 and 2 or epoch >= 81 and 1 or 0
      decay = epoch > self.opt.nEpochs*0.6 and 2 or epoch > self.opt.nEpochs*0.4 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
