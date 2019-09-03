--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  CIFAR-10 dataset loader
--

local t = require 'datasets/transforms'

local M = {}
local CifarDatasetHundred = torch.class('resnet.CifarDatasetHundred', M)

function CifarDatasetHundred:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function CifarDatasetHundred:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function CifarDatasetHundred:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire CIFAR-10 training set
local meanstd = {
   mean = {129.3, 124.1, 112.4},
   std  = {68.2,  65.4,  70.4},
}

function CifarDatasetHundred:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CifarDatasetHundred
