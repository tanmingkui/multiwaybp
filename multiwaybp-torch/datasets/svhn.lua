--
--  SVHN dataset loader
--

local t = require 'datasets/transforms'

local M = {}
local SvhnDataset = torch.class('resnet.SvhnDataset', M)

function SvhnDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function SvhnDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function SvhnDataset:size()
   return self.imageInfo.data:size(1)
end

local meanstd = {
   mean = {0, 0, 0},
   std  = {255, 255, 255},
}


function SvhnDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.ColorNormalize(meanstd),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.SvhnDataset
