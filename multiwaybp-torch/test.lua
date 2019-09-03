require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'

local DataLoader = require 'dataloader'
local opts = require 'opts'
local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

local criterion = nn.CrossEntropyCriterion():cuda() 
local modelPath = paths.concat(opt.pretrain, opt.model .. '.t7')
local segments = torch.load(modelPath)
-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)
local dataloader = valLoader
local epoch = 0

local input = nil
local target = nil

local function computeScore(output, target, nCrops)
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

local function copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   local input = input or torch.CudaTensor()
   local target = target or torch.CudaTensor()

   input:resize(sample.input:size()):copy(sample.input)
   target:resize(sample.target:size()):copy(sample.target)
   return input, target
end


   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   for i=1,table.getn(segments) do
      segments[i]:evaluate()
      segments[i]:cuda()
   end

   -- the model is seperated into n segments w.r.t n outputs
   local nOutputs = table.getn(segments)


   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      local input, target = copyInputs(sample)

      local outputSet = {}

      local temp_input = input
      -- all the trained auxiliary outputs are saved in an individual file, but the final output is directly incorporated into the last segment
      for i=1, nOutputs-1 do
         local temp_output = segments[i]:forward(temp_input)
         table.insert(outputSet, temp_output)
         temp_input = temp_output
      end
      local output = segments[nOutputs]:forward(outputSet[nOutputs-1])
      local loss = criterion:forward(segments[nOutputs].output, target)

      local top1, top5 = computeScore(output:float(), target, nCrops)
      top1Sum = top1Sum + top1
      top5Sum = top5Sum + top5      

      N = N + 1
      
      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum/N, top5, top5Sum/N))

      timer:reset()
      dataTimer:reset()
   end


   print((' * Finished epoch # %d     nCropTop1: %7.3f  nCropTop5: %7.3f\n'):format(
      epoch, top1Sum/N, top5Sum/N))

