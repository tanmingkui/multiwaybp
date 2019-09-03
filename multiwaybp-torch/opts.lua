local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 AuxResNet Testing script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------

   cmd:option('-dataset',    'cifar10', 'Options: cifar10 | cifar100 | svhn')
   cmd:option('-manualSeed', 1,          'Manually set RNG seed')
   cmd:option('-pretrain',        'pretrain',      'Path to save trained models')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        10, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-batchSize',       128,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'true', 'Run on validation set only')
   ---------- Model options ----------------------------------
   cmd:option('-model',      'none',   'Path to the trained model')
   cmd:option('-outputs',   'none',   'Path to an outputs file')

   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'

   local modelList = {
      ['cifar10-auxnet-56-2'] = '',
      ['cifar10-auxnet-56-5'] = '',
      ['cifar10-auxnet-26-2-wide-10'] = '',
      ['cifar100-auxnet-56-2'] = '',
      ['cifar100-auxnet-56-5'] = '',
      ['cifar100-auxnet-26-2-wide-10'] = '',
      ['svhn-auxnet-56-3-dropout'] = '',
   }
   local outputsList = {
      ['cifar10-auxnet-56-5-outputs'] = '',
   }
   if opt.model ~= 'none' then
      assert(modelList[opt.model], 'Invalid model: ' .. opt.model)
   end
   if opt.outputs ~= 'none' then
      assert(outputsList[opt.outputs], 'Invalid outputs: ' .. opt.outputs)
   end
   return opt
end

return M