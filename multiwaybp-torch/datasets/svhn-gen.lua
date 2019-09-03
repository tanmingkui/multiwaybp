--
--  This automatically downloads the SVHN dataset from
--  http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
--
local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/svhn.t7.tgz'

local M = {}

function M.exec(opt, cacheFile)
   print("=> Downloading SVHN dataset from " .. URL)
   
   local ok = os.execute('curl ' .. URL .. ' | tar xz -C  gen/')
   assert(ok == true or ok == 0, 'error downloading SVHN')

   local train = torch.load('gen/housenumbers/train_32x32.t7','ascii')
   local extra = torch.load('gen/housenumbers/extra_32x32.t7','ascii')
   local test  = torch.load('gen/housenumbers/test_32x32.t7', 'ascii')

   print(" | saving SVHN dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = {
         data = torch.cat(train.X:transpose(3,4), extra.X:transpose(3,4),1),
         labels = torch.cat(train.y[1], extra.y[1], 1),
      },
      val = {
         data = test.X:transpose(3,4),
         labels = test.y[1],
      },
   })
end

return M
