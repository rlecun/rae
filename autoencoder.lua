----------------------------------------------------------------------
-- autoencoder class (possibly with regularizer on code)

require 'nn'

function nn.Linear:normalize()
    for i=1,self.weight:size(2) do
        local col = self.weight:select(2, i)
        local norm = torch.norm(col)
        if norm > 1 then torch.mul(col, col, 1/norm) end
    end
end

local autoencoder, parent = torch.class('nn.autoencoder', 'nn.Module')

function autoencoder:__init(encoder, decoder, cost)
   parent.__init(self)
   self.encoder = encoder   -- encoder module
   self.decoder = decoder   -- decoder module
   self.cost = cost         -- reconstruction cost module
   self.code = 0            -- code vector
   self.gradcode = 0        -- gradient w.r.t. code
   self.recons = 0          -- reconstruction
   self.gradrecons = 0      -- gradient w.r.t. reconstruction
   self.energy = 0          -- reconstruction energy
end

function autoencoder:updateOutput(input)
   self.code = self.encoder:updateOutput(input)
   self.recons = self.decoder:updateOutput(self.code)
   self.energy = self.cost:updateOutput(self.recons, input)
   return self.energy
end

function autoencoder:updateGradInput(input)
   self.gradrecons = self.cost:updateGradInput(self.recons, input)
   self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons)
   return self.encoder:updateGradInput(input, self.gradcode)
end

function autoencoder:accGradParameters(input)
   self.gradrecons = self.cost:updateGradInput(self.recons, input)
   -- self.gradrecons = self.cost:accGradParameters(recons, input)
   self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons)
   self.decoder:accGradParameters(self.code, self.gradrecons)
   self.encoder:accGradParameters(input, self.gradcode)
end

function autoencoder:normalize()
    return self.decoder:normalize()
end

-- collect the parameters so they can be flattened
-- this assumes that the cost doesn't have parameters.
function autoencoder:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do tinsert(to,from[i]) end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   local mw,mgw = self.encoder:parameters()
   if mw then tinsert(w,mw) tinsert(gw,mgw) end
   local mw,mgw = self.decoder:parameters()
   if mw then tinsert(w,mw) tinsert(gw,mgw) end
   return w,gw
end


function autoencoder:weights()
  return decoder.weight, module.encoder:weights()
end

function nn.Sequential:weights()
  return module.encoder.modules[1].weight
end

-- args: number columns of filters, 
-- width of each filter, height of each filter.
-- product of width and height must equal dimension of input.
function autoencoder:displayweights(ncol,w,h)
  local dw,ew = self.weights()
  dw = dw:transpose(1,2):unfold(2,h,w)
  ew = ew:unfold(2,h,w)
  dd = image.toDisplayTensor{input=dw,
                             padding=2,
                             nrow=ncol,
                             symmetric=true}
  de = image.toDisplayTensor{input=ew,
                             padding=2,
                             nrow=ncol,
                             symmetric=true}
  return dd,de
end

------------------------------------------------------------------

-- an auto-encoder with a regularizer on the code vector
local regautoencoder, parent = 
  torch.class('nn.regautoencoder', 'nn.autoencoder')

function regautoencoder:__init(encoder, decoder, cost, regularizer)
   parent.__init(self)
   self.encoder = encoder
   self.decoder = decoder
   self.cost = cost
   self.regularizer = regularizer   -- regularizer module
   self.code = 0
   self.gradcode = 0
   self.recons = 0
   self.gradrecons = 0
   self.recenergy = 0       -- reconstruction energy
   self.regenergy = 0       -- regularizer energy
   self.energy = 0          -- total energy (sum of the above)
end

function regautoencoder:updateOutput(input)
   self.code = self.encoder:updateOutput(input)
   self.regenergy = self.regularizer:updateOutput(self.code)
   self.recons = self.decoder:updateOutput(self.code)
   self.recenergy = self.cost:updateOutput(self.recons, input)
   self.energy = self.regenergy + self.recenergy
   return self.energy
end

function regautoencoder:updateGradInput(input)
   self.gradrecons = self.cost:updateGradInput(self.recons, input)
   self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons) +
                   self.regularizer:updateGradInput(self.code)
   return self.encoder:updateGradInput(input, self.gradcode)
end

function regautoencoder:accGradParameters(input)
   self.gradrecons = self.cost:updateGradInput(self.recons, input)
   -- self.gradrecons = self.cost:accGradParameters(recons, input)
   self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons) +
                   self.regularizer:updateGradInput(self.code)
   self.decoder:accGradParameters(self.code, self.gradrecons)
   self.encoder:updateGradInput(input, self.gradcode)
   self.encoder:accGradParameters(input, self.gradcode)
end


