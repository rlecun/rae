require('nn')

function makeS(size, band)
    S = torch.zeros(size, size)
    local v = 1 / (size - 1)
    if band then v = 1 / (2 * band) end
    if not band then
        for i = 1, size do
            for j = 1, size do
                if i ~= j then S[i][j] = v end
            end
        end
    else
       
        for i = 1, size do
            for j = i - band, i + band do
                if j < 1 then S[i][j - 1] = v
                elseif j > size then S[i][j - size] = v
                else S[i][j] = v end
            end
        end
    end
    return S
end

local SMatrixCriterion, parent = torch.class('nn.SMatrixCriterion', 'nn.Criterion')

function SMatrixCriterion:__init(alpha, S)
    parent.__init(self)
    self.alpha = alpha
    self.S = S
    self.R = torch.add(self.S, self.S:transpose(1, 2))
end

-- calculates the 
function SMatrixCriterion:updateOutput(input)
    local inputabs = torch.abs(input)
    self.output = self.alpha * torch.dot(inputabs, torch.mv(self.S, inputabs))
    return self.output
end

-- calculates the gradient of the 
function SMatrixCriterion:updateGradInput(input)
    self.gradInput = torch.mul(torch.cmul(torch.sign(input), torch.mv(self.R, torch.abs(input))), self.alpha)
    return self.gradInput
end
