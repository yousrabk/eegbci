require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'xlua'

require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-data_dir','data/preprocessed','directory containing sampled files')
cmd:option('-submission',false,'sample test set instead of validation set')
cmd:option('-calc_roc',false,'')
cmd:text()

-- clear old sampled files
os.execute('rm -rf tmp/sampled_files')
lfs.mkdir('tmp/sampled_files')

-- parse input params
opt = cmd:parse(arg)

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
if checkpoint.type == "lstm" then
  require 'lstm.sampler'
  sampler = LSTMSampler()
else
  require 'cnn.sampler'
  sampler = CNNSampler()
end
sampler:load_model(checkpoint, opt)

local info_file = io.open('data/preprocessed/info', 'r')
local data_info = info_file:read("*all"):split('\n')
subsample = tonumber(data_info[2])
info_file:close()

-- start sampling

if opt.submission then
    orig_dir = 'data/test/'
    suffix = 'data.csv.test'
else
    orig_dir = 'data/train/'
    suffix = 'data.csv.val'
end

for file in lfs.dir(opt.data_dir) do
    if file:find(suffix) then
        print(file)

        -- count samples in original file
        local orig_num_samples = -1 -- -1 for the header
        local orig_name = opt.submission and file:sub(1, -6) or file:sub(1, -5)
        for _ in io.lines(orig_dir .. orig_name) do
            orig_num_samples = orig_num_samples + 1
        end

        -- load the data
        local data_table = {}
        local data_fh = io.open(path.join(opt.data_dir, file))
        local data_content = data_fh:read('*all'):split('\n')
        data_fh:close()

        -- parse data file
        for i,line in ipairs(data_content) do
            if i > 1 then -- ignore header
                local fields = line:split(',')
                table.remove(fields, 1)
                table.insert(data_table, fields)
            end
        end

        -- create data tensor
        local data_tensor = torch.Tensor(data_table)
        data_content = nil
        collectgarbage()
        if opt.gpuid >= 0 then data_tensor = data_tensor:cuda() end
        local num_samples = data_tensor:size(1)

        local out_file = io.open('tmp/sampled_files/' .. file, 'w')
        local lines_written = sampler:prepare_file(out_file)

        local line
        for t = lines_written + 1, num_samples do
            if t % 100 == 0 then
              xlua.progress(t, num_samples)
            end
            if t % 1000 == 0 then
                out_file:flush()
                collectgarbage()
            end

            -- generate prediction and next state
            local prediction = sampler:predict(t, data_tensor)

            -- save properly formatted output
            line = ""
            for i = 1,prediction:size(2) do
                if i > 1 then
                    line = line .. ','
                end

                line = line .. string.format('%.5f', prediction[1][i])
            end
            line = line .. '\n'

            -- don't write too many lines
            for i = 1,math.min(subsample, orig_num_samples - t * subsample) do
                out_file:write(line)
                lines_written = lines_written + 1
            end
        end

        -- fill the remaining samples with last prediction
        while lines_written < orig_num_samples do
            out_file:write(line)
            lines_written = lines_written + 1
        end

        out_file:close()
        xlua.progress(num_samples, num_samples)
        collectgarbage()
    end
end

if not opt.submission and opt.calc_roc then
  print("calculating ROC")
  os.execute('python3 python_utils/calc_roc.py')
end
