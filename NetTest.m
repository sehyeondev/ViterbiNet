% Get train, test data
load('v_fYtest.mat')
load('m_fXtrain.mat')
load('v_fYtrain.mat')

% Generate Neural Network
inputSize = 1;
numHiddenUnits = 100;
numClasses = s_nStates;

LSTMLayer = lstmLayer(numHiddenUnits,'OutputMode','last'... 
    , 'RecurrentWeightsLearnRateFactor', 0 ...
    , 'RecurrentWeightsL2Factor', 0 ...
    );
LSTMLayer.RecurrentWeights = zeros(4*numHiddenUnits,numHiddenUnits);
layers = [ ...
    sequenceInputLayer(inputSize)
    LSTMLayer
    fullyConnectedLayer(floor(numHiddenUnits/2))
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Preprocess training data
% Combine each set of inputs as a single unique category
v_fCombineVec = s_nConst.^(0:s_nMemSize-1);
% format training to comply with Matlab's deep learning toolbox settings
v_fXcat = categorical((v_fCombineVec*(m_fXtrain-1))');
v_fYcat = num2cell(v_fYtrain');

% Train Neural Network
learnRate = 0.01;
maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ... 
    'ExecutionEnvironment','cpu', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false);
    % 'Plots','training-progress' ...
    % ); % This can be unmasked to display training convergence

net = trainNetwork(v_fYcat,v_fXcat,layers,options);

% Test Neural Network
m_fpS_Y = predict(net,num2cell(v_fYtest'));