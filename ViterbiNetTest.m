s_nConst = 2;       % Constellation size (2 = BPSK)
s_nMemSize = 4;     % Number of taps
s_fTrainSize = 500; % Training size
s_fTestSize = 10; % Test data size

s_nStates = s_nConst^s_nMemSize;

% v_fExps =  0.1:0.1:2;
v_fExps =  0.5:0.1:0.5;
% v_fSigWdB=   -2:2:6;  %Noise variance in dB
v_fSigWdB=   2:2:2;  %Noise variance in dB

% s_fSigmaW = 10^(-0.1*2); % Noise variance of LTI AWGN channel
% s_fChannelExp = 0.5;
v_fSERAvg = zeros(1,length(v_fSigWdB));
for wIdx=1:length(v_fSigWdB)
    s_fSigmaWdB = v_fSigWdB(wIdx);
    s_fSigmaW = 10^(-0.1*s_fSigmaWdB); % Noise variance of LTI AWGN channel
    s_fSERAvg = 0;
    for eIdx=1:length(v_fExps)
        s_fChannelExp = v_fExps(eIdx);
        v_fChannel = exp(-s_fChannelExp*(0:(s_nMemSize-1)));
        
        % Generate training labels
        v_fXtrain = randi(s_nConst,1,s_fTrainSize);
        v_fStrain = 2*(v_fXtrain - 0.5*(s_nConst+1));
        m_fXtrain = m_fMyReshape(v_fXtrain, s_nMemSize);
        m_fStrain = m_fMyReshape(v_fStrain, s_nMemSize);
        v_Rtrain = fliplr(v_fChannel) * m_fStrain;          % perfect CSI
        v_fYtrain = v_Rtrain + sqrt(s_fSigmaW)*randn(size(v_Rtrain));
        
        % Generate test labels
        v_fXtest = randi(s_nConst,1,s_fTestSize);
        v_fStest = 2*(v_fXtest - 0.5*(s_nConst+1));
        m_fStest= m_fMyReshape(v_fStest, s_nMemSize);
        v_Rtest = fliplr(v_fChannel) * m_fStest;
        v_fYtest = v_Rtest + sqrt(s_fSigmaW)*randn(size(v_Rtest));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Generate neural network
        inputSize = 1;
        numHiddenUnits = 100;
        numClasses = s_nStates;
        
        % Generate network model
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
        
        % Combine each set of inputs as a single unique category
        v_fCombineVec = s_nConst.^(0:s_nMemSize-1);
        % format training to comply with Matlab's deep learning toolbox settings
        v_fXcat = categorical((v_fCombineVec*(m_fXtrain-1))');
        v_fYcat = num2cell(v_fYtrain');
        
        % Network parameters
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
        
        % Train netowrk
        net = trainNetwork(v_fYcat,v_fXcat,layers,options);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Use network to compute likelihood function
        m_fpS_Y = predict(net,num2cell(v_fYtest'));
        
        % Compute output PDF using GMM fitting
        s_nMixtureSize = s_nStates;
        GMModel = fitgmdist(v_fYtrain',s_nMixtureSize,'RegularizationValue',0.1);
        v_fpY = pdf(GMModel, v_fYtest');
        
        % Compute likelihoods
        m_fLikelihood = (m_fpS_Y .* v_fpY)*s_nStates;       
        
        % Apply Viterbi output layer
        v_fXhat = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize);
        
        % Evaluate error rate
        s_fSER = mean(v_fXhat ~= v_fXtest);
        s_fSERAvg = s_fSERAvg + s_fSER;
        s_fChannelExp
    end
    s_fSERAvg = s_fSERAvg/length(v_fExps);
    v_fSERAvg(1,wIdx) = s_fSERAvg;
    s_fSigmaWdB
end
semilogy(v_fSigWdB, v_fSERAvg, '-rs','LineWidth',1,'MarkerSize',10)