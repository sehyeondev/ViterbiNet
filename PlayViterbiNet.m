% ViterbiNet example code - ISI channel with AWGN
clear all; % Clear all variables
close all; % Close all figures
clc; % Clear command window

rng(1); % Set random seed

%% Parameters setting
s_nConst = 2; % Constellation size (2 = BPSK)
s_nMemSize = 4; % Number of taps
s_fTrainSize = 5000; % Training size
s_fTestSize = 50000; % Test data size

s_nStates = s_nConst^s_nMemSize; % Number of states = 2^4 = 16

v_fSigWdB = -6:2:10; % Noise variance in dB, total number of SNR values = 9

s_nMixtureSize = s_nStates; % Mixture size = Number of states = 16

% ==============================================================
decay = 1;
% Exponentially decaying channel
v_fChannel = exp(decay*(0:(s_nMemSize-1)));

v_fSER_alg = zeros(length(v_fSigWdB), 1);
v_fSER_net = zeros(length(v_fSigWdB), 1);

% Generate training labels
v_fXtrain = randi(s_nConst, 1, s_fTrainSize); 
v_fStrain = 2*(v_fXtrain - 0.5*(s_nConst+1)); 
m_fStrain = m_fMyReshape(v_fStrain, s_nMemSize);
v_Rtrain = fliplr(v_fChannel) * m_fStrain;

% Generate the test labels
v_fXtest = randi(s_nConst, 1, s_fTestSize); % Generate random integers from 1 to s_nConst
v_fStest = 2*(v_fXtest - 0.5*(s_nConst+1)); % Generate the test labels (BPSK)
m_fStest = m_fMyReshape(v_fStest, s_nMemSize); % Reshape the test labels
v_Rtest = fliplr(v_fChannel) * m_fStest; % Multiply the test labels by the flipped channel


% % Print the channel
% disp(['Channel: ', num2str(v_fChannel)]);
% % Print the training labels
% disp(['Training labels: ', num2str(v_fStrain)]);
% % Print the training labels matrix
% % disp(['Training labels matrix: ', num2str(m_fStrain)]);
% % Print the flipped channel
% disp(['Flipped channel: ', num2str(fliplr(v_fChannel))]);
% % Print the training labels matrix multiplied by the flipped channel
% disp(['Training labels matrix multiplied by the flipped channel: ', num2str(v_Rtrain)]);


%% Loop over the SNR values
for mm=1:length(v_fSigWdB)
    s_fSigmaW = 10^(-0.1*v_fSigWdB(mm)); % Calculate the noise variance
    v_fYtrain = v_Rtrain + sqrt(s_fSigmaW)*randn(size(v_Rtrain)); % Add noise to the training labels
    v_fYtest = v_Rtest + sqrt(s_fSigmaW)*randn(size(v_Rtest)); % Add noise to the test labels

    % ViterbiNet
    % Train network
    [net, GMModel] = GetMyViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize, s_nMixtureSize);
    % Apply ViterbiNet detector
    v_fXhat = ApplyViterbiNet(v_fYtest, net, GMModel, s_nConst, s_nMemSize);
    % Evaluate error rate
    v_fSER_net(mm) = mean(v_fXhat ~= v_fXtest); % Calculate the symbol error rate (SER=mean(Xhat~=Xtest)) 

    % Viterbi Algorithm
    m_fLikelihood = zeros(s_fTestSize, s_nStates); % Initialize the likelihood matrix
    % Compute conditional PDF for each state
    for ii = 1:s_nStates
        v_fX = zeros(s_nMemSize, 1); % Initialize the X vector
        Idx = ii - 1; % Initialize the index
        for ll=1:s_nMemSize % Loop over the number of taps
            v_fX(ll) = mod(Idx, s_nConst); % Calculate the modulo of the index and the constellation size
            Idx = floor(Idx/s_nConst); % Calculate the floor of the index divided by the constellation size
        end
        v_fS = 2*(v_fX - 0.5*(s_nConst+1)); % Calculate the S vector
        m_fLikelihood(:, ii) = normpdf(v_fYtest' - fliplr(v_fChannel)*v_fS,0,s_fSigmaW); % Calculate the likelihood
    end
    % Apply Viterbi detection based on computed likelihoods
    v_fXhat = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize); % Apply the Viterbi detection
    % Evaluate error rate
    v_fSER_alg(mm) = mean(v_fXhat ~= v_fXtest); % Calculate the symbol error rate (SER=mean(Xhat~=Xtest))

    % Display SNR index
    disp(['SNR index: ', num2str(mm)]);
end

% Plot the results
figure;
semilogy(v_fSigWdB, v_fSER_alg, 'r', 'LineWidth', 2);
hold on;
semilogy(v_fSigWdB, v_fSER_net, 'b', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('SER');
legend('Viterbi Algorithm', 'ViterbiNet');
title('ViterbiNet example code - ISI channel with AWGN');
% ==============================================================