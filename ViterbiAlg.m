% ViterbiNet example code - ISI channel with AWGN
clear all; % Clear all variables
close all; % Close all figures
clc; % Clear command window

rng(1); % Set random seed

%% Parameters setting
s_nConst = 2; % Constellation size (2 = BPSK)
s_nMemSize = 4; % Number of taps
s_fTestSize = 10; % Test data size

s_nStates = s_nConst^s_nMemSize; % Number of states = 2^4 = 16

v_fSigWdB = 8:2:10; % Noise variance in dB, total number of SNR values = 9

% ==============================================================
decay = 1;
% Exponentially decaying channel
v_fChannel = exp(decay*(0:(s_nMemSize-1)));

% Initialize the symbol error rate (SER) vectors
v_fSER_alg = zeros(length(v_fSigWdB), 1);

% Generate the test labels
v_fXtest = randi(s_nConst, 1, s_fTestSize); % Generate random integers from 1 to s_nConst
v_fStest = 2*(v_fXtest - 0.5*(s_nConst+1)); % Generate the test labels (BPSK)
m_fStest = m_fMyReshape(v_fStest, s_nMemSize); % Reshape the test labels
v_Rtest = fliplr(v_fChannel) * m_fStest; % Multiply the test labels by the flipped channel

%% Loop over the SNR values
for mm=1:length(v_fSigWdB)
    s_fSigmaW = 10^(-0.1*v_fSigWdB(mm)); % Calculate the noise variance, equal to 10^(-0.1*SNR)
    v_fYtest = v_Rtest + sqrt(s_fSigmaW)*randn(size(v_Rtest)); % Add noise to the test labels

    % Viterbi Algorithm
    m_fLikelihood = zeros(s_fTestSize, s_nStates); % Initialize the likelihood matrix
    % Compute conditional PDF for each state
    for ii = 1:s_nStates
        v_fX = zeros(s_nMemSize, 1); % Initialize the X vector
        Idx = ii - 1; % Initialize the index
        % Generate the X vector
        for ll=1:s_nMemSize % Loop over the number of taps
            v_fX(ll) = mod(Idx, s_nConst); % Calculate the modulo of the index and the constellation size
            Idx = floor(Idx/s_nConst); % Calculate the floor of the index divided by the constellation size
        end
        v_fS = 2*(v_fX - 0.5*(s_nConst+1)); % Calculate the S vector
        m_fLikelihood(:, ii) = normpdf(v_fYtest' - fliplr(v_fChannel)*v_fS,0,s_fSigmaW); % Calculate the likelihood
        PrintMatrix(m_fLikelihood);
    end

    % Apply Viterbi detection based on computed likelihoods
    v_fXhat = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize); % Apply the Viterbi detection
    % Evaluate error rate
    v_fSER_alg(mm) = mean(v_fXhat ~= v_fXtest); % Calculate the symbol error rate (SER=mean(Xhat~=Xtest))

    % Display SNR index
    disp(['SNR index: ', num2str(mm)]);
end

% Plot the results
% figure;
% semilogy(v_fSigWdB, v_fSER_alg, 'r', 'LineWidth', 2);
% hold on;
% grid on;
% xlabel('SNR (dB)');
% ylabel('SER');
% legend('Viterbi Algorithm');
% ==============================================================

