%% Simulation parameters
M = 2;               % Modulation order
bitsPerIter = 1.2e4; % Number of bits to simulate
EbNo = 1.5:0.5:5;            % Information bit Eb/No in dB

%% Coding properties
codeRate = 1/2;          % Code rate of convolutional encoder

SNR = convertSNR( ...
    EbNo,"ebno", ...
    "BitsPerSymbol",log2(M), ...
    "CodingRate",codeRate);

[rxSig, nVar] = awgn(0, SNR(2));

SNRdB = 1./10.^(0.1.*SNR);