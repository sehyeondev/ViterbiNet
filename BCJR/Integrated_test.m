%% Simulation parameters
M = 2;               % Modulation order
bitsPerIter = 1.2e4; % Number of bits to simulate
SNR = -6:2:10;

%% Coding properties
codeRate = 1/2;          % Code rate of convolutional encoder
constLen = 3;            % Constraint length of encoder
codeGenPoly = [6 7]; % Code generator polynomial of encoder
tblen = 10;              % Traceback depth of Viterbi decoder
trellis = poly2trellis(constLen,codeGenPoly);

%% Define encoder
enc = comm.ConvolutionalEncoder(trellis);

%% Define decoders
decHard = comm.ViterbiDecoder(trellis,'InputFormat','Hard', ...
    'TracebackDepth',tblen);

decUnquant = comm.ViterbiDecoder(trellis,'InputFormat','Unquantized', ...
    'TracebackDepth',tblen);

decSoft = comm.ViterbiDecoder(trellis,'InputFormat','Soft', ...
    'SoftInputWordLength',3,'TracebackDepth',tblen);

decAPP = comm.APPDecoder( ...
    'TrellisStructure',trellis, ...
    'Algorithm','True APP', ...
    'CodedBitLLROutputPort',false);

%% Define ErrorRates
errHard = comm.ErrorRate('ReceiveDelay',tblen);
errUnquant = comm.ErrorRate('ReceiveDelay',tblen);
errSoft = comm.ErrorRate('ReceiveDelay',tblen);
errAPP = comm.ErrorRate;
errBCJR = comm.ErrorRate;
errMyBCJR = comm.ErrorRate;

%% System simulation
ber = zeros(6,length(SNR));
maxIter = 30;

for ii=1:length(SNR)
    reset(errHard)
    reset(errUnquant)
    reset(errSoft)
    reset(errAPP)
    reset(errBCJR)
    reset(errMyBCJR)
    reset(enc)
    reset(decHard)
    reset(decUnquant)
    reset(decSoft)
    reset(decAPP)
 
    for mm=1:maxIter
        txData = randi([0 1],bitsPerIter,1); 
        
        % Encoding
        encData = enc(txData);
        
        % Modulation
        modData = pskmod(encData,M,0,InputType="bit");
        
        % Channel
        [rxSig,nVar] = awgn(modData,SNR(ii));

        % Demodulation
        hardData = pskdemod(rxSig,M,0,OutputType="bit");
        LLRData = pskdemod(rxSig,M,0,OutputType="llr",NoiseVariance=nVar);
        
        %%% Decoding %%%
        % Hard Viterbi
        rxDataHard = decHard(hardData);
        berHard = errHard(txData,rxDataHard);
        
        % Unquant Viterbi
        rxDataUnquant = decUnquant(LLRData);
        berUnquant = errUnquant(txData,rxDataUnquant);
        
        % Soft Viterbi
        partitionPoints = (-2.1:.7:2.1)/nVar;
        quantizedValue = quantiz(-LLRData,partitionPoints);
        rxDataSoft = decSoft(double(quantizedValue));
        berSoft = errSoft(txData,rxDataSoft);
    
        % APP Decoder
        rxDataAPP = double(decAPP(zeros(bitsPerIter,1),-LLRData) > 0);
        berAPP = errAPP(txData,rxDataAPP);

        % BCJR reference
        % rxDataBCJR = double(BCJR_ref(hardData',trellis,sqrt(nVar)) < 0);
        % berBCJR = errBCJR(txData,rxDataBCJR');
        
        % My BCJR
        rxDataMyBCJR = double(bcjrAlg(-LLRData',trellis,sqrt(nVar)) > 0);
        berMyBCJR = errMyBCJR(txData,rxDataMyBCJR');

        %%% Store BER %%%
        ber(1,ii) = berHard(1);
        ber(2,ii) = berUnquant(1);
        ber(3,ii) = berSoft(1);
        ber(4,ii) = berAPP(1);
        % ber(5,ii) = berBCJR(1);
        ber(6,ii) = berMyBCJR(1);
    end
end
% ber = ber/mcCount;
    
%% Plot
fig = figure;
grid on;
ax = fig.CurrentAxes;
hold(ax,'on');
ax.YScale = 'log';
xlim(ax, [SNR(1), SNR(end)]); ylim(ax, [1e-6 1]);
xlabel(ax,'SNR [dB]'); ylabel(ax, 'BER');
title(ax,'LLR vs. Hard Decision Demodulation');
fig.NumberTitle = 'off';
set(fig,'DefaultLegendAutoUpdate','off');

% Perform curve fitting and plot the results
BER_HD  = ber(1,:);
BER_LLR = ber(2,:);
BER_SD = ber(3,:);
BER_APP = ber(4,:);
% BER_BCJR = ber(5,:);
BER_MyBCJR = ber(6,:);

semilogy(ax,SNR,BER_HD, 'g*-', ...
            SNR,BER_LLR,'k*-', ...
            SNR,BER_APP,'b*-', ...
            SNR,BER_MyBCJR,'ro-');
hold(ax,'off');

legend('Viterbi Hard', ...
       'Viterbi Unquant',...
       'APP',...
       'My BCJR',...
       'Location', 'SouthWest');


