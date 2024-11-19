% BCJR algorithm with AWGN channel
clear all;
close all;
clc;

rng(1);

% Generate trellis
trellis = struct('numInputSymbols',2,'numOutputSymbols',4, ...
'numStates',4,'nextStates',[0 2;0 2;1 3;1 3], ...
'outputs',[0 3;1 2;3 0;2 1]);
% 'outputs',[0 3;3 0;2 1;1 2]);
coderate = trellis.numInputSymbols/trellis.numOutputSymbols;

% convEncoder = comm.ConvolutionalEncoder( ...
%     'TerminationMethod','Truncated');

appDecoder = comm.APPDecoder(...
    'TrellisStructure',trellis, ...
    'Algorithm','True APP', ...
    'CodedBitLLROutputPort',true);

tblen = 10;
vitDecoder = comm.ViterbiDecoder(trellis,'InputFormat','Soft', ...
    'SoftInputWordLength',1,'TracebackDepth',tblen);
errSoft = comm.ErrorRate('ReceiveDelay',tblen);


mcCount = 1;
numData = 5000;
sigWdB = -2:2:12;
v_ber = zeros(1,length(sigWdB));
v_ber_vit = zeros(1,length(sigWdB));
v_ber_app = zeros(1,length(sigWdB));
for mm=1:mcCount
    v_u = randi([0,1],1,numData);       % message bits
    % v_u = [1 1 0 1 0 0];
    codedBits = convenc(v_u, trellis);  % coded bits
    v_x = 2.*codedBits-1;               % modulated bits
    
    % m_x = reshape(codewords, trellis.numInputSymbols, length(v_u))';

    for ww=1:length(sigWdB)
        sigW = 10.^(-0.1.*sigWdB(ww));
        v_y = v_x + randn(size(v_x))*sqrt(sigW); % received symbols
        v_y_llr = 2 .* v_y ./ sigW;

        % my BCJR decoder
        llr = bcjrAlg(v_y,trellis,sigW); % output of bcjr

        v_uHat = 0.5.*sign(llr) + 0.5;
        ber = mean(v_u~=v_uHat);
        v_ber(ww)= v_ber(ww) + ber;

        % my Viterbi decoder
        demodLLR.Variance = sigW;
        partitionPoints = (-1.5:0.5:1.5)/sigW;
        quantizedValue = quantiz(-v_y_llr',partitionPoints);
        rxDataSoft = vitDecoder(double(quantizedValue));
        ber_vit = errSoft(v_u',rxDataSoft);
        % v_uHat_vit = vitdec(v_y,trellis,sigW); % output of viterbi
        % v_ber_vit = mean(v_u~=v_uHat_vit);
        v_ber_vit(ww) = v_ber_vit(ww) + ber_vit(1);

        % app decoder
        [llr_app, ~] = appDecoder( ...
         zeros(numData,1),v_y_llr');
        v_uHat_app = 0.5.*sign(llr_app)+0.5;
        ber_app = mean(v_u ~= v_uHat_app');
        v_ber_app(ww) = v_ber_app(ww) + ber_app;
    end
end
v_ber = v_ber/mcCount;
v_ber_app = v_ber_app/mcCount;


fig1 = figure;
set(fig1, 'WindowStyle', 'docked');

semilogy(sigWdB,v_ber, '-rs', 'LineWidth',1,'MarkerSize',10);
hold on;
semilogy(sigWdB,v_ber_app, '--go', 'LineWidth',1,'MarkerSize',10);
hold on;
semilogy(sigWdB, v_ber_vit, '-.b^', 'LineWidth',1,'MarkerSize',10);

xlabel('SNR [dB]');
ylabel('Bit error rate');
grid on;
legend(['BCJR Algorithm' 'Viterbi'],'Location','SouthWest');
hold off;