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

convEncoder = comm.ConvolutionalEncoder( ...
    'TerminationMethod','Truncated');

mcCount = 1;
numData = 5000;
sigWdB = -2:2:12;
v_ber = zeros(1,length(sigWdB));
for mm=1:mcCount
    v_u = randi(2,1,numData)-1;
    % v_u = [1 1 0 1 0 0];
    v_x = convenc(v_u, trellis);
    % m_x = reshape(codewords, trellis.numInputSymbols, length(v_u))';

    for ww=1:length(sigWdB)
        sigW = 10.^(-0.1.*sigWdB(ww));
        % Received symbols
        v_y = v_x + randn(size(v_x))*sqrt(sigW);
        % m_y = m_x + randn(size(m_x))*sqrt(sigW);
        % v_y = reshape(m_y',1,numData/coderate);
        % m_y = [0.3  0.1;
        %       -0.5  0.2;
        %        0.8  0.5;
        %       -0.5  0.3;
        %        0.1 -0.7;
        %        1.5 -0.4];
        
        % llr = BCJR_conv(v_y,trellis,sqrt(sigW));
        llr = bcjrAlg(v_y,trellis,sigW);
        
        v_uHat = sign(llr)/2 + 0.5;
        ber = mean(v_u~=v_uHat);
        v_ber(ww)= v_ber(ww) + ber;
    end
end
v_ber = v_ber/mcCount;

fig1 = figure;
set(fig1, 'WindowStyle', 'docked');

semilogy(sigWdB,v_ber, '-rs', 'LineWidth',1,'MarkerSize',10);
xlabel('SNR [dB]');
ylabel('Symbol error rate');
grid on;
legend(['BCJR Algorithm'],'Location','SouthWest');
hold off;