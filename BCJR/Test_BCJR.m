clear all;
close all;
clc;

rng(1);



% Generate trellis
trellis = struct('numInputSymbols',2,'numOutputSymbols',4, ...
'numStates',4,'nextStates',[0 2;0 2;1 3;1 3], ...
'outputs',[0 3;1 2;3 0;2 1]);

appDecoder = comm.APPDecoder(...
    'TrellisStructure',trellis, ...
    'Algorithm','True APP', ...
    'CodedBitLLROutputPort',true);

vitDecoder = comm.ViterbiDecoder('TrellisStructure',trellis);

numData = 5000;
u = randi([0 1], numData, 1);
c = convenc(u, trellis);
x = 2.*c - 1;


sigWdB = -2:2:12;

ber_arr = zeros(3,length(sigWdB));
for ww=1:length(sigWdB)
    sigW = 10.^(-0.1.*sigWdB(ww));
    
    y = x + randn(size(x))*sqrt(sigW);
    y_llr = 2.*y / sigW;
    
    % Decode
    llr_ref = BCJR_conv(y',trellis,sigW);
    u_ref = 0.5.*sign(llr_ref)+0.5;
    
    llr_my = bcjrAlg(y',trellis,sigW);
    u_my = 0.5.*sign(llr_my)+0.5;
    
    [llr_app, llr_app_c] = appDecoder( ...
             zeros(length(y)/2,1),y_llr);
    u_app = 0.5.*sign(llr_app)+0.5;
    
    
    llr_vit = vitDecoder(y);
    
    
    % BER
    ber_arr(1,ww) = ber_arr(1,ww) + mean(u~=u_ref');
    ber_arr(2,ww) = ber_arr(2,ww) +  mean(u~=u_my');
    ber_arr(3,ww) = ber_arr(3,ww) + mean(u~=u_app);
end

semilogy(sigWdB, ber_arr(1,:), '-rs', 'LineWidth',1,'MarkerSize',10);
hold on;
semilogy(sigWdB, ber_arr(2,:), '--go', 'LineWidth',1,'MarkerSize',10);
hold on;
semilogy(sigWdB, ber_arr(3,:), '-.b^', 'LineWidth',1,'MarkerSize',10);


% tbdepth = 1;
% x_vit = vitdec(y,trellis,tbdepth,'trunc','soft');