M = 2;
snr = 7;
frameLength = 300;
convEncoder = comm.ConvolutionalEncoder( ...
    'TerminationMethod','Truncated');

trellis = struct('numInputSymbols',2,'numOutputSymbols',4, ...
'numStates',4,'nextStates',[0 2;0 2;1 3;1 3], ...
'outputs',[0 3;3 0;2 1;1 2]);

appDecoder = comm.APPDecoder(...
    'TrellisStructure',trellis, ...
    'Algorithm','True APP', ...
    'CodedBitLLROutputPort',true);
errRate = comm.ErrorRate;

for counter = 1:5
     data = randi([0 1],frameLength,1);
     % encodedData = convEncoder(data);
     encodedData = convenc(data, trellis);
     psksig = pskmod(encodedData,M,InputType='bit');
     [rxsig,noisevar] = awgn(psksig,snr);
     demodsig = pskdemod(rxsig,M, ...
         OutputType='approxllr', ...
         NoiseVariance=noisevar);
     % The APP decoder assumes a polarization of the soft
     % inputs that is inverse to that of the demodulator 
     % soft outputs. Change the sign of demodulated signal.
     [receivedSoftBits, LUD] = appDecoder( ...
         zeros(frameLength,1),-demodsig);
     % Convert from soft-decision to hard-decision.
     receivedBits = double(receivedSoftBits > 0);
     % Count errors
     errorStats = errRate(data,receivedBits);
end

fprintf('Error rate = %f\nNumber of errors = %d\n', ...
     errorStats(1), errorStats(2))