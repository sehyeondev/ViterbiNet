% Define trellis for a convolutional code
trellis = poly2trellis(3,[6 7]);

% Define the APP decoder object using the BCJR algorithm
appDecoder = comm.APPDecoder('TrellisStructure', trellis, ...
                          'Algorithm', 'True APP', ...
                          'TerminationMethod', 'Terminated');


% Generate encoded data (example data)
dataBits = randi([0 1], 5000, 1);
encodedData = convenc(dataBits, trellis);

% Assume some noisy received data
receivedData = encodedData + 10 * randn(size(encodedData));

% Decode using the APP decoder
% decodedBits = appDecoder(receivedData);

tbdepth = 43;
decodedData = vitdec(encodedData,trellis,tbdepth,'trunc','hard');

err = biterr(dataBits,decodedData);