function [net, GMModel] = GetMyViterbiNet(v_fXtrain, v_fYtrain ,s_nConst, s_nMemSize, s_nMixtureSize)

% Generate and train a new ViterbiNet conditional distribution network
%
% Syntax
% -------------------------------------------------------
% [net, GMModel] = GetViterbiNet(m_fXtrain,v_fYtrain ,s_nConst)
%
% INPUT:
% -------------------------------------------------------
% v_fXtrain - training symobls vector
% v_fYtrain - training channel outputs (vector with training size entries)
% s_nConst - constellation size (positive integer)
% s_nMemSize - channel memory length
% s_nMixtureSize - finite mixture size for PDF estimator (positive integer)
%
%
% OUTPUT:
% -------------------------------------------------------
% net - trained neural network model
% GMModel - trained mixture model PDF estimate

% Reshape input symbols into a matrix representation
m_fXtrain = m_fMyReshape(v_fXtrain, s_nMemSize);
 

% Generate neural network
inputSize = 1;
numHiddenUnits = 100;
numClasses = s_nConst^s_nMemSize;

% Initialize Python interface
if count(py.sys.path, '') == 0
    insert(py.sys.path, int32(0), '');
end

% Convert MATLAB arrays to Python-compatible format
x_train_py = py.numpy.array(m_fXtrain);
y_train_py = py.numpy.array(v_fYtrain);

% Call Python training function
try
    interface = py.MatlabInterface();
    model_file = interface.train_viterbi_net(...
        x_train_py, ...
        y_train_py, ...
        int32(inputSize), ...
        int32(numHiddenUnits), ...
        int32(numClasses));
    
    % Load the trained model weights
    net = load(char(model_file));
catch e
    error('Error in Python execution: %s', e.message);
end

% Compute output PDF using GMM fitting
GMModel = fitgmdist(v_fYtrain',s_nMixtureSize,'RegularizationValue',0.1);
