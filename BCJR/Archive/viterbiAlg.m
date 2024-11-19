% function v_uHat = viterbiAlg(y,trellis,sigW)
% function v_fXhat = viterbiAlg(m_fPriors, s_nConst, s_nMemSize)

% Apply Viterbi detection from computed priors
%
% Syntax
% -------------------------------------------------------
% v_fXhat = v_fViterbi(m_fPriors, s_nConst, s_nMemSize)
%
% INPUT:
% -------------------------------------------------------
% m_fPriors - evaluated likelihoods for each state at each time instance
% s_nConst - constellation size (positive integer)
% s_nMemSize - channel memory length
% 
%
% OUTPUT:
% -------------------------------------------------------
% v_fXhat - recovered symbols vector

y = [1 1 1 0 1 1 0 0 0 1 1 0];
sigW = 1;
trellis = struct('numInputSymbols',2,'numOutputSymbols',4, ...
'numStates',4,'nextStates',[0 2;0 2;1 3;1 3], ...
'outputs',[0 3;1 2;3 0;2 1]);

numStates = trellis.numStates;
memSize = log2(trellis.numOutputSymbols);
coderate = log2(trellis.numInputSymbols)/memSize;
numData = length(y)*coderate;

m_fPriors = zeros(numData,numStates);
s_nConst = trellis.numInputSymbols;
% Compute coditional PDF for each state
for ss=1:numStates
    m_y = reshape(y, log2(trells.numOutputSymbols), numData)';
    v_x = binary(ss,2);
    m_fPriors(:,ss) = prod(normpdf(m_y-a*v_x,0,sigW));
end

v_fXhat = zeros(1, numData);
% Apply Viterbi 
% cost = hamming distance
m_fCost = zeros(numData, numStates*s_nConst)+Inf;
% m_fCost = -log(m_fPriors); % soft cost

v_fCtilde = zeros(numStates,1)+Inf;
v_fCtilde(1,1) = 0;

for kk=1:numData
    m_fCtildeNext = zeros(numStates,1)+Inf;
    for ss=1:numStates % next state
        v_fTemp = zeros(s_nConst,1);
        for s=1:numStates % previous state
            [msg,in] = ismember(ss-1,trellis.nextStates(s,:));
            if msg == 1
                v_fTemp()
            end
            v_fTemp(ll) = v_fCtilde(trellis.nextStates(ss,ll)) + m_fCost(kk,ss);
            m_fCtildeNext(ss) = min(v_fTemp);
        end
    end
        
    v_fCtilde = m_fCtildeNext;
    [~, I] = min(v_fCtilde);
    % return index of first symbol in current state
    v_fXhat(kk) = mod(I-1,s_nConst)+1;
end