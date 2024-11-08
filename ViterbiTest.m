% Test script for Viterbi detector
% Test Case 1: Simple case with known output
disp('Running Test Case 1: Simple case');
s_nConst = 2;      % Binary constellation
s_nMemSize = 1;    % Memory length of 1
s_nDataSize = 5;   % 5 time instances

% Create some test priors (probabilities should sum to 1 for each time instance)
m_fPriors = [0.9 0.1;   % Strong probability of state 1
             0.1 0.9;   % Strong probability of state 2
             0.9 0.1;   % Strong probability of state 1
             0.1 0.9;   % Strong probability of state 2
             0.9 0.1];  % Strong probability of state 1

v_fXhat = v_fViterbi(m_fPriors, s_nConst, s_nMemSize);
disp('Expected output: [1 2 1 2 1]');
disp('Actual output:   ');
disp(v_fXhat);

% Test Case 2: Longer sequence
disp('Running Test Case 2: Longer sequence');
s_nDataSize = 10;
% Create alternating probabilities
m_fPriors = zeros(s_nDataSize, 2);
for i = 1:s_nDataSize
    if mod(i,2) == 1
        m_fPriors(i,:) = [0.9 0.1];
    else
        m_fPriors(i,:) = [0.1 0.9];
    end
end

v_fXhat = v_fViterbi(m_fPriors, s_nConst, s_nMemSize);
disp('Expected output: [1 2 1 2 1 2 1 2 1 2]');
disp('Actual output:   ');
disp(v_fXhat);

% Test Case 3: Larger constellation
disp('Running Test Case 3: Larger constellation');
s_nConst = 4;      % 4-symbol constellation
s_nMemSize = 2;    % Memory length of 2
s_nDataSize = 5;

% Create priors for 4-symbol constellation
m_fPriors = zeros(s_nDataSize, s_nConst^s_nMemSize);
% Set strong probabilities for a specific pattern
for i = 1:s_nDataSize
    m_fPriors(i,:) = ones(1,s_nConst^s_nMemSize) * 0.1;
    m_fPriors(i,mod(i,4)+1) = 0.7;  % Strong probability for symbol i mod 4
end

v_fXhat = v_fViterbi(m_fPriors, s_nConst, s_nMemSize);
disp('Expected pattern should follow 1,2,3,4,1');
disp('Actual output:   ');
disp(v_fXhat);

% Verification functions
verify_probability_sum(m_fPriors);
verify_output_range(v_fXhat, s_nConst);

function verify_probability_sum(m_fPriors)
    % Verify that probabilities sum to approximately 1 for each time instance
    sums = sum(m_fPriors, 2);
    if any(abs(sums - 1) > 1e-10)
        warning('Probabilities do not sum to 1 for all time instances');
        disp('Sums:');
        disp(sums);
    end
end

function verify_output_range(v_fXhat, s_nConst)
    % Verify that all outputs are within the valid range
    if any(v_fXhat < 1) || any(v_fXhat > s_nConst)
        warning('Output contains invalid symbol indices');
        disp('Range should be: 1 to ');
        disp(s_nConst);
        disp('Actual values:');
        disp(v_fXhat);
    end
end