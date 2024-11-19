function llr = bcjrAlg(y,trellis,sigW)
    % parameters for AWGN
    a = 1;
    P_uk = 1/2; % P(uk=+1)

    numStates = trellis.numStates;
    memSize = log2(trellis.numOutputSymbols);
    coderate = log2(trellis.numInputSymbols)/memSize;
    numData = length(y)*coderate;
    
    m_gamma = zeros(numData, numStates, numStates);
    m_alpha = zeros(numData+1, numStates);
    m_beta = zeros(numData+1, numStates);
    
    %% Compute gamma
    for k = 1:numData
        v_yk = y(k*memSize-memSize+1:k*memSize);
        for s = 1:numStates % previous state
            for ss = 1:numStates % next state
                [msg,in]=ismember(ss-1,trellis.nextStates(s,:));
                if msg==1
                    v_xk = binary(trellis.outputs(s, in),memSize);
                    P_y_x = prod(normpdf(v_yk-a*v_xk,0,sigW));
                    gamma = P_uk * P_y_x; % equal prob
                    m_gamma(k, ss, s) = gamma;
                end
            end
        end
    end
    
    %% compute alpha
    m_alpha(1,1) = 1; % start with state 0
    for k = 2:numData+1
        for ss = 1:numStates % for each previous state
            m_alpha(k,ss) = dot(m_alpha(k-1,:), ...
                reshape(m_gamma(k-1, ss,:),1,numStates));
        end
        m_alpha(k,:) = m_alpha(k,:) / sum(m_alpha(k,:)); % normalization
    end
    
    %% compute beta
    m_beta(numData+1,:) = (1/numStates).*ones(1,numStates);
    for k=numData+1:-1:2
        for s = 1:numStates % for each next state
            m_beta(k-1,s) = dot(m_beta(k,:), ...
                reshape(m_gamma(k-1,:,s),1,numStates));
        end
        m_beta(k-1,:) = m_beta(k-1,:)/sum(m_beta(k-1,:)); % normalization
    end
    
    %% compute llr
    llr = zeros(1, numData);
    for k=1:numData
        up = 0;
        down = 0;
        for s=1:numStates % previous state
            for ss=1:numStates % next state
                [msg,in]=ismember(ss-1,trellis.nextStates(s,:));
                if (msg==1 && in==1) % input=0
                    down = down + m_alpha(k,s)*m_gamma(k,ss,s)*m_beta(k+1,ss);
                elseif (msg==1 && in==2) % input=1
                    up = up + m_alpha(k,s)*m_gamma(k,ss,s)*m_beta(k+1,ss);
                end
            end
        end
        % normalization
        numer = up/(up+down);
        denom = down/(up+down);
        llr(k) = log(numer/denom);
    end
end
    