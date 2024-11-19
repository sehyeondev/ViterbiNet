errorRate = comm.ErrorRate( ...
    "ReceiveDelay",2 ...
    );
    % "Samples","Input port");
tx = [1 0 1 0 1 0 1 0 1 0]';
rx = tx;
rx(1) = ~rx(1);
rx(end) = ~rx(end);
y = errorRate(tx,rx);
