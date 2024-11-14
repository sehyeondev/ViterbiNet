function result = bin2array(bin)
    result = zeros(1, length(bin));
    for i = 1:length(bin)
        result(i) = 2*str2double(bin(i))-1;
    end
end