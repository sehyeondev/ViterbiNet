% function for print matrix
function PrintMatrix(m_fMatrix)
    [s_nRows, s_nCols] = size(m_fMatrix);
    for ii=1:s_nRows
        for jj=1:s_nCols
            fprintf('%f ', m_fMatrix(ii,jj));
        end
        fprintf('\n');
    end
end