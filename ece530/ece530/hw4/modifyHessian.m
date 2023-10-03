
function HTilde = modifyHessian(H)
    epsilon = 1e-3 % A small value to make the values truly "greater than"
    n = size(H, 1);
    
    % Problem 4.1: Write your code here to compute D
    % using your answer in part (c).
    h_ii = diag(H);
    D_ii = zeros(size(h_ii));
    off_diag = H-diag(h_ii);
    for i = 1:n
        if sum(abs(off_diag(i,:))) > h_ii(i)
          D_ii(i) = sum(abs(off_diag(i,:))) - h_ii(i) + 1/2 + epsilon;
        end
    end

    D = diag(D_ii);
    
    HTilde = H + D;
end

