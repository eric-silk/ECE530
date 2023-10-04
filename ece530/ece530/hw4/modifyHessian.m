
function HTilde = modifyHessian(H)
    n = size(H, 1);

    % Problem 4.1: Write your code here to compute D
    % using your answer in part (c).
    h_ii = diag(H);
    D_ii = zeros(size(h_ii));
    off_diag = H-diag(h_ii);

    for i = 1:n
        r = sum(abs(off_diag(i,:)));
        min_eigenvalue_estimate = h_ii(i) - r;
        if min_eigenvalue_estimate < (1/2)
            D_ii(i) = 1/2 + r - h_ii(i);
        end
    end

    D = diag(D_ii)
    HTilde = H + D;
end

